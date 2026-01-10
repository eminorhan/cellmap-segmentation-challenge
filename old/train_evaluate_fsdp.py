import os
import random
import time
import functools
import argparse
import glob
import re
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.v2 as T
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
)
from torch.utils.data import DataLoader, DistributedSampler

# Evaluation imports
import zarr
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import dice as scipy_dice
from skimage.measure import label as relabel
from pykdtree.kdtree import KDTree as cKDTree
from sklearn.metrics import accuracy_score, jaccard_score
from tqdm import tqdm
from upath import UPath
from tensorboardX import SummaryWriter

# Internal imports
from cellmap_data import CellMapDatasetWriter, CellMapImage
from cellmap_data.utils import (
    array_has_singleton_dim,
    is_array_2D,
    permute_singleton_dimension,
)
from cellmap_data.transforms.augment import NaNtoNum, Binarize

from cellmap_segmentation_challenge.utils import (
    CellMapLossWrapper,
    get_dataloader,
    load_safe_config,
    make_datasplit_csv,
    make_s3_datasplit_csv,
    format_string,
)

# Constants for evaluation
INSTANCE_CLASSES = ["nuc", "vim", "ves", "endo", "lyso", "ld", "perox", "mito", "np", "mt", "cell", "instance"]
HAUSDORFF_DISTANCE_MAX = np.inf
INSTANCE_RATIO_CUTOFF = 50.0

# Evaluation metrics
def compute_hausdorff_distance(image0, image1, voxel_size, max_distance):
    a_points = np.argwhere(image0)
    b_points = np.argwhere(image1)
    if len(a_points) == 0 and len(b_points) == 0: return 0
    elif len(a_points) == 0 or len(b_points) == 0: return np.inf
    a_points = a_points * np.array(voxel_size)
    b_points = b_points * np.array(voxel_size)
    a_tree = cKDTree(a_points)
    b_tree = cKDTree(b_points)
    fwd = a_tree.query(b_points, k=1)[0]
    bwd = b_tree.query(a_points, k=1)[0]
    fwd[fwd > max_distance] = max_distance
    bwd[bwd > max_distance] = max_distance
    return max(fwd.max(), bwd.max())

def score_instance(pred_label, truth_label, voxel_size):
    pred_label = relabel(pred_label, connectivity=pred_label.ndim)
    truth_ids = np.unique(truth_label); truth_ids = truth_ids[truth_ids != 0]
    pred_ids = np.unique(pred_label); pred_ids = pred_ids[pred_ids != 0]
    
    if len(truth_ids) > 0 and len(pred_ids) / len(truth_ids) > INSTANCE_RATIO_CUTOFF:
        return {"accuracy": 0, "hausdorff_distance": np.inf, "normalized_hausdorff_distance": 0, "combined_score": 0}

    truth_flat = truth_label.flatten()
    pred_flat = pred_label.flatten()
    matched_pred_label = np.zeros_like(pred_label)
    hausdorff_distances = []

    if len(pred_ids) > 0 and len(truth_ids) > 0:
        truth_masks = {tid: truth_flat == tid for tid in truth_ids}
        cost_matrix = np.zeros((len(truth_ids), len(pred_ids)))
        for j, pid in enumerate(pred_ids):
            pred_mask = pred_flat == pid
            overlapping_truth = np.unique(truth_flat[pred_mask])
            overlapping_truth = overlapping_truth[overlapping_truth != 0]
            for tid in overlapping_truth:
                i = np.where(truth_ids == tid)[0][0]
                truth_mask = truth_masks[tid]
                tp = np.sum(truth_mask & pred_mask)
                fp = np.sum((~truth_mask) & pred_mask)
                fn = np.sum(truth_mask & (~pred_mask))
                cost_matrix[i, j] = tp / (tp + fp + fn)

        row_inds, col_inds = linear_sum_assignment(cost_matrix, maximize=True)
        for i, j in zip(col_inds, row_inds):
            pred_mask = pred_label == pred_ids[i]
            matched_pred_label[pred_mask] = truth_ids[j]

        for tid in truth_ids:
            t_mask = truth_label == tid
            p_mask = matched_pred_label == tid
            if not np.any(t_mask) and not np.any(p_mask): continue
            h_dist = compute_hausdorff_distance(t_mask, p_mask, voxel_size, HAUSDORFF_DISTANCE_MAX)
            hausdorff_distances.append(h_dist)
    
    accuracy = accuracy_score(truth_flat, matched_pred_label.flatten())
    hausdorff_dist = np.mean(hausdorff_distances) if len(hausdorff_distances) > 0 else 0
    norm = np.linalg.norm(voxel_size)
    normalized_hausdorff_dist = 1.01 ** (-hausdorff_dist / norm)
    combined_score = (accuracy * normalized_hausdorff_dist) ** 0.5
    return {"accuracy": accuracy, "hausdorff_distance": hausdorff_dist, "normalized_hausdorff_distance": normalized_hausdorff_dist, "combined_score": combined_score}

def score_semantic(pred_label, truth_label):
    pred_bin = (pred_label > 0).flatten()
    truth_bin = (truth_label > 0).flatten()
    if np.sum(truth_bin) + np.sum(pred_bin) == 0: return {"iou": 1.0, "dice_score": 1.0}
    dice_score = 1 - scipy_dice(truth_bin, pred_bin)
    iou = jaccard_score(truth_bin, pred_bin, zero_division=1)
    return {"iou": iou, "dice_score": dice_score if not np.isnan(dice_score) else 1}

# Prediction methods
def consolidate_ranks(rank_path: str, final_path: str, is_rank0: bool):
    """Moves data chunks from the rank-specific temporary Zarr to the final Zarr."""
    rank_path = Path(rank_path)
    final_path = Path(final_path)
    for root, dirs, files in os.walk(rank_path):
        for file in files:
            if file.startswith(".") and not file.startswith(".z"): continue
            src_file = Path(root) / file
            rel_path = src_file.relative_to(rank_path)
            dest_file = final_path / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                if dest_file.exists(): os.remove(src_file)
                else: shutil.move(str(src_file), str(dest_file))
            except Exception as e: print(f"[Error] Failed to move {src_file}: {e}")
    try: shutil.rmtree(rank_path)
    except Exception: pass

def _predict_distributed(model: torch.nn.Module, dataset_writer_kwargs: dict, batch_size: int, input_array_info):
    """Core distributed prediction logic."""
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    is_rank0 = rank == 0

    value_transforms = T.Compose([
        T.ToDtype(torch.float, scale=True),
        T.Normalize(mean=[0.449,], std=[0.226,]),
        NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
    ])

    final_target_path = dataset_writer_kwargs["target_path"]
    rank_target_path = f"{final_target_path}_rank{rank}"
    
    # 1. Rank-Specific Writer
    rank_writer_kwargs = dataset_writer_kwargs.copy()
    rank_writer_kwargs["target_path"] = rank_target_path
    rank_writer_kwargs["overwrite"] = True

    # 2. Rank 0 Init Structure
    if is_rank0:
        _ = CellMapDatasetWriter(**dataset_writer_kwargs, raw_value_transforms=value_transforms)

    # 3. Local Writer
    dataset_writer = CellMapDatasetWriter(**rank_writer_kwargs, raw_value_transforms=value_transforms)

    sampler = DistributedSampler(dataset_writer.blocks, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset_writer.blocks, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True)

    model.eval()
    
    # Check for 2D vs 3D logic
    singleton_dim = np.where([s == 1 for s in dataset_writer_kwargs["input_arrays"]["input"]["shape"]])[0]
    singleton_dim = singleton_dim[0] if singleton_dim.size > 0 else None

    iterator = tqdm(dataloader, dynamic_ncols=True, desc="Predicting") if is_rank0 else dataloader

    with torch.no_grad():
        for batch in iterator:
            inputs = batch["input"].to(local_rank, non_blocking=True)
            if singleton_dim is not None: inputs = inputs.squeeze(dim=singleton_dim + 2)
            
            outputs = model(inputs)

            if input_array_info["shape"][0] == 1:
                outputs = torch.nn.functional.interpolate(input=outputs, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
            else:
                outputs = torch.nn.functional.interpolate(input=outputs, size=inputs.shape[-3:], mode="trilinear", align_corners=False)
            
            if singleton_dim is not None: outputs = outputs.unsqueeze(dim=singleton_dim + 2)
            
            outputs = outputs.float()
            dataset_writer[batch["idx"]] = {"output": outputs}

    dist.barrier() # Ensure all ranks finished writing
    consolidate_ranks(rank_target_path, final_target_path, is_rank0)
    dist.barrier() # Ensure consolidation finished before scoring

# Orthoplane prediction (separate prediction runs along x, y, z axes)  
def predict_orthoplanes(model, dataset_writer_kwargs, batch_size, input_array_info):
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    is_rank0 = rank == 0
    base_target_path = dataset_writer_kwargs["target_path"]
    shared_tmp_path = str(UPath(base_target_path).parent / f"temp_ortho_{UPath(base_target_path).name}")
    
    if is_rank0: os.makedirs(shared_tmp_path, exist_ok=True)
    dist.barrier(device_ids=[local_rank])

    for axis in range(3):
        if is_rank0: print(f"Processing Axis {axis}...")
        temp_kwargs = dataset_writer_kwargs.copy()
        temp_kwargs["target_path"] = os.path.join(shared_tmp_path, "output.zarr", str(axis))
        
        input_arrays = {k: v.copy() for k, v in temp_kwargs["input_arrays"].items()}
        target_arrays = {k: v.copy() for k, v in temp_kwargs["target_arrays"].items()}
        permute_singleton_dimension(input_arrays, axis)
        permute_singleton_dimension(target_arrays, axis)
        temp_kwargs["input_arrays"] = input_arrays
        temp_kwargs["target_arrays"] = target_arrays
        
        _predict_distributed(model, temp_kwargs, batch_size, input_array_info)

    if is_rank0: print("Combining predictions...")
    
    final_target_path = dataset_writer_kwargs["target_path"]
    rank_target_path = f"{final_target_path}_rank{rank}"
    
    rank_writer_kwargs = dataset_writer_kwargs.copy()
    rank_writer_kwargs["target_path"] = rank_target_path
    rank_writer_kwargs["overwrite"] = True
    
    if is_rank0: _ = CellMapDatasetWriter(**dataset_writer_kwargs)
    dataset_writer = CellMapDatasetWriter(**rank_writer_kwargs)

    single_axis_images = {
        array_name: {
            label: [
                CellMapImage(
                    os.path.join(shared_tmp_path, "output.zarr", str(axis), label),
                    target_class=label, target_scale=array_info["scale"], target_voxel_shape=array_info["shape"], pad=True, pad_value=0,
                ) for axis in range(3)
            ] for label in dataset_writer_kwargs["classes"]
        } for array_name, array_info in dataset_writer_kwargs["target_arrays"].items()
    }

    sampler = DistributedSampler(dataset_writer.blocks, num_replicas=dist.get_world_size(), rank=rank, shuffle=False, drop_last=False)
    tiled_loader = DataLoader(dataset_writer.blocks, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    iterator = tqdm(tiled_loader, dynamic_ncols=True) if is_rank0 else tiled_loader
    
    for batch in iterator:    
        outputs = {}
        for array_name, images in single_axis_images.items():
            outputs[array_name] = {}
            for label in dataset_writer_kwargs["classes"]:
                outputs[array_name][label] = []
                for idx in batch["idx"]:
                    avg_pred = torch.stack([img[dataset_writer.get_center(idx)] for img in images[label]]).mean(dim=0)
                    outputs[array_name][label].append(avg_pred)
                outputs[array_name][label] = torch.stack(outputs[array_name][label])
        dataset_writer[batch["idx"]] = outputs

    dist.barrier()
    consolidate_ranks(rank_target_path, final_target_path, is_rank0)
    
    if is_rank0:
        try: shutil.rmtree(shared_tmp_path)
        except Exception: pass

def run_validation_inference(model, datasplit_path, classes, input_array_info, target_array_info, batch_size, output_dir_base):
    """
    Orchestrates the validation inference using the distributed prediction logic.
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    is_rank0 = dist.get_rank() == 0

    val_rows = []
    with open(datasplit_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == "validate": val_rows.append(row)

    if not val_rows:
        if is_rank0: print("No validation rows found.")
        return

    do_orthoplanes = False # array_has_singleton_dim(input_array_info) or is_array_2D(input_array_info, summary=any)
    predict_func = predict_orthoplanes if do_orthoplanes else _predict_distributed
    
    if is_rank0: 
        print(f"Starting Validation Inference on {len(val_rows)} volumes.")
        print(f"Method: {'Orthoplanes' if do_orthoplanes else 'Standard 3D Blocks'}")

    scores = {}

    for row in val_rows:
        # row: split, raw_zarr, raw_ds, labels_zarr, labels_ds_list
        raw_full_path = os.path.join(row[1], row[2])
        labels_base_path = row[3]
        crop_name = os.path.basename(labels_base_path)
        
        # Temporary output path for predictions
        prediction_output_path = os.path.join(output_dir_base, "predictions", crop_name)
        
        input_arrays = {"input": input_array_info}
        target_arrays = {"output": target_array_info}
        
        # --- FIXED: Robust raw bounds reading (Handling Groups vs Arrays) ---
        try:
            ds = zarr.open(raw_full_path, mode='r')
            
            # 1. Handle OME-NGFF Multiscale Structure (Group -> Array)
            # If ds is a Group, we look for 's0' (CellMap/COSEM standard) or '0' (OME-NGFF standard)
            if isinstance(ds, zarr.hierarchy.Group):
                if 's0' in ds:
                    ds = ds['s0']
                elif '0' in ds:
                    ds = ds['0']
            
            # 2. Get Shape (Now safe even if we started with a Group)
            if not hasattr(ds, 'shape'):
                # If we still don't have a shape, it's a Group without standard sub-arrays
                raise ValueError(f"Object at {raw_full_path} is a Zarr Group without 's0' or '0' arrays.")
                
            shape = ds.shape
            attrs = ds.attrs

            # 3. Get Scale/Translation safely
            # Try standard keys first
            if "voxel_size" in attrs: scale = attrs["voxel_size"]
            elif "resolution" in attrs: scale = attrs["resolution"]
            elif "scale" in attrs: scale = attrs["scale"]
            else: scale = [8, 8, 8] # Fallback default
            
            if "translation" in attrs: translation = attrs["translation"]
            elif "offset" in attrs: translation = attrs["offset"]
            else: translation = [0, 0, 0]

            # Construct 3D bounding box: {axis: [min, max]}
            # Assuming standard ZYX order for 3D data
            bounds_3d = {}
            for i, axis in enumerate(["z", "y", "x"]):
                # Safety checks for dimension mismatches
                s = scale[i] if i < len(scale) else 1.0
                t = translation[i] if i < len(translation) else 0.0
                sh = shape[i] if i < len(shape) else 0
                
                start = t
                end = start + (sh * s)
                bounds_3d[axis] = [start, end]
            
            target_bounds = {"output": bounds_3d}
            
        except Exception as e:
            if is_rank0: print(f"Skipping {crop_name}, failed to read raw bounds: {e}")
            continue
        # ----------------------------------------------------------------------

        dataset_writer_kwargs = {
            "raw_path": raw_full_path,
            "target_path": prediction_output_path,
            "classes": classes,
            "input_arrays": input_arrays,
            "target_arrays": target_arrays,
            "target_bounds": target_bounds,
            "overwrite": True,
            "device": f"cuda:{local_rank}",
        }

        # RUN INFERENCE
        predict_func(model, dataset_writer_kwargs, batch_size, input_array_info)
        
        # SCORING (Rank 0 only, after barrier)
        dist.barrier()
        
        if is_rank0:
            print(f"Scoring {crop_name}...")
            scores[crop_name] = {}
            try:
                pred_zarr = zarr.open(prediction_output_path, mode='r')
                
                for class_name in classes:
                    # Clean up the label list string from CSV if present
                    if isinstance(row[4], str) and "[" in row[4]:
                        possible_labels = row[4].strip("[]").split(',')
                        possible_labels = [l.strip() for l in possible_labels]
                        if class_name not in possible_labels:
                            continue
                    
                    gt_path_direct = os.path.join(labels_base_path, class_name)
                    if not os.path.exists(gt_path_direct):
                        continue

                    # Load GT
                    gt_ds = zarr.open(gt_path_direct, mode='r')
                    gt_data = gt_ds[:]
                    voxel_size = gt_ds.attrs.get("voxel_size", [8.0, 8.0, 8.0])

                    # Load Pred
                    if class_name not in pred_zarr: continue
                    pred_prob = pred_zarr[class_name][:]
                    pred_mask = pred_prob > 0.5

                    # Masking (Optional)
                    mask_path = os.path.join(labels_base_path, f"{class_name}_mask")
                    if os.path.exists(mask_path):
                        mask = zarr.open(mask_path, mode='r')[:]
                        pred_mask = pred_mask * mask
                        gt_data = gt_data * mask

                    # Score
                    if class_name in INSTANCE_CLASSES:
                        pred_instances = relabel(pred_mask)
                        res = score_instance(pred_instances, gt_data, voxel_size)
                        scores[crop_name][class_name] = res
                    else:
                        res = score_semantic(pred_mask, gt_data)
                        scores[crop_name][class_name] = res
            except Exception as e:
                print(f"Error scoring {crop_name}: {e}")

    # Aggregate & Print
    if is_rank0:
        total_instance = []
        total_semantic = []
        print("\n" + "="*50 + "\nFINAL VALIDATION SCORES\n" + "="*50)
        for crop, label_res in scores.items():
            print(f"\n--- {crop} ---")
            for label, res in label_res.items():
                if label in INSTANCE_CLASSES:
                    s = res['combined_score']
                    total_instance.append(s)
                    print(f"{label:<10} (Instance): Combined={s:.4f} | Acc={res['accuracy']:.4f}")
                else:
                    s = res['iou']
                    total_semantic.append(s)
                    print(f"{label:<10} (Semantic): IoU={s:.4f} | Dice={res['dice_score']:.4f}")

        mean_inst = np.mean(total_instance) if total_instance else 0
        mean_sem = np.mean(total_semantic) if total_semantic else 0
        final = (mean_inst * mean_sem) ** 0.5
        print("\n" + "*"*30)
        print(f"Overall Instance: {mean_inst:.4f}")
        print(f"Overall Semantic: {mean_sem:.4f}")
        print(f"FINAL SCORE:      {final:.4f}")
        print("*"*30 + "\n")
        
        # Save JSON
        out_json = os.path.join(output_dir_base, "final_validation_scores.json")
        with open(out_json, "w") as f:
            json.dump(scores, f, indent=4, default=lambda x: float(x))
            
# Methods for training
def get_lr_lambda(current_step, warmup_steps, train_steps):
    if current_step < warmup_steps: return float(current_step) / float(max(1, warmup_steps))
    return max(0.0, float(train_steps - current_step) / float(max(1, train_steps - warmup_steps)))

def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank

def cleanup(): dist.destroy_process_group()

def find_latest_checkpoint(save_dir, model_name):
    if not os.path.exists(save_dir): return None
    pattern = re.compile(rf"{re.escape(model_name)}_(\d+)\.pth$")
    files = os.listdir(save_dir)
    checkpoints = []
    for f in files:
        match = pattern.match(f)
        if match: checkpoints.append((int(match.group(1)), os.path.join(save_dir, f)))
    if not checkpoints: return None
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]

def train(config_path: str):
    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_rank0 = rank == 0

    torch.backends.cudnn.benchmark = True
    config = load_safe_config(config_path)

    # Config Vars
    base_experiment_path = getattr(config, "base_experiment_path", UPath(config_path).parent)
    base_experiment_path = UPath(base_experiment_path)
    model_save_path_template = getattr(config, "model_save_path", (base_experiment_path / "checkpoints" / "{model_name}_{epoch}.pth").path)
    logs_save_path = getattr(config, "logs_save_path", (base_experiment_path / "tensorboard" / "{model_name}").path)
    datasplit_path = getattr(config, "datasplit_path", (base_experiment_path / "datasplit.csv").path)
    
    validation_prob = getattr(config, "validation_prob", 0.1)
    learning_rate = getattr(config, "learning_rate", 0.0001)
    batch_size = getattr(config, "batch_size", 8)
    filter_by_scale = getattr(config, "filter_by_scale", False)
    input_array_info = getattr(config, "input_array_info", {"shape": (1, 128, 128), "scale": (8, 8, 8)})
    target_array_info = getattr(config, "target_array_info", input_array_info)
    epochs = getattr(config, "epochs", 1000)
    iterations_per_epoch = getattr(config, "iterations_per_epoch", 1000)
    warmup_steps = getattr(config, "warmup_steps", 100)
    random_seed = getattr(config, "random_seed", 1)
    classes = getattr(config, "classes", ["nuc", "er"])
    model_name = getattr(config, "model_name", "2d_unet")
    model_config_obj = getattr(config, "model", None) 
    spatial_transforms = getattr(config, "spatial_transforms", {})
    validation_time_limit = getattr(config, "validation_time_limit", None)
    validation_batch_limit = getattr(config, "validation_batch_limit", None)
    use_s3 = getattr(config, "use_s3", False)
    use_mutual_exclusion = getattr(config, "use_mutual_exclusion", False)
    train_raw_value_transforms = getattr(config, "train_raw_value_transforms", T.Compose([T.ToDtype(torch.float, scale=True), T.Normalize(mean=[0.449,], std=[0.226,]), NaNtoNum({"nan": 0, "posinf": None, "neginf": None})]))
    val_raw_value_transforms = getattr(config, "val_raw_value_transforms", train_raw_value_transforms)
    target_value_transforms = getattr(config, "target_value_transforms", T.Compose([T.ToDtype(torch.float), Binarize()]))
    max_grad_norm = getattr(config, "max_grad_norm", 1.0)
    force_all_classes = getattr(config, "force_all_classes", "validate")
    log_steps = getattr(config, "log_steps", 10)
    
    # Loss
    criterion_cls = getattr(config, "criterion", torch.nn.BCEWithLogitsLoss)
    criterion_kwargs = getattr(config, "criterion_kwargs", {})
    weight_loss = getattr(config, "weight_loss", True)
    gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)

    save_dir = os.path.dirname(model_save_path_template)
    if is_rank0:
        if len(save_dir) > 0:
            os.makedirs(save_dir, exist_ok=True)
        if len(os.path.dirname(logs_save_path)) > 0:
            os.makedirs(os.path.dirname(logs_save_path), exist_ok=True)
        if len(os.path.dirname(datasplit_path)) > 0:
            os.makedirs(os.path.dirname(datasplit_path), exist_ok=True)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    # Datasplit
    if is_rank0 and not os.path.exists(datasplit_path):
        if filter_by_scale is not False:
            if filter_by_scale is not True:
                scale = filter_by_scale
                if isinstance(scale, (int, float)): scale = (scale, scale, scale)
            elif "scale" in input_array_info: scale = input_array_info["scale"]
            else: scale = None
        else: scale = None
        
        make_datasplit_csv(classes=classes, scale=scale, csv_path=datasplit_path, validation_prob=validation_prob, force_all_classes=force_all_classes)
    
    dist.barrier(device_ids=[local_rank])

    # Loader
    train_loader, val_loader = get_dataloader(
        datasplit_path=datasplit_path,
        classes=classes,
        batch_size=batch_size, 
        input_array_info=input_array_info,
        target_array_info=target_array_info,
        spatial_transforms=spatial_transforms,
        iterations_per_epoch=iterations_per_epoch,
        random_validation=validation_time_limit or validation_batch_limit,
        device=f"cuda:{local_rank}", 
        weighted_sampler=False,
        use_mutual_exclusion=use_mutual_exclusion,
        train_raw_value_transforms=train_raw_value_transforms,
        val_raw_value_transforms=val_raw_value_transforms,
        target_value_transforms=target_value_transforms,
    )
    
    train_loader.sampler = DistributedSampler(train_loader.dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    if val_loader:
        val_loader.sampler = DistributedSampler(val_loader.dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        val_loader.refresh()

    # FSDP setup (training)
    model = model_config_obj.to(local_rank)
    mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    model = FSDP(model, device_id=local_rank, mixed_precision=mp_policy)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: get_lr_lambda(s, warmup_steps, epochs*iterations_per_epoch))

    # Resume
    start_epoch = 1
    n_iter = 0
    resume_path = find_latest_checkpoint(save_dir, model_name)
    if resume_path:
        if is_rank0: print(f"Resuming: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            sharded_osd = FSDP.scatter_full_optim_state_dict(checkpoint["optimizer_state_dict"], model)
            optimizer.load_state_dict(sharded_osd)
        if "scheduler_state_dict" in checkpoint: scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        n_iter = checkpoint.get("step", (start_epoch - 1) * iterations_per_epoch)
        del checkpoint

    # Deduce spatial dims
    if "shape" in target_array_info:
        spatial_dims = sum([s > 1 for s in target_array_info["shape"]])
    else:
        spatial_dims = sum([s > 1 for s in list(target_array_info.values())[0]["shape"]])

    # Loss
    if weight_loss:
        pos_weight = list(train_loader.dataset.class_weights.values())
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(local_rank).flatten()
        pos_weight = pos_weight[:, None, None]
        if spatial_dims == 3:
            pos_weight = pos_weight[..., None]
        criterion_kwargs["pos_weight"] = pos_weight

    criterion = CellMapLossWrapper(criterion_cls, **criterion_kwargs)

    writer = SummaryWriter(format_string(logs_save_path, {"model_name": model_name})) if is_rank0 else None

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        train_loader.refresh()
        model.train()
        loader_iter = iter(train_loader.loader)
        running_loss = 0.0
        steps_in_log = 0

        for epoch_iter in range(iterations_per_epoch):
            try: batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader.loader); batch = next(loader_iter)
            n_iter += 1

            # Determine inputs/targets
            input_keys = list(train_loader.dataset.input_arrays.keys())
            target_keys = list(train_loader.dataset.target_arrays.keys())
            inputs = {k: batch[k] for k in input_keys} if len(input_keys) > 1 else batch[input_keys[0]]
            targets = {k: batch[k] for k in target_keys} if len(target_keys) > 1 else batch[target_keys[0]]

            is_accumulating = (epoch_iter + 1) % gradient_accumulation_steps != 0
            with model.no_sync() if is_accumulating else torch.enable_grad():
                outputs = model(inputs)
                if input_array_info["shape"][0] == 1:
                    outputs = torch.nn.functional.interpolate(outputs, size=targets.shape[-2:], mode="bilinear", align_corners=False)
                else:
                    outputs = torch.nn.functional.interpolate(outputs, size=targets.shape[-3:], mode="trilinear", align_corners=False)
                loss = criterion(outputs, targets) / gradient_accumulation_steps
                loss.backward()

            running_loss += loss.item() * gradient_accumulation_steps
            steps_in_log += 1

            if not is_accumulating:
                if max_grad_norm: model.clip_grad_norm_(max_grad_norm)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

            if is_rank0 and steps_in_log >= log_steps:
                avg_loss = running_loss / steps_in_log
                print(f"Epoch {epoch} | Step {n_iter} | Loss: {avg_loss:.6f} | lr: {scheduler.get_last_lr()[0]:.6f}")
                writer.add_scalar("loss", avg_loss, n_iter)
                running_loss = 0.0; steps_in_log = 0

        # Save checkpoint
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_model_state = model.state_dict()
            cpu_optim_state = FSDP.full_optim_state_dict(model, optimizer)
            if is_rank0:
                save_dict = {"epoch": epoch, "step": n_iter, "model_state_dict": cpu_model_state, "optimizer_state_dict": cpu_optim_state, "scheduler_state_dict": scheduler.state_dict()}
                save_path = format_string(model_save_path_template, {"epoch": epoch, "model_name": model_name})
                torch.save(save_dict, save_path)
                print(f"Saved: {save_path}")

        dist.barrier(device_ids=[local_rank])
        
        # Validation (loss only)
        if len(val_loader.loader) > 0:
            val_loss = 0.0; val_steps = 0
            model.eval()
            with torch.no_grad():
                for batch in val_loader.loader:

                    inputs = {k: batch[k] for k in input_keys} if len(input_keys) > 1 else batch[input_keys[0]]
                    targets = {k: batch[k] for k in target_keys} if len(target_keys) > 1 else batch[target_keys[0]]
                    
                    outputs = model(inputs)
                    
                    if input_array_info["shape"][0] == 1: 
                        outputs = torch.nn.functional.interpolate(outputs, size=targets.shape[-2:], mode="bilinear", align_corners=False)
                    else: 
                        outputs = torch.nn.functional.interpolate(outputs, size=targets.shape[-3:], mode="trilinear", align_corners=False)
                    
                    val_loss += criterion(outputs, targets)
                    val_steps += 1
                    
                    if validation_batch_limit and val_steps >= validation_batch_limit: 
                        break
            
            total_val_loss = torch.tensor(val_loss, device=local_rank)
            total_val_steps = torch.tensor(val_steps, device=local_rank)
            dist.all_reduce(total_val_loss)
            dist.all_reduce(total_val_steps)

            if is_rank0: 
                writer.add_scalar("validation", (total_val_loss/total_val_steps).item(), n_iter)
                print(f"Epoch {epoch} Validation Loss: {(total_val_loss/total_val_steps).item():.6f}")

    # ==========================================================
    # Final inference & scoring on validation crops (distributed)
    # ==========================================================
    print("Training finished. Starting Final Evaluation...")
    
    # # 1. Load the best/latest checkpoint model onto CPU (clean slate)
    # inference_model = model_config_obj # The original unwrapped model config object
    # resume_path = find_latest_checkpoint(save_dir, model_name)
    # if resume_path:
    #     checkpoint = torch.load(resume_path, map_location="cpu")
    #     inference_model.load_state_dict(checkpoint["model_state_dict"])
    
    # inference_model.to(local_rank)
    
    # # 2. Wrap in FSDP with NO_SHARD (replicates model on all GPUs for inference)
    # inference_model = FSDP(
    #     inference_model,
    #     device_id=local_rank,
    #     mixed_precision=mp_policy,
    #     sharding_strategy=ShardingStrategy.NO_SHARD
    # )

    # 3. Run Inference
    run_validation_inference(
        model,
        datasplit_path,
        classes,
        input_array_info,
        target_array_info,
        batch_size * 2, # Can usually fit larger batches for inference
        base_experiment_path.path
    )

    if is_rank0: writer.close()
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()
    train(args.config_path)