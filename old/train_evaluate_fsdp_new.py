import os
import random
import time
import functools
import argparse
import glob
import re
import csv
import shutil
import tempfile
import json
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
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DistributedSampler

from cellmap_data import CellMapImage  # <--- Added Import
from cellmap_data.utils import get_fig_dict, longest_common_substring
from cellmap_data.transforms.augment import NaNtoNum, Binarize
from tensorboardX import SummaryWriter
from upath import UPath
import matplotlib.pyplot as plt

# Internal imports
from cellmap_segmentation_challenge.utils import (
    CellMapLossWrapper,
    get_dataloader,
    load_safe_config,
    make_datasplit_csv,
    make_s3_datasplit_csv,
    format_string,
)

# Imports for Prediction and Evaluation
from cellmap_segmentation_challenge.predict_fsdp import _predict
from cellmap_segmentation_challenge.evaluate import (
    score_label,
    combine_scores,
    INSTANCE_CLASSES,
)


# LR Schedule Lambda function
def get_lr_lambda(current_step, warmup_steps, train_steps):
    if current_step < warmup_steps:
        # Linear warm-up
        return float(current_step) / float(max(1, warmup_steps))
    # Linear decay
    return max(0.0, float(train_steps - current_step) / float(max(1, train_steps - warmup_steps)))


def setup():
    """Initialize the distributed process group."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank


def cleanup():
    """Destroy the distributed process group."""
    dist.destroy_process_group()


def find_latest_checkpoint(save_dir, model_name):
    """
    Finds the latest checkpoint in save_dir matching the model_name pattern.
    Assumes naming convention: {model_name}_{epoch}.pth
    Returns path to the latest checkpoint or None.
    """
    if not os.path.exists(save_dir):
        return None
        
    # Pattern matching files like: mymodel_1.pth, mymodel_100.pth
    # We use a regex to extract the epoch number safely
    pattern = re.compile(rf"{re.escape(model_name)}_(\d+)\.pth$")
    
    files = os.listdir(save_dir)
    checkpoints = []
    
    for f in files:
        match = pattern.match(f)
        if match:
            epoch_num = int(match.group(1))
            checkpoints.append((epoch_num, os.path.join(save_dir, f)))
            
    if not checkpoints:
        return None
        
    # Sort by epoch number descending
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1] # Return path of highest epoch


def run_final_validation(
    model,
    datasplit_path,
    input_array_info,
    target_array_info,
    classes,
    local_rank,
    batch_size,
    output_dir=None
):
    """
    Runs inference on the validation set defined in datasplit.csv and scores the results.
    """
    is_rank0 = dist.get_rank() == 0
    
    if is_rank0:
        print("\n" + "="*40)
        print("STARTING FINAL VALIDATION SCORING")
        print("="*40)

    # 1. Parse datasplit.csv to find validation volumes
    val_entries = []
    if os.path.exists(datasplit_path):
        with open(datasplit_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 5 and row[0] == "validate":
                    # Row format: split, raw_zarr, raw_ds, gt_zarr, gt_ds_with_classes
                    raw_zarr, raw_ds = row[1], row[2]
                    gt_zarr, gt_ds_raw = row[3], row[4]
                    
                    if "[" in gt_ds_raw:
                        gt_ds = gt_ds_raw.split("[")[0].rstrip("/")
                    else:
                        gt_ds = gt_ds_raw
                    
                    crop_name = os.path.basename(gt_ds)
                    if not crop_name.startswith("crop"):
                        crop_name = f"val_{len(val_entries)}"

                    val_entries.append({
                        "raw_path": str(UPath(raw_zarr) / raw_ds),
                        "gt_path": str(UPath(gt_zarr) / gt_ds),
                        "gt_root": str(UPath(gt_zarr) / os.path.dirname(gt_ds)), 
                        "crop_name": crop_name
                    })

    if not val_entries:
        if is_rank0:
            print("No validation entries found in datasplit.csv. Skipping scoring.")
        return

    # 2. Setup Temporary Directory for Predictions
    if output_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="fsdp_val_preds_", dir=os.getcwd())
        temp_pred_path = temp_dir_obj.name
    else:
        temp_pred_path = output_dir
        temp_dir_obj = None

    if is_rank0:
        print(f"Generating validation predictions in: {temp_pred_path}")

    if is_rank0:
        os.makedirs(temp_pred_path, exist_ok=True)
    dist.barrier(device_ids=[local_rank])

    # 3. Run Inference
    input_arrays = {"input": input_array_info}
    target_arrays = {"output": target_array_info} 

    for entry in val_entries:
        crop_pred_path = os.path.join(temp_pred_path, entry["crop_name"])
        
        if is_rank0:
            print(f"Predicting {entry['crop_name']}...")
        
        # --- FIX START: Calculate target_bounds from Ground Truth ---
        target_bounds = None
        # Try finding a class that exists in the GT to reference bounds from
        for cls in classes:
            ref_path = UPath(entry["gt_path"]) / cls
            if ref_path.exists():
                try:
                    # We instantiate CellMapImage to get the bounding box consistent with the requested scale
                    ref_img = CellMapImage(
                        str(ref_path),
                        target_class=cls,
                        target_scale=input_array_info.get("scale", None),
                        target_voxel_shape=input_array_info.get("shape", None)
                    )
                    target_bounds = {"output": ref_img.bounding_box}
                    break
                except Exception:
                    continue
        
        if target_bounds is None:
             if is_rank0: 
                 print(f"[Warning] Skipping {entry['crop_name']}: Could not find any ground truth class {classes} at {entry['gt_path']} to calculate bounds.")
             continue
        # --- FIX END ---

        dataset_writer_kwargs = {
            "raw_path": entry["raw_path"],
            "target_path": crop_pred_path,
            "classes": classes,
            "input_arrays": input_arrays,
            "target_arrays": target_arrays,
            "target_bounds": target_bounds,  # <--- Now passing target_bounds
            "overwrite": True,
            "device": f"cuda:{local_rank}",
        }
        
        # Run distributed prediction
        _predict(model, dataset_writer_kwargs, batch_size, input_array_info)
        
        dist.barrier(device_ids=[local_rank])

    # 4. Score Predictions (Rank 0 Only)
    if is_rank0:
        print("Predictions complete. Calculating metrics...")
        scores = {}
        
        for entry in val_entries:
            crop_name = entry["crop_name"]
            pred_volume_path = UPath(temp_pred_path) / crop_name / "output"
            truth_root = UPath(entry["gt_root"])

            crop_scores = {}
            for label in classes:
                try:
                    _, _, result = score_label(
                        pred_label_path=pred_volume_path / label,
                        label_name=label,
                        crop_name=crop_name,
                        truth_path=truth_root,
                        instance_classes=INSTANCE_CLASSES
                    )
                    crop_scores[label] = result
                except Exception as e:
                    # Silence errors for missing labels (common in sparsely annotated crops)
                    pass
            
            if crop_scores:
                scores[crop_name] = crop_scores

        # 5. Combine and Print Scores
        if scores:
            try:
                final_scores = combine_scores(scores, include_missing=False, instance_classes=INSTANCE_CLASSES)
                
                print("\n" + "="*40)
                print("FINAL VALIDATION RESULTS")
                print("="*40)
                print(f"Overall Instance Score: {final_scores.get('overall_instance_score', 0):.4f}")
                print(f"Overall Semantic Score: {final_scores.get('overall_semantic_score', 0):.4f}")
                print(f"Overall Geometric Mean: {final_scores.get('overall_score', 0):.4f}")
                print("-" * 40)
                
                if "label_scores" in final_scores:
                    print("Per Class Scores:")
                    print(json.dumps(final_scores["label_scores"], indent=2))
            except Exception as e:
                print(f"Error combining scores: {e}")
        else:
            print("No scores computed.")

    # 6. Cleanup
    dist.barrier(device_ids=[local_rank])
    if is_rank0 and temp_dir_obj:
        print("Cleaning up temporary prediction files...")
        temp_dir_obj.cleanup()


def train(config_path: str):
    """
    Train a model using FSDP. Resumes from the latest checkpoint if available.
    """
    # Initialize Distributed Environment
    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    is_rank0 = rank == 0

    torch.backends.cudnn.benchmark = True

    # %% Load the configuration file
    config = load_safe_config(config_path)

    # %% Set hyperparameters
    base_experiment_path = getattr(config, "base_experiment_path", UPath(config_path).parent)
    base_experiment_path = UPath(base_experiment_path)
    
    # Define paths
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
    model = getattr(config, "model", None)
    spatial_transforms = getattr(config, "spatial_transforms", {"mirror": {"axes": {"x": 0.5, "y": 0.5}}, "transpose": {"axes": ["x", "y"]}, "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}}})
    validation_time_limit = getattr(config, "validation_time_limit", None)
    validation_batch_limit = getattr(config, "validation_batch_limit", None)
    use_s3 = getattr(config, "use_s3", False)
    use_mutual_exclusion = getattr(config, "use_mutual_exclusion", False)
    train_raw_value_transforms = getattr(config, "train_raw_value_transforms", T.Compose([T.ToDtype(torch.float, scale=True), T.Normalize(mean=[0.449,], std=[0.226,]), NaNtoNum({"nan": 0, "posinf": None, "neginf": None})]))
    val_raw_value_transforms = getattr(config, "val_raw_value_transforms", T.Compose([T.ToDtype(torch.float, scale=True), T.Normalize(mean=[0.449,], std=[0.226,]), NaNtoNum({"nan": 0, "posinf": None, "neginf": None})]))
    target_value_transforms = getattr(config, "target_value_transforms", T.Compose([T.ToDtype(torch.float), Binarize()]))
    max_grad_norm = getattr(config, "max_grad_norm", 1.0)
    force_all_classes = getattr(config, "force_all_classes", "validate")
    log_steps = getattr(config, "log_steps", 10)

    # %% Define the loss function
    criterion_cls = getattr(config, "criterion", torch.nn.BCEWithLogitsLoss)
    criterion_kwargs = getattr(config, "criterion_kwargs", {})
    weight_loss = getattr(config, "weight_loss", True)

    gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
    if gradient_accumulation_steps < 1:
        raise ValueError(f"gradient_accumulation_steps must be >= 1, but got {gradient_accumulation_steps}")

    # %% Make sure the save path exists (Rank 0 only)
    save_dir = os.path.dirname(model_save_path_template)
    if is_rank0:
        if len(save_dir) > 0:
            os.makedirs(save_dir, exist_ok=True)
        if len(os.path.dirname(logs_save_path)) > 0:
            os.makedirs(os.path.dirname(logs_save_path), exist_ok=True)
        if len(os.path.dirname(datasplit_path)) > 0:
            os.makedirs(os.path.dirname(datasplit_path), exist_ok=True)

    # %% Set the random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    if is_rank0:
        print(f"Training on {world_size} GPUs using FSDP.")

    # %% Make the datasplit file if it doesn't exist
    if is_rank0:
        if not os.path.exists(datasplit_path):
            if filter_by_scale is not False:
                if filter_by_scale is not True:
                    scale = filter_by_scale
                    if isinstance(scale, (int, float)):
                        scale = (scale, scale, scale)
                elif "scale" in input_array_info:
                    scale = input_array_info["scale"]
                else:
                    highest_res = [np.inf, np.inf, np.inf]
                    for key, info in input_array_info.items():
                        if "scale" in info:
                            res = np.prod(info["scale"])
                            if res < np.prod(highest_res):
                                highest_res = info["scale"]
                    scale = highest_res
            else:
                scale = None
            
            if use_s3:
                make_s3_datasplit_csv(
                    classes=classes,
                    scale=scale,
                    csv_path=datasplit_path,
                    validation_prob=validation_prob,
                    force_all_classes=force_all_classes,
                )
            else:
                make_datasplit_csv(
                    classes=classes,
                    scale=scale,
                    csv_path=datasplit_path,
                    validation_prob=validation_prob,
                    force_all_classes=force_all_classes,
                )
    
    dist.barrier(device_ids=[local_rank])

    # %% Data Loading
    dataloader_kwargs= {"pin_memory": False}  # cellmap-data puts data on GPU
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
        **dataloader_kwargs,
    )

    train_dataset = train_loader.dataset
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    train_loader.sampler = train_sampler
    
    if val_loader is not None:
        val_dataset = val_loader.dataset
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
        val_loader.sampler = val_sampler
        val_loader.refresh()

    # Move to Device
    model = model.to(local_rank)

    # Wrap in FSDP
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32, 
        buffer_dtype=torch.float32,
    )

    model = FSDP(
        model,
        device_id=local_rank,
        mixed_precision=mp_policy,
    )

    if is_rank0:
        print(f"Model Wrapped in FSDP: {type(model)}")

    # %% Optimizer
    opt_cls = torch.optim.AdamW
    opt_kwargs = {"lr": learning_rate}
    optimizer = opt_cls(model.parameters(), **opt_kwargs)
    
    total_train_steps = epochs * iterations_per_epoch
    lr_lambda = lambda step: get_lr_lambda(step, warmup_steps, total_train_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # %% Resume Logic
    start_epoch = 1
    n_iter = 0

    # Determine if we have a checkpoint to resume from
    resume_path = find_latest_checkpoint(save_dir, model_name)
    
    if resume_path:
        if is_rank0:
            print(f"Resuming training from checkpoint: {resume_path}")
        
        # Load checkpoint on CPU to avoid VRAM spikes
        checkpoint = torch.load(resume_path, map_location="cpu")
        
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(checkpoint["model_state_dict"])
        
        if "optimizer_state_dict" in checkpoint:
            full_osd = checkpoint["optimizer_state_dict"]
            sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)
            optimizer.load_state_dict(sharded_osd)
            del full_osd 
        elif is_rank0:
            print("Warning: Optimizer state not found in checkpoint. Optimizer initialized from scratch.")

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        else:
            match = re.search(r"_(\d+)\.pth$", resume_path)
            if match:
                start_epoch = int(match.group(1)) + 1

        n_iter = checkpoint.get("step", (start_epoch - 1) * iterations_per_epoch)

        del checkpoint
        torch.cuda.empty_cache()
    else:
        if is_rank0:
            print("No checkpoints found in model_save_path. Starting training from scratch.")

    # Deduce spatial dims
    if "shape" in target_array_info:
        spatial_dims = sum([s > 1 for s in target_array_info["shape"]])
    else:
        spatial_dims = sum([s > 1 for s in list(target_array_info.values())[0]["shape"]])

    # Loss Setup
    if weight_loss:
        pos_weight = list(train_loader.dataset.class_weights.values())
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(local_rank).flatten()
        pos_weight = pos_weight[:, None, None]
        if spatial_dims == 3:
            pos_weight = pos_weight[..., None]
        criterion_kwargs["pos_weight"] = pos_weight
    
    criterion = CellMapLossWrapper(criterion_cls, **criterion_kwargs)

    input_keys = list(train_loader.dataset.input_arrays.keys())
    target_keys = list(train_loader.dataset.target_arrays.keys())

    # %% Tensorboard (Rank 0 only)
    writer = None
    if is_rank0:
        writer = SummaryWriter(format_string(logs_save_path, {"model_name": model_name}))

    # %% Training Loop
    epochs_rng = np.arange(start_epoch, epochs + 1)
    
    if is_rank0 and len(epochs_rng) > 0:
        print(f"Training from Epoch {start_epoch} to {epochs}")

    for epoch in epochs_rng:
        train_sampler.set_epoch(int(epoch))
        train_loader.refresh()

        model.train()
        post_fix_dict = {}
        post_fix_dict["Epoch"] = epoch

        loader_iter = iter(train_loader.loader)
        
        running_loss = 0.0
        steps_in_log = 0

        optimizer.zero_grad()

        for epoch_iter in range(iterations_per_epoch):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader.loader)
                batch = next(loader_iter)

            n_iter += 1

            if len(input_keys) > 1:
                inputs = {key: batch[key] for key in input_keys}
            else:
                inputs = batch[input_keys[0]]
            
            if len(target_keys) > 1:
                targets = {key: batch[key] for key in target_keys}
            else:
                targets = batch[target_keys[0]]

            is_accumulating = (epoch_iter + 1) % gradient_accumulation_steps != 0
            context = model.no_sync() if is_accumulating else torch.enable_grad()

            with context:
                outputs = model(inputs)

                if input_array_info["shape"][0] == 1:
                    outputs = torch.nn.functional.interpolate(input=outputs, size=targets.shape[-2:], mode="bilinear", align_corners=False)
                else:
                    outputs = torch.nn.functional.interpolate(input=outputs, size=targets.shape[-3:], mode="trilinear", align_corners=False)

                loss = criterion(outputs, targets) / gradient_accumulation_steps
                loss.backward()
            
            running_loss += loss.item() * gradient_accumulation_steps
            steps_in_log += 1

            if not is_accumulating:
                if max_grad_norm is not None:
                    model.clip_grad_norm_(max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if is_rank0 and steps_in_log >= log_steps:
                avg_loss = running_loss / steps_in_log
                print(f"Epoch {epoch} | Step {n_iter} | Loss: {avg_loss:.6f} | lr: {scheduler.get_last_lr()[0]:.6f}")
                
                writer.add_scalar("loss", avg_loss, n_iter)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], n_iter)
                
                running_loss = 0.0
                steps_in_log = 0

        # %% Save Checkpoint
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_model_state = model.state_dict()
            cpu_optim_state = FSDP.full_optim_state_dict(model, optimizer)
            
            if is_rank0:
                save_dict = {
                    "epoch": int(epoch),
                    "step": n_iter,
                    "model_state_dict": cpu_model_state,
                    "optimizer_state_dict": cpu_optim_state,
                    "scheduler_state_dict": scheduler.state_dict(),
                    "model_name": model_name
                }
                save_path = format_string(model_save_path_template, {"epoch": epoch, "model_name": model_name})
                torch.save(save_dict, save_path)
                print(f"Saved checkpoint to {save_path}")

        dist.barrier(device_ids=[local_rank])

        # %% Validation
        if len(val_loader.loader) > 0:
            val_loss_accum = torch.zeros(1, device=local_rank)
            val_batches_accum = torch.zeros(1, device=local_rank)
            
            val_loader.refresh()
            
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            model.eval()

            start_time = time.time()
            i = 0
            
            with torch.no_grad():
                for batch in val_loader.loader:
                    if len(input_keys) > 1:
                        inputs = {key: batch[key] for key in input_keys}
                    else:
                        inputs = batch[input_keys[0]]
                    
                    if len(target_keys) > 1:
                        targets = {key: batch[key] for key in target_keys}
                    else:
                        targets = batch[target_keys[0]]

                    outputs = model(inputs)

                    if input_array_info["shape"][0] == 1:
                        outputs = torch.nn.functional.interpolate(input=outputs, size=targets.shape[-2:], mode="bilinear", align_corners=False)
                    else:
                        outputs = torch.nn.functional.interpolate(input=outputs, size=targets.shape[-3:], mode="trilinear", align_corners=False)

                    loss = criterion(outputs, targets)
                    
                    val_loss_accum += loss
                    val_batches_accum += 1
                    i += 1
                    
                    if validation_time_limit and (time.time() - start_time) >= validation_time_limit:
                        break
                    if validation_batch_limit and i >= validation_batch_limit:
                        break
            
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_batches_accum, op=dist.ReduceOp.SUM)
            
            avg_val_loss = (val_loss_accum / val_batches_accum).item()

            if is_rank0:
                writer.add_scalar("validation", avg_val_loss, n_iter)
                post_fix_dict["Validation"] = f"{avg_val_loss:.6f}"
                print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.6f}")

    if is_rank0:
        writer.close()
    
    # %% Final Evaluation on Validation Set
    run_final_validation(
        model=model,
        datasplit_path=datasplit_path,
        input_array_info=input_array_info,
        target_array_info=target_array_info,
        classes=classes,
        local_rank=local_rank,
        batch_size=batch_size
    )
    
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to config file")
    args = parser.parse_args()
    
    train(args.config_path)