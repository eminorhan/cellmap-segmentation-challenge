import os
import shutil
import argparse
import time
from glob import glob
from typing import Any
import re
from pathlib import Path
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, DistributedSampler

# FSDP Imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)

from cellmap_data import CellMapDatasetWriter, CellMapImage
from cellmap_data.utils import (
    array_has_singleton_dim,
    is_array_2D,
    permute_singleton_dimension,
)
from cellmap_data.transforms.augment import NaNtoNum, Normalize
from tqdm import tqdm
from upath import UPath

# Use relative imports assuming this file is placed alongside the original predict.py
from .config import CROP_NAME, PREDICTIONS_PATH, RAW_NAME, SEARCH_PATH
from .utils import load_safe_config, get_test_crops
from .utils.datasplit import get_formatted_fields, get_raw_path


def setup():
    """Initialize the distributed process group."""
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    """Destroy the distributed process group."""
    dist.destroy_process_group()

def find_latest_checkpoint(config, is_rank0=False):
    """
    Finds the latest checkpoint file based on the config.
    Runs on Rank 0 and broadcasts to avoid metadata storms on Lustre.
    """
    checkpoint_path = None
    
    if is_rank0:
        try:
            model_name = getattr(config, "model_name", "model")
            save_path_template = getattr(config, "model_save_path", None)
            
            if save_path_template:
                pattern = save_path_template.format(model_name=model_name, epoch="*")
                files = glob(pattern)
                
                if files:
                    def extract_epoch(f):
                        match = re.findall(r'(\d+)', os.path.basename(f))
                        return int(match[-1]) if match else 0
                    
                    checkpoint_path = sorted(files, key=extract_epoch)[-1]
        except Exception as e:
            print(f"[Warn] Could not resolve checkpoint path automatically: {e}")

    obj_list = [checkpoint_path]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]

def consolidate_ranks(rank_path: str, final_path: str, is_rank0: bool):
    """
    Moves data chunks from the rank-specific temporary Zarr to the final Zarr. This bypasses CellMapDatasetWriter concurrency issues.
    """
    rank_path = Path(rank_path)
    final_path = Path(final_path)
    
    # Walk the temporary directory
    for root, dirs, files in os.walk(rank_path):
        for file in files:
            # Do NOT skip Zarr metadata files (.zarray, .zgroup, .zattrs)
            # We only want to skip system files like .DS_Store
            if file.startswith(".") and not file.startswith(".z"):
                continue
                
            src_file = Path(root) / file
            
            # Calculate relative path to mirror structure
            rel_path = src_file.relative_to(rank_path)
            dest_file = final_path / rel_path
            
            # Ensure destination directory exists
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the file
            try:
                # If destination exists (e.g. metadata created by Rank 0), don't overwrite blindly
                if dest_file.exists():
                    # # If it's a chunk (numeric), we shouldn't have collisions due to DistributedSampler
                    # # If it's metadata (.zarray), it should be identical, so we can ignore
                    # if not file.startswith(".z"):
                    #     print(f"[Warn] Chunk collision detected: {dest_file}")
                    # Delete source to keep cleanup clean
                    os.remove(src_file)
                else:
                    shutil.move(str(src_file), str(dest_file))
            except Exception as e:
                print(f"[Error] Failed to move {src_file} to {dest_file}: {e}")

    # Cleanup rank temp dir
    try:
        shutil.rmtree(rank_path)
    except Exception:
        pass

def predict_orthoplanes(model: torch.nn.Module, dataset_writer_kwargs: dict[str, Any], batch_size: int, input_array_info: dict):
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    is_rank0 = rank == 0

    if is_rank0:
        print("Predicting orthogonal planes (FSDP).")

    # -------------------------------------------------------------------------
    # 1. Setup Shared Temporary Directory
    # -------------------------------------------------------------------------
    base_target_path = dataset_writer_kwargs["target_path"]
    shared_tmp_path = str(UPath(base_target_path).parent / f"temp_ortho_{UPath(base_target_path).name}")
    
    if is_rank0:
        if not os.path.exists(shared_tmp_path):
            os.makedirs(shared_tmp_path, exist_ok=True)
    
    dist.barrier(device_ids=[local_rank])

    # -------------------------------------------------------------------------
    # 2. Predict each Axis
    # -------------------------------------------------------------------------
    for axis in range(3):
        if is_rank0:
            print(f"Processing Axis {axis}...")
            
        temp_kwargs = dataset_writer_kwargs.copy()
        # Point target to shared temp dir, but specific to this axis
        axis_path = os.path.join(shared_tmp_path, "output.zarr", str(axis))
        temp_kwargs["target_path"] = axis_path
        
        input_arrays = {k: v.copy() for k, v in temp_kwargs["input_arrays"].items()}
        target_arrays = {k: v.copy() for k, v in temp_kwargs["target_arrays"].items()}
        permute_singleton_dimension(input_arrays, axis)
        permute_singleton_dimension(target_arrays, axis)
        temp_kwargs["input_arrays"] = input_arrays
        temp_kwargs["target_arrays"] = target_arrays
        
        # _predict handles the rank-isolation logic internally
        _predict(
            model,
            temp_kwargs,
            batch_size=batch_size,
            input_array_info=input_array_info
        )
        
        dist.barrier(device_ids=[local_rank])

    # -------------------------------------------------------------------------
    # 3. Combine Predictions
    # -------------------------------------------------------------------------
    if is_rank0:
        print("Combining predictions from shared temp storage.")

    # ISOLATION STRATEGY:
    # We cannot write to the final path concurrently.
    # Each rank writes to a temp path, then we merge.
    final_target_path = dataset_writer_kwargs["target_path"]
    rank_target_path = f"{final_target_path}_rank{rank}"
    
    # 3a. Prepare Rank-Specific Writer
    rank_writer_kwargs = dataset_writer_kwargs.copy()
    rank_writer_kwargs["target_path"] = rank_target_path
    rank_writer_kwargs["overwrite"] = True # Safe, it's my own dir
    
    # 3b. Initialize Main Writer Structure (Rank 0 only)
    # We do this just to establish the .zarray/.zgroup metadata at the destination
    if is_rank0:
        # We init it to create structure, then discard object
        _ = CellMapDatasetWriter(**dataset_writer_kwargs)
        
    dataset_writer = CellMapDatasetWriter(**rank_writer_kwargs)

    # Load the images from the shared temp directory (Source)
    single_axis_images = {
        array_name: {
            label: [
                CellMapImage(
                    os.path.join(shared_tmp_path, "output.zarr", str(axis), label),
                    target_class=label,
                    target_scale=array_info["scale"],
                    target_voxel_shape=array_info["shape"],
                    pad=True,
                    pad_value=0,
                )
                for axis in range(3)
            ]
            for label in dataset_writer_kwargs["classes"]
        }
        for array_name, array_info in dataset_writer_kwargs["target_arrays"].items()
    }

    # Distributed Sampler for reading source blocks
    sampler = DistributedSampler(
        dataset_writer.blocks, 
        num_replicas=dist.get_world_size(), 
        rank=rank, 
        shuffle=False, 
        drop_last=False
    )
    
    tiled_loader = DataLoader(
        dataset_writer.blocks, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )
    
    iterator = tqdm(tiled_loader, dynamic_ncols=True) if is_rank0 else tiled_loader
    
    for batch in iterator:    
        outputs = {}
        for array_name, images in single_axis_images.items():
            outputs[array_name] = {}
            for label in dataset_writer_kwargs["classes"]:
                outputs[array_name][label] = []
                for idx in batch["idx"]:
                    average_prediction = []
                    for image in images[label]:
                        average_prediction.append(image[dataset_writer.get_center(idx)])
                    average_prediction = torch.stack(average_prediction).mean(dim=0)
                    outputs[array_name][label].append(average_prediction)
                outputs[array_name][label] = torch.stack(outputs[array_name][label])

        dataset_writer[batch["idx"]] = outputs

    # 3c. Consolidate Results
    # dist.barrier(device_ids=[local_rank])
    consolidate_ranks(rank_target_path, final_target_path, is_rank0)
    
    # -------------------------------------------------------------------------
    # 4. Cleanup
    # -------------------------------------------------------------------------
    # dist.barrier(device_ids=[local_rank])
    
    if is_rank0:
        print("Cleaning up shared temp directory...")
        try:
            shutil.rmtree(shared_tmp_path)
        except Exception as e:
            print(f"Warning: Could not remove temp dir {shared_tmp_path}: {e}")
        print("Combined predictions complete.")


def _predict(model: torch.nn.Module, dataset_writer_kwargs: dict[str, Any], batch_size: int, input_array_info):
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    is_rank0 = rank == 0

    value_transforms = T.Compose(
        [
            T.ToDtype(torch.float, scale=True),
            Normalize(),
            NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
        ],
    )

    # ISOLATION STRATEGY:
    # Each rank writes to a separate, temporary Zarr container.
    # Rank 0 initializes the FINAL container structure.
    # At end, all ranks move their chunks to final container.
    
    final_target_path = dataset_writer_kwargs["target_path"]
    rank_target_path = f"{final_target_path}_rank{rank}"
    
    # 1. Prepare Rank-Specific Writer
    rank_writer_kwargs = dataset_writer_kwargs.copy()
    rank_writer_kwargs["target_path"] = rank_target_path
    rank_writer_kwargs["overwrite"] = True # Safe, it's my own dir

    # 2. Initialize Main Writer Structure (Rank 0 only) to set up metadata
    if is_rank0:
        # Create structure at final destination
        # We assume dataset_writer_kwargs['overwrite'] respects what user passed to predict()
        _ = CellMapDatasetWriter(**dataset_writer_kwargs, raw_value_transforms=value_transforms)

    # 3. Initialize Local Writer
    dataset_writer = CellMapDatasetWriter(**rank_writer_kwargs, raw_value_transforms=value_transforms)

    # # Wait for Rank 0 to finish filesystem ops on final path (just for safety)
    # dist.barrier(device_ids=[local_rank])

    # -------------------------------------------------------------------------
    # Distributed Setup
    # -------------------------------------------------------------------------
    sampler = DistributedSampler(
        dataset_writer.blocks,
        num_replicas=world_size,
        rank=rank,
        shuffle=False, 
        drop_last=False
    )

    dataloader = DataLoader(
        dataset_writer.blocks, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=0, 
        pin_memory=True
    )

    model.eval()

    if is_rank0:
        print(f"dataset_writer: {dataset_writer}")
        print(f"Total blocks: {len(dataset_writer.blocks)}")
        print(f"Blocks per Rank: {len(dataloader) * batch_size} (approx)")

    singleton_dim = np.where([s == 1 for s in dataset_writer_kwargs["input_arrays"]["input"]["shape"]])[0]
    singleton_dim = singleton_dim[0] if singleton_dim.size > 0 else None

    iterator = tqdm(dataloader, dynamic_ncols=True) if is_rank0 else dataloader

    with torch.no_grad():
        for batch in iterator:
            inputs = batch["input"].to(local_rank, non_blocking=True)
            
            if singleton_dim is not None:
                inputs = inputs.squeeze(dim=singleton_dim + 2)
            
            outputs = model(inputs)

            if input_array_info["shape"][0] == 1:
                outputs = torch.nn.functional.interpolate(input=outputs, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
            else:
                outputs = torch.nn.functional.interpolate(input=outputs, size=inputs.shape[-3:], mode="trilinear", align_corners=False)
            
            if singleton_dim is not None:
                outputs = outputs.unsqueeze(dim=singleton_dim + 2)

            # Cast to float32 before writing. NumPy/Writers don't support BFloat16.
            outputs = outputs.float()

            outputs = {"output": outputs}
            dataset_writer[batch["idx"]] = outputs

    # 4. Consolidate
    # dist.barrier(device_ids=[local_rank])
    # Each rank moves its own files. This is parallel IO.
    consolidate_ranks(rank_target_path, final_target_path, is_rank0)


def predict(
    config_path: str,
    crops: str = "test",
    output_path: str = PREDICTIONS_PATH,
    do_orthoplanes: bool = False,
    overwrite: bool = False,
    search_path: str = SEARCH_PATH,
    raw_name: str = RAW_NAME,
    crop_name: str = CROP_NAME,
):
    # 1. Initialize Process Group
    local_rank = setup()
    rank = dist.get_rank()
    is_rank0 = rank == 0

    if is_rank0:
        print(f"Running FSDP Prediction on {dist.get_world_size()} GPUs")

    # 2. Load Config
    config = load_safe_config(config_path)
    classes = config.classes
    batch_size = 2 * config.batch_size  
    input_array_info = config.input_array_info
    target_array_info = config.target_array_info
    model = config.model

    if is_rank0:
        print(f"Batch size (per GPU): {batch_size}")
        print(f"Input array info: {input_array_info}")

    # 3. Move model to local device
    model = model.to(local_rank)

    # -------------------------------------------------------------------------
    # Manual Checkpoint Loading
    # -------------------------------------------------------------------------
    checkpoint_path = find_latest_checkpoint(config, is_rank0)
    
    if checkpoint_path:
        if is_rank0:
            print(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            if is_rank0:
                print(f"[Error] Failed to load state dict: {e}")
            raise e
    else:
        if is_rank0:
            print("[Warning] No checkpoint found! Using random initialization.")

    # -------------------------------------------------------------------------
    # FSDP WRAPPING
    # -------------------------------------------------------------------------
    if torch.cuda.is_bf16_supported():
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        if is_rank0:
            print("Mixed Precision: Enabled (BFloat16)")
    else:
        mp_policy = None
        if is_rank0:
            print("Mixed Precision: Disabled (FP32) - BF16 not supported")

    model = FSDP(
        model,
        device_id=local_rank,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.NO_SHARD 
    )
    
    if is_rank0:
        print(f"Model Wrapped in FSDP (NO_SHARD): {type(model)}")

    if do_orthoplanes and (array_has_singleton_dim(input_array_info) or is_array_2D(input_array_info, summary=any)):
        predict_func = predict_orthoplanes
    else:
        predict_func = _predict

    input_arrays = {"input": input_array_info}
    target_arrays = {"output": target_array_info}

    # 4. Prepare Dataset Writers
    if crops == "test":
        test_crops = get_test_crops()
        dataset_writers = []
        for crop in test_crops:
            raw_path = search_path.format(dataset=crop.dataset, name=raw_name)
            target_bounds = {"output": {axis: [crop.gt_source.translation[i], crop.gt_source.translation[i] + crop.gt_source.voxel_size[i] * crop.gt_source.shape[i]] for i, axis in enumerate("zyx")}}

            dataset_writers.append(
                {
                    "raw_path": raw_path,
                    "target_path": output_path.format(crop=f"crop{crop.id}", dataset=crop.dataset),
                    "classes": classes,
                    "input_arrays": input_arrays,
                    "target_arrays": target_arrays,
                    "target_bounds": target_bounds,
                    "overwrite": overwrite,
                    "device": f"cuda:{local_rank}",
                }
            )
    else:
        crop_list = crops.split(",")
        crop_paths = []
        for i, crop in enumerate(crop_list):
            if (isinstance(crop, str) and crop.isnumeric()) or isinstance(crop, int):
                crop = f"crop{crop}"
                crop_list[i] = crop

            crop_paths.extend(glob(search_path.format(dataset="*", name=crop_name.format(crop=crop, label="")).rstrip(os.path.sep)))

        dataset_writers = []
        for crop, crop_path in zip(crop_list, crop_paths):
            raw_path = get_raw_path(crop_path, label="")
            gt_images = {
                array_name: CellMapImage(
                    str(UPath(crop_path) / classes[0]),
                    target_class=classes[0],
                    target_scale=array_info["scale"],
                    target_voxel_shape=array_info["shape"],
                    pad=True,
                    pad_value=0,
                )
                for array_name, array_info in target_arrays.items()
            }

            target_bounds = {
                array_name: image.bounding_box
                for array_name, image in gt_images.items()
            }

            dataset = get_formatted_fields(raw_path, search_path, ["{dataset}"])["dataset"]

            dataset_writers.append(
                {
                    "raw_path": raw_path,
                    "target_path": output_path.format(crop=crop, dataset=dataset),
                    "classes": classes,
                    "input_arrays": input_arrays,
                    "target_arrays": target_arrays,
                    "target_bounds": target_bounds,
                    "overwrite": overwrite,
                    "device": f"cuda:{local_rank}",
                }
            )

    if is_rank0:
        print(f"Dataset writers len (predict): {len(dataset_writers)}")

    # 5. Iterate through crops
    for dataset_writer in dataset_writers:
        if is_rank0:
            print(f"Processing: {dataset_writer['target_path']}")
        
        predict_func(model, dataset_writer, batch_size, input_array_info)        

    cleanup()