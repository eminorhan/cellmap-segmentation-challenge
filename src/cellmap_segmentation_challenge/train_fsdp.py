import os
import random
import time
import functools
import argparse
import glob
import re

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

from cellmap_data.utils import get_fig_dict, longest_common_substring
from cellmap_data.transforms.augment import NaNtoNum, Binarize, Normalize
from tensorboardX import SummaryWriter
from tqdm import tqdm
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
    # Note: model_save_path is a template string e.g., ".../{model_name}_{epoch}.pth"
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
    train_raw_value_transforms = getattr(config, "train_raw_value_transforms", T.Compose([T.ToDtype(torch.float, scale=True), Normalize(), NaNtoNum({"nan": 0, "posinf": None, "neginf": None})]))
    val_raw_value_transforms = getattr(config, "val_raw_value_transforms", T.Compose([T.ToDtype(torch.float, scale=True), Normalize(), NaNtoNum({"nan": 0, "posinf": None, "neginf": None})]))
    target_value_transforms = getattr(config, "target_value_transforms", T.Compose([T.ToDtype(torch.float), Binarize()]))
    max_grad_norm = getattr(config, "max_grad_norm", 1.0)
    force_all_classes = getattr(config, "force_all_classes", "validate")

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
    dataloader_kwargs = config.get("dataloader_kwargs", {})
    
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
        # All ranks load the checkpoint so they can access the full optimizer state
        checkpoint = torch.load(resume_path, map_location="cpu")
        
        # 1. Load Model State
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(checkpoint["model_state_dict"])
        
        # 2. Load Optimizer State
        # The checkpoint contains the FULL optimizer state. We must scatter it 
        # to the local shard for this specific rank.
        if "optimizer_state_dict" in checkpoint:
            full_osd = checkpoint["optimizer_state_dict"]
            sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)
            optimizer.load_state_dict(sharded_osd)
            del full_osd # Free memory
        elif is_rank0:
            print("Warning: Optimizer state not found in checkpoint. Optimizer initialized from scratch.")

        # 3. Load Scheduler
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # 4. Restore Metadata
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        else:
            # Fallback regex if epoch not in dict
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
        
        if is_rank0:
            epoch_bar = tqdm(range(iterations_per_epoch), desc=f"Epoch {epoch}", dynamic_ncols=True)
        else:
            epoch_bar = range(iterations_per_epoch)

        optimizer.zero_grad()

        for epoch_iter in epoch_bar:
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

            if not is_accumulating:
                if max_grad_norm is not None:
                    model.clip_grad_norm_(max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if is_rank0:
                post_fix_dict["Loss"] = f"{loss.item() * gradient_accumulation_steps:.6f}"
                post_fix_dict["lr"] = f"{scheduler.get_last_lr()[0]:.6f}"
                post_fix_dict["Step"] = f"{n_iter}"
                
                if hasattr(epoch_bar, 'set_postfix'):
                    epoch_bar.set_postfix(post_fix_dict)
                
                writer.add_scalar("loss", loss.item() * gradient_accumulation_steps, n_iter)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], n_iter)

        # %% Save Checkpoint
        # Use FULL_STATE_DICT to make resuming easier, but offload to CPU
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_model_state = model.state_dict()
            # Important: Gather full optimizer state for resuming on different topology
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
            
            if is_rank0:
                if validation_time_limit is not None:
                    val_bar = tqdm(total=validation_time_limit, desc="Validation", unit="s", dynamic_ncols=True)
                else:
                    val_bar = tqdm(total=validation_batch_limit or len(val_loader.loader), desc="Validation", dynamic_ncols=True)
            
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
                    
                    if is_rank0:
                        elapsed = time.time() - start_time
                        if validation_time_limit:
                            val_bar.n = min(elapsed, validation_time_limit)
                            val_bar.refresh()
                        else:
                            val_bar.update(1)

                    if validation_time_limit and (time.time() - start_time) >= validation_time_limit:
                        break
                    if validation_batch_limit and i >= validation_batch_limit:
                        break
            
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_batches_accum, op=dist.ReduceOp.SUM)
            
            avg_val_loss = (val_loss_accum / val_batches_accum).item()

            if is_rank0:
                val_bar.close()
                writer.add_scalar("validation", avg_val_loss, n_iter)
                post_fix_dict["Validation"] = f"{avg_val_loss:.6f}"
                print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.6f}")

    if is_rank0:
        writer.close()
    
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to config file")
    args = parser.parse_args()
    
    train(args.config_path)