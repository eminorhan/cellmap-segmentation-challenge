# This is an FSDP-compatible configuration file.
# Note: FSDP requires the optimizer to be initialized AFTER the model is wrapped.
# Therefore, we do not strictly instantiate the optimizer here, but define params for the training script to use.

import torch
from upath import UPath
# Assuming these are available in your path
from cellmap_segmentation_challenge.utils import get_tested_classes
from dinov3.eval.segmentation.models import build_segmentation_decoder

# %% Set hyperparameters
learning_rate = 3e-5
batch_size = 8  # batch size per GPU
gradient_accumulation_steps = 1
input_array_info = {"shape": (1, 1024, 1024), "scale": (32, 32, 32)}
target_array_info = {"shape": (1, 1024, 1024), "scale": (32, 32, 32)}
epochs = 9
iterations_per_epoch = 150
warmup_steps = 150
random_seed = 42

classes = get_tested_classes()

# ###### dinov3 model ######
TORCH_HUB_PATH = "/lustre/gale/stf218/scratch/emin/torch_hub"
DINOV3_REPO_PATH = "/lustre/gale/stf218/scratch/emin/dinov3"

torch.hub.set_dir(TORCH_HUB_PATH)

model_name = "dinov3_vitl16_linear"

# We construct the model here.
# NOTE: For FSDP, do not move the model to .cuda() here. The training script handles device placement.
backbone = torch.hub.load(
    DINOV3_REPO_PATH, 
    "dinov3_vitl16", 
    source="local", 
    weights=f"{TORCH_HUB_PATH}/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", 
    in_chans=1, 
    pretrained=True,
    use_fa3=True
)
model = build_segmentation_decoder(backbone, decoder_type="linear", num_classes=len(classes))
############################

logs_save_path = UPath("tensorboard/{model_name}").path
model_save_path = UPath("checkpoints/{model_name}_{epoch}.pth").path
datasplit_path = "datasplit.csv"

spatial_transforms = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "transpose": {"axes": ["x", "y"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}

filter_by_scale = True

if __name__ == "__main__":
    # When running via `torchrun`, we pass the config file path to the training script
    from cellmap_segmentation_challenge.train_fsdp import train as train_fsdp

    # Call the train function with the configuration file
    train_fsdp(__file__)