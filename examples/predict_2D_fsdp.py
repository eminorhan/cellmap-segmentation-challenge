# This file is used to predict the segmentation logits of the 3D test datasets using the model trained in the train_2D.py script.
# %%
# Imports
from cellmap_segmentation_challenge.predict_fsdp import predict as predict_fsdp

config_path = "/lustre/gale/stf218/scratch/emin/cellmap-segmentation-challenge/examples/train_2D_fsdp.py"

# Overwrite the predictions if they already exist
predict_fsdp(config_path, crops="test", overwrite=True)

# %%
