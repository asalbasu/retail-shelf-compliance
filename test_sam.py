import torch
from groundingdino.util import inference  # type: ignore[import]


print("Torch version:", torch.__version__)
print("Loading SAM only...")

# 👇 this should be similar to whatever you're doing in app.py
# If you have a helper function to build SAM, import & call it here instead.
from segment_anything import sam_model_registry

checkpoint_path = "sam_vit_h_4b8939.pth"  # change to your actual .pth path

sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
sam.to("cpu")
sam.eval()

print("SAM loaded OK on CPU.")
