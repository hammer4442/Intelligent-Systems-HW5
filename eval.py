import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import everything needed from your training file
from model1 import (
    TEMLatticeSeparator,
    load_png_tensor
)

# -----------------------
# CONFIG
# -----------------------
WEIGHTS_PATH = "run_test5_results/weights/tem_lattice_separator_10.pth"
SAMPLE_IMAGE = "sample.png"
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# -----------------------
# LOAD MODEL
# -----------------------
model = TEMLatticeSeparator().to(DEVICE)

state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)

model.eval()
print("Model loaded successfully")

# -----------------------
# LOAD SAMPLE IMAGE
# -----------------------
X = load_png_tensor(SAMPLE_IMAGE)      # (1, H, W)
X = X.unsqueeze(0).to(DEVICE)          # (B=1, 1, H, W)

print("Input shape:", X.shape)

# -----------------------
# RUN INFERENCE
# -----------------------
with torch.no_grad():
    y1, y2 = model(X)

print("Output shapes:")
print("y1:", y1.shape)
print("y2:", y2.shape)

# -----------------------
# MOVE TO CPU FOR DISPLAY
# -----------------------
x_np  = X.squeeze().cpu()
y1_np = y1.squeeze().cpu()
y2_np = y2.squeeze().cpu()

# -----------------------
# VISUALIZE
# -----------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(x_np, cmap="gray")
axes[0].set_title("Input (Moire)")
axes[0].axis("off")

axes[1].imshow(y1_np, cmap="gray")
axes[1].set_title("Predicted Layer 1")
axes[1].axis("off")

axes[2].imshow(y2_np, cmap="gray")
axes[2].set_title("Predicted Layer 2")
axes[2].axis("off")

plt.tight_layout()
plt.show()