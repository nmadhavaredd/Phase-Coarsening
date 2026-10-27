import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.morphology import convex_hull_image
from scipy.spatial.distance import pdist

# Load data
input_path = '/home1/nmadhavaredd2021/spectralETD/Dendrite_growth/dendrite_growth/dendrite_10in_50out_5w/work_dirs/custom_exp/saved/inputs.npy'
preds_path = '/home1/nmadhavaredd2021/spectralETD/Dendrite_growth/dendrite_growth/dendrite_10in_50out_5w/work_dirs/custom_exp/saved/preds.npy'
trues_path = '/home1/nmadhavaredd2021/spectralETD/Dendrite_growth/dendrite_growth/dendrite_10in_50out_5w/work_dirs/custom_exp/saved/trues.npy'

preds = np.load(preds_path)  # (900, 50, 1, 64, 64)
trues = np.load(trues_path)  # (900, 50, 1, 64, 64)

print(f"Predictions shape: {preds.shape}")
print(f"Ground truth shape: {trues.shape}")

num_samples = trues.shape[0]   # 900
num_timesteps = trues.shape[1] # 50

# ── Helper functions ──────────────────────────────────────────────
def binarize(frame):
    """Convert frame to binary image."""
    img = frame.squeeze()  # (64, 64)
    binary = img > 0.5     # threshold at 0.5 for float 0-1 images
    return binary.astype(bool)

def get_shape_descriptors(binary):
    """Calculate convexity, solidity, and d_feret from binary image."""
    labeled = label(binary)
    props = regionprops(labeled)

    if len(props) == 0:
        return np.nan, np.nan, np.nan

    # Take the largest region
    region = max(props, key=lambda r: r.area)

    # Solidity = area / convex hull area
    solidity = region.solidity

    # Convexity = convex hull perimeter / region perimeter
    hull = convex_hull_image(binary)
    hull_labeled = label(hull)
    hull_props = regionprops(hull_labeled)
    if len(hull_props) == 0:
        return np.nan, solidity, np.nan
    hull_perimeter = hull_props[0].perimeter
    region_perimeter = region.perimeter
    convexity = hull_perimeter / region_perimeter if region_perimeter > 0 else np.nan

    # Feret diameter = max distance between boundary points
    coords = region.coords  # (N, 2)
    if len(coords) > 1:
        # Sample max 500 points for speed
        idx = np.random.choice(len(coords), min(500, len(coords)), replace=False)
        sampled = coords[idx]
        dists = pdist(sampled)
        d_feret = dists.max() if len(dists) > 0 else np.nan
    else:
        d_feret = np.nan

    return convexity, solidity, d_feret

# ── Calculate shape descriptors for all samples and timesteps ─────
print("Calculating shape descriptors... this may take a few minutes")

gt_convexity  = np.zeros((num_samples, num_timesteps))
gt_solidity   = np.zeros((num_samples, num_timesteps))
gt_dferet     = np.zeros((num_samples, num_timesteps))

pred_convexity = np.zeros((num_samples, num_timesteps))
pred_solidity  = np.zeros((num_samples, num_timesteps))
pred_dferet    = np.zeros((num_samples, num_timesteps))

for i in range(num_samples):
    if i % 100 == 0:
        print(f"Processing sample {i}/{num_samples}...")
    for t in range(num_timesteps):
        gt_bin   = binarize(trues[i, t])
        pred_bin = binarize(preds[i, t])

        gt_convexity[i, t],  gt_solidity[i, t],  gt_dferet[i, t]   = get_shape_descriptors(gt_bin)
        pred_convexity[i, t], pred_solidity[i, t], pred_dferet[i, t] = get_shape_descriptors(pred_bin)

# ── Average over all samples ──────────────────────────────────────
mean_gt_convexity  = np.nanmean(gt_convexity,  axis=0)  # (50,)
mean_gt_solidity   = np.nanmean(gt_solidity,   axis=0)
mean_gt_dferet     = np.nanmean(gt_dferet,     axis=0)

mean_pred_convexity = np.nanmean(pred_convexity, axis=0)
mean_pred_solidity  = np.nanmean(pred_solidity,  axis=0)
mean_pred_dferet    = np.nanmean(pred_dferet,    axis=0)

# x-axis: timesteps remapped to 11-60
timesteps = np.arange(11, 61)  # 11 to 60

# ── Plot B ────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(6, 5))
ax2 = ax1.twinx()

ax1.plot(timesteps, mean_gt_convexity,   'k-',  linewidth=2, label='convexity (G)')
ax1.plot(timesteps, mean_gt_solidity,    'r-',  linewidth=2, label='solidity (G)')
ax1.plot(timesteps, mean_pred_convexity, 'k--', linewidth=2, label='convexity (P)')
ax1.plot(timesteps, mean_pred_solidity,  'r--', linewidth=2, label='solidity (P)')

ax2.plot(timesteps, mean_gt_dferet,   'b-',  linewidth=2, label='d_feret (G)')
ax2.plot(timesteps, mean_pred_dferet, 'b--', linewidth=2, label='d_feret (P)')

ax1.set_xlabel('Frame Index (j)')
ax1.set_ylabel('convexity / solidity')
ax2.set_ylabel('d_feret')
ax1.set_xticks([10, 20, 30, 40, 50, 60])

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=8)

# plt.title('Time Evolution of Shape Descriptors')
plt.tight_layout()

output_dir = '/home1/nmadhavaredd2021/spectralETD/Dendrite_growth/dendrite_growth/dendrite_10in_50out_5w/PSD/plotb_c'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'Dendrite_plot_B.png'), dpi=300, bbox_inches='tight')
plt.show()

# ── Plot C ────────────────────────────────────────────────────────
# Relative error % = |GT - Pred| / GT * 100
rel_err_convexity = np.nanmean(np.abs(gt_convexity - pred_convexity) / np.abs(gt_convexity) * 100, axis=0)
rel_err_solidity  = np.nanmean(np.abs(gt_solidity  - pred_solidity)  / np.abs(gt_solidity)  * 100, axis=0)
rel_err_dferet    = np.nanmean(np.abs(gt_dferet    - pred_dferet)    / np.abs(gt_dferet)    * 100, axis=0)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(timesteps, rel_err_convexity, 'k-', linewidth=2, label='convexity')
ax.plot(timesteps, rel_err_solidity,  'r-', linewidth=2, label='solidity')
ax.plot(timesteps, rel_err_dferet,    'b-', linewidth=2, label='d_feret')

ax.set_xlabel('Frame Index (j)')
ax.set_ylabel('Relative Error (%)')
ax.set_xticks([10, 20, 30, 40, 50, 60])
ax.legend(loc='upper left')
# plt.title('Relative Error of Shape Descriptors')
plt.tight_layout()

plt.savefig(os.path.join(output_dir, 'Dendrite_plot_C.png'), dpi=300, bbox_inches='tight')
plt.show()

# Save the computed arrays for later use in plot D
np.save(os.path.join(output_dir, 'gt_convexity.npy'),   gt_convexity)
np.save(os.path.join(output_dir, 'gt_solidity.npy'),    gt_solidity)
np.save(os.path.join(output_dir, 'gt_dferet.npy'),      gt_dferet)
np.save(os.path.join(output_dir, 'pred_convexity.npy'), pred_convexity)
np.save(os.path.join(output_dir, 'pred_solidity.npy'),  pred_solidity)
np.save(os.path.join(output_dir, 'pred_dferet.npy'),    pred_dferet)

print("Done! Plots B and C saved.")
print("trues min/max:", trues[0, 0, 0].min(), trues[0, 0, 0].max())
print("preds min/max:", preds[0, 0, 0].min(), preds[0, 0, 0].max())
print("trues dtype:", trues[0, 0, 0].dtype)
print("preds dtype:", preds[0, 0, 0].dtype)

# Check how many non-zero pixels in first pred frame
binary_pred = preds[0, 0, 0].squeeze()
print("pred unique values:", np.unique(binary_pred)[:10])
print("trues unique values:", np.unique(trues[0, 0, 0].squeeze())[:10])