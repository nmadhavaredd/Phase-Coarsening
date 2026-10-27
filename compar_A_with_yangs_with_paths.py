"""
RMSE and SSIM vs. frame index j  --  Sri work vs. Yang et al.

Left axis : RMSE  (dark red)      Solid line  = sri work
Right axis: SSIM  (blue)          Dashed line = Yang et al.

This work : mean RMSE/SSIM over all test samples, read from the metrics CSV
Yang et al.: values digitized from Yang et al.'s published Fig. A
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ── Paths ─────────────────────────────────────────────────────────
base       = '/home1/nmadhavaredd2021/spectralETD/Dendrite_growth/dendrite_growth/dendrite_10in_50out_5w'
file_path  = os.path.join(base, 'vis/sv_dendrte_allvGT_PT_RMSE_SSIM_w2_all.csv')
yang_csv   = os.path.join(base, 'PSD/plotb_c/yang_digitized.csv')
output_dir = os.path.join(base, 'PSD/sv_RMSE_SSIM')

# ── Settings ──────────────────────────────────────────────────────
INPUT_LEN = 10        # frames 1-10 are inputs, outputs start at j = 11
J_START, J_END = 20, 60
SMOOTH = 5            # moving-average window (same as your plot A); set 1 for no smoothing

# ── This work: average over all samples at each timestep ─────────
data = pd.read_csv(file_path)
grouped = data.groupby('Timestep').agg({'RMSE': 'mean', 'SSIM': 'mean'}).reset_index()

j_mine    = grouped['Timestep'].values + INPUT_LEN + 1      # Timestep 0 -> j = 11
rmse_mine = grouped['RMSE'].values
ssim_mine = grouped['SSIM'].values
if SMOOTH > 1:
    rmse_mine = uniform_filter1d(rmse_mine, size=SMOOTH)
    ssim_mine = uniform_filter1d(ssim_mine, size=SMOOTH)

keep = (j_mine >= J_START) & (j_mine <= J_END)
j_mine, rmse_mine, ssim_mine = j_mine[keep], rmse_mine[keep], ssim_mine[keep]

# ── Yang et al.: digitized values ────────────────────────────────
yang = pd.read_csv(yang_csv)
yang = yang[(yang['j'] >= J_START) & (yang['j'] <= J_END)]
j_yang, rmse_yang, ssim_yang = yang['j'].values, yang['A_rmse'].values, yang['A_ssim'].values

# ── Plot ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
})
C_RMSE = "#8B0000"   # dark red
C_SSIM = "#1F3A93"   # blue

fig, ax1 = plt.subplots(figsize=(4.2, 3.5))
ax2 = ax1.twinx()

l1, = ax1.plot(j_mine, rmse_mine, color=C_RMSE, lw=1.8, ls="-",  label="RMSE (sri work)")
l2, = ax1.plot(j_yang, rmse_yang, color=C_RMSE, lw=1.5, ls="--", label="RMSE (Yang et al.)")
l3, = ax2.plot(j_mine, ssim_mine, color=C_SSIM, lw=1.8, ls="-",  label="SSIM (sri work)")
l4, = ax2.plot(j_yang, ssim_yang, color=C_SSIM, lw=1.5, ls="--", label="SSIM (Yang et al.)")

ax1.set_xlim(J_START, J_END)
ax1.set_xticks(np.arange(J_START, J_END + 1, 10))
ax1.set_xlabel(r"Frame index $j$")

ax1.set_ylim(0.0, 0.16)
ax1.set_yticks(np.arange(0.0, 0.161, 0.04))
ax1.set_ylabel("RMSE", color=C_RMSE)
ax1.tick_params(axis="y", colors=C_RMSE)

ax2.set_ylim(0.80, 1.00)
ax2.set_yticks(np.arange(0.80, 1.001, 0.05))
ax2.set_ylabel("SSIM", color=C_SSIM)
ax2.tick_params(axis="y", colors=C_SSIM)
ax2.spines["left"].set_color(C_RMSE)
ax2.spines["right"].set_color(C_SSIM)

ax1.legend(handles=[l1, l3, l2, l4], loc="lower center", bbox_to_anchor=(0.5, 1.01),
           ncol=2, frameon=False, fontsize=8, handlelength=2.5, columnspacing=1.2)
ax1.text(-0.22, 1.13, "A", transform=ax1.transAxes, fontsize=12, fontweight="bold")

fig.tight_layout()
os.makedirs(output_dir, exist_ok=True)
fig.savefig(os.path.join(output_dir, "rmse_ssim_vs_yang.png"), dpi=600)
fig.savefig(os.path.join(output_dir, "rmse_ssim_vs_yang.pdf"))
print("saved to", output_dir)