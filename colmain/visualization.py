import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# # Data
# occ_pairs = ["30/30", "30/50", "50/50"]
# baseline_pair = np.array([0.6229, 0.43015, 0.2818])
# wbf_pair  = np.array([0.7313, 0.6038, 0.3579])
# feat_pair = np.array([0.5270, 0.56625, 0.1877])

# x = np.arange(len(occ_pairs))
# width = 0.25

# fig, ax = plt.subplots(figsize=(9, 5))

# b1 = ax.bar(x - width, baseline_pair, width, label='Baseline', color="#DDDDDD", edgecolor='black')
# b2 = ax.bar(x, wbf_pair, width, label='Decision-level (WBF)', color="#999999",edgecolor='black')
# b3 = ax.bar(x + width, feat_pair, width, label='Feature-level', color="#555555",edgecolor='black')

# ax.set_xticks(x)
# ax.set_xticklabels(occ_pairs)
# ax.set_xlabel("Occlusion Pair (View 1 / View 2)")
# ax.set_ylabel("mAP@0.5")
# ax.legend()

# # Annotate bars
# def annotate(bars):
#     for bar in bars:
#         h = bar.get_height()
#         ax.annotate(f'{h:.3f}',
#                     xy=(bar.get_x() + bar.get_width()/2, h),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom', fontsize=8)

# annotate(b1); annotate(b2); annotate(b3)

# plt.tight_layout()
# os.makedirs("./colmain", exist_ok=True)
# plt.savefig("./colmain/output/fusion_mAP_group.png", dpi=300, bbox_inches='tight')

# Example mAP@50 numbers 
# Data 
# map_baseline = [0.6229, 0.4974, 0.3929, 0.2833]
# map_2views = [0.6793, 0.5538, 0.5038, 0.3425]   # baseline (2 views)
# map_3views = [0.7471, 0.7016, 0.6410, 0.4659]   # result (3 views)

# labels = [
#     "30/30/30",
#     "30/30/50",
#     "30/50/50",
#     "50/50/50"
# ]

# # Setup
# width = 0.25
# x = np.arange(len(labels)) * (2 * width + 0.4)  # spacing

# fig, ax = plt.subplots(figsize=(9, 6))

# # Reference line at 0
# ax.axhline(0, color='gray', linewidth=1, linestyle='--', zorder=0)

# # Bars
# bar1 = ax.bar(x - width, map_baseline, width, label='Baseline',
#               color="#DDDDDD", edgecolor='black')
# bar2 = ax.bar(x, map_2views, width, label='2-View',
#               color="#999999", edgecolor='black')
# bar3 = ax.bar(x + width, map_3views, width, label='3-View',
#               color="#555555", edgecolor='black')

# # X-axis
# ax.set_xticks(x)
# ax.set_xticklabels(labels, fontsize=10, rotation=0, ha='center')
# ax.set_xlabel("Occlusion Triplet", fontsize=12)

# # Y-axis
# ax.set_ylabel("mAP@0.5", fontsize=12)

# # Annotate bars
# for bars in [bar1, bar2, bar3]:
#     for b in bars:
#         h = b.get_height()
#         ax.annotate(f'{h:.2f}',
#                     xy=(b.get_x() + b.get_width()/2, h),
#                     xytext=(0, 3 if h >= 0 else -3),
#                     textcoords="offset points",
#                     ha='center',
#                     va='bottom' if h >= 0 else 'top',
#                     fontsize=8)

# # Legend
# ax.legend(loc='upper left', fontsize=9)

# # Save / show
# plt.tight_layout()
# os.makedirs("./colmain", exist_ok=True)
# plt.savefig("./colmain/output/fusion_comp_threeviews.png", dpi=300, bbox_inches='tight')
# plt.show()


# File paths to your PR curve images
# img_paths = [
#     "./colmain/decision_fusion/result/pr_curve_comp_view1_occ30_occ30.png",
#     "./colmain/decision_fusion/result/pr_curve_comp_view1_occ30_occ50.png",
#     "./colmain/decision_fusion/result/pr_curve_comp_view1_occ50_occ50.png",
# ]

# titles = ["Occlusion 30/30", "Occlusion 30/50", "Occlusion 50/50"]

# # Create 2 rows × 2 columns grid, but adjust layout
# fig = plt.figure(figsize=(10, 8))

# # First two images in row 1
# for i in range(2):
#     ax = plt.subplot(2, 2, i + 1)
#     img = mpimg.imread(img_paths[i])
#     ax.imshow(img)
#     ax.axis("off")
#     ax.set_title(titles[i], fontsize=12)

# # Third image spanning both columns in row 2
# ax = plt.subplot(2, 2, (3, 4))  # span columns 3 and 4
# img = mpimg.imread(img_paths[2])
# ax.imshow(img)
# ax.axis("off")
# ax.set_title(titles[2], fontsize=12)

# plt.tight_layout()
# plt.savefig("./colmain/output/pr_curve_view1_panel.png", dpi=300, bbox_inches="tight")
# plt.show()

# map_baseline = [0.6229, 0.4974, 0.3929, 0.2833]
# map_2views = [0.6793, 0.5538, 0.5038, 0.3425]   
# map_3views = [0.7471, 0.7016, 0.6410, 0.4659]  
payload = [0, 1.312 * 2, 1.312 * 6]  # KB
map_values_gp1 = [0.6229, 0.6793, 0.7471]  # "30/30/30"
map_values_gp2 = [0.4974, 0.5538, 0.7016]  # "30/30/50"
map_values_gp3 = [0.3929, 0.5038, 0.6410]  # "30/50/50"
map_values_gp4 = [0.2833, 0.3425, 0.4659]  # "50/50/50"


plt.figure(figsize=(7, 5))
plt.plot(payload, map_values_gp1, marker='o', linestyle='-', color="#2563EB", label="30/30/30")
plt.plot(payload, map_values_gp2, marker='s', linestyle='-', color="#16A34A", label="30/30/50")
plt.plot(payload, map_values_gp3, marker='^', linestyle='-', color="#F97316", label="30/50/50")
plt.plot(payload, map_values_gp4, marker='d', linestyle='-', color="#DC2626", label="50/50/50")

# Add value labels for each line
for x, y in zip(payload, map_values_gp1):
    plt.text(x, y + 0.01, f"{y:.2f}", ha='center', fontsize=9, color="#2563EB")
for x, y in zip(payload, map_values_gp2):
    plt.text(x, y + 0.01, f"{y:.2f}", ha='center', fontsize=9, color="#16A34A")
for x, y in zip(payload, map_values_gp3):
    plt.text(x, y + 0.01, f"{y:.2f}", ha='center', fontsize=9, color="#F97316")
for x, y in zip(payload, map_values_gp4):
    plt.text(x, y + 0.01, f"{y:.2f}", ha='center', fontsize=9, color="#DC2626")

# Labels, legend, etc.
plt.xlabel("Payload Size (KB)", fontsize=10)
plt.ylabel("mAP@0.5", fontsize=10)
plt.ylim(0, 0.8)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Occlusion Pairs")
plt.tight_layout()

plt.savefig("./colmain/output/trade_off.png", dpi=300, bbox_inches="tight")
plt.show()