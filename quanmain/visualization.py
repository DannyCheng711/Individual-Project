import matplotlib.pyplot as plt
import numpy as np
  
""" ==== Draw the Figure ==== """

resolutions = ["128x128", "160x160", "192x192", "224x224", "256x256"]
mAP = [0.2055, 0.2575, 0.2780, 0.3052, 0.3096]
flops = [205.41, 320.95, 462.17, 629.07, 821.64]  # in M
ram = [39.7, 40.8, 42.1, 43.7, 45.5]  # MB # alloc

x = np.array(flops)
y = np.array(mAP)

plt.figure(figsize=(7, 5))

# Scatter + line
plt.plot(x, y, "-o", color="red", label="Input Resolution")

# Annotate each point
for px, py, res in zip(x, y, resolutions):
    plt.text(px * 1.02, py, f"{res}", fontsize=8, va="center")

plt.xlabel("FLOPs (Millions)")
plt.ylabel("mAP@0.5")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("./quanmain/profile/flops_vs_map.png", dpi=300)

""" ==== Draw the Different Backbone Comparison Figure ==== """

backbones = ["MCUNet", "MobileNetV2", "ResNet-18"]
params_m = [2.362, 0.963, 12.553]          # Params (M)
flops_m  = [320.952, 99.009, 1929]           # FLOPs (Millions) @160x160 
peak_ram = [40.8, 21.8, 227.2]           # Peak RAM (MB) (host estimate or on-device)
map50    = [0.2575, 0.1890, 0.1832]    # mAP@0.5

metrics = [
    ("Params (M)", params_m, "{:.2f}"),
    ("FLOPs (M)", flops_m,  "{:.0f}"),
    ("Peak RAM (MB)", peak_ram, "{:.0f}"),
    ("mAP@0.5", map50, "{:.3f}"),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=False)

for ax, (title, values, fmt) in zip(axes.flatten(), metrics):
    color = "#FDBA74" if title == "mAP@0.5" else "#BBBBBB"
    bars = ax.bar(backbones, values, width=0.3, color=color, edgecolor='black')
    ax.set_title(title, fontsize=12)
    ax.set_xticklabels(backbones, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)   # y-axis tick labels size

    ymax = max(values) if max(values) > 0 else 1.0
    ax.set_ylim(0, ymax * 1.1)  
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width()/2, 
            v + 0.0005*ymax, 
            fmt.format(v),
            ha='center', va='bottom', fontsize=10
        )

# plt.suptitle("Backbone Comparison across Metrics", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("./quanmain/profile/backbone_comparison.png", dpi=300)
plt.show()