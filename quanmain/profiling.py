import json
import os
import torch 
from thop import profile 
from dotenv import load_dotenv
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from utils.config_utils import fix_state_dict_keys 
from models.mcunetYolo.tinynas.nn.proxyless_net import ProxylessNASNets
from models.mcunet.mcunet.model_zoo import build_model
from models.dethead.yolodet import McunetTaps, MobilenetV2Taps, ResNet18Taps, McuYolo

load_dotenv()  # Loads .env from current directory

DEVICE = torch.device(
    "cuda" if os.getenv("DEVICE") == "cuda" and torch.cuda.is_available() else "cpu")
DATASET_ROOT = os.getenv("DATASET_ROOT")
VOC_ROOT = os.getenv("VOC_ROOT")


def _to_device_eval(model, device = "cuda"):
    model = model.to(device)
    model.eval()
    return model 

def count_params(model):
    # numel: return total number or elements in the input tensor
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total params": total, "trainable params": trainable}

# Multiply ACcumulate operations (MAC)
# Floating-Point Operations Per Second
def profile_macs_params(model, image_size, device):
    model = _to_device_eval(model, device)
    x = torch.zeros(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        macs, params = profile(model, inputs = (x, ), verbose=False)
    flops = 2 * macs   # convention: FLOPs = 2 × MACs
    return {"macs": int(macs), "flops": int(flops), "params": int(params)}

def peak_gpu_mem_bytes_forward(model, image_size, device="cuda"):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = _to_device_eval(model, device)
    x = torch.zeros(1, 3, image_size, image_size, device=device)
    torch.cuda.synchronize()
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    return {
        "peak_mem_alloc": int(torch.cuda.max_memory_allocated(device)),
        "peak_mem_reserved": int(torch.cuda.max_memory_reserved(device)),
    }
    
def human_bytes(n):
    if n is None: return "n/a"
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024: return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"

def human_big(n):
    if n is None: return "n/a"
    for u in ["","K","M","G","T"]:
        if abs(n) < 1000: return f"{n:.3f}{u}"
        n /= 1000
    return f"{n:.3f}P"


def write_profile_json(run_dir, image_size, flops, macs, params, peak_mem_alloc=None, peak_mem_reserved=None, extra=None):
    os.makedirs(run_dir, exist_ok=True)
    payload = {
        "image_size": image_size,
        "macs": macs,
        "params": params,
        "flops_human": human_big(flops),
        "macs_human": human_big(macs),
        "params_human": human_big(params),
        "peak_mem_alloc_bytes": peak_mem_alloc,
        "peak_mem_alloc_human": human_bytes(peak_mem_alloc) if peak_mem_alloc is not None else None,
        "peak_mem_reserved_bytes": peak_mem_reserved,
        "peak_mem_reserved_human": human_bytes(peak_mem_reserved) if peak_mem_reserved is not None else None,
    }
    if extra: payload.update(extra)
    path = os.path.join(run_dir, "profile.json")
    with open(path, "w") as f: json.dump(payload, f, indent=2)
    return path


def quick_profile(model, image_size=160, run_dir=None, device="cuda", extra=None):
    p = profile_macs_params(model, image_size, device)
    m = peak_gpu_mem_bytes_forward(model, image_size, device)
    flops, macs, params = p.get("flops"), p.get("macs"), p.get("params")
    peak_mem_alloc, peak_mem_reserved = m.get("peak_mem_alloc"), m.get("peak_mem_reserved")
    if run_dir:
        path = write_profile_json(run_dir, image_size, flops, macs, params, peak_mem_alloc, peak_mem_reserved, extra)
        print(f"[PROFILE] Wrote {path}")
    else:
        print(f"[PROFILE] size={image_size} | MACs={human_big(macs)} | Params={human_big(params)} | "
              f"PeakMem={human_bytes(peak_mem_alloc)} (alloc)")
        

def make_detector(net_id, num_classes=20, num_anchors=5):
    
    if net_id == "mbv2-w0.35":
        backbone, _, _ = build_model(net_id=net_id, pretrained=True)
        taps = MobilenetV2Taps(backbone, passthrough_idx=12, final_idx=16)
        model = McuYolo(taps=taps, num_classes=num_classes, num_anchors=num_anchors, final_ch=112, passthrough_ch=32, mid_ch=512).to(DEVICE)
    if net_id == "mcunet-in4":
        backbone, _, _ = build_model(net_id=net_id, pretrained=True)
        taps = McunetTaps(backbone, passthrough_idx=12, final_idx=16)
        model = McuYolo(taps=taps, num_classes=num_classes, num_anchors=num_anchors, final_ch=320, passthrough_ch=96, mid_ch=512).to(DEVICE)
    if net_id == "resnet-18":
        taps = ResNet18Taps(pretrained=True)
        model = McuYolo(taps=taps, num_classes=num_classes, num_anchors=num_anchors, final_ch=512, passthrough_ch=256, mid_ch=512).to(DEVICE)

    return model


if __name__ == "__main__":

    """ ==== Profiling Different Backbone ==== """
    for net_id in ["mbv2-w0.35", "resnet-18"]:
        # 1. Build model (architecture only, random init)
        model = make_detector(net_id, num_classes=20, num_anchors=5)

        # 2. Load weights from checkpoint
        if net_id == "mcunet-in4":
            run_dir = f"runs/mcunet_S5_res160_pkg_lrdecay"
        if net_id == "mbv2-w0.35":
            run_dir = f"runs/mbv2net_S5_res160_pkg_lrdecay"
        if net_id == "resnet-18":
            run_dir = f"runs/resnet_S5_res160_pkg_lrdecay"
        

        checkpoint = torch.load(os.path.join(run_dir, "best.pth"), map_location='cuda')

        # In case the checkpoint was saved as {"model": state_dict, "optimizer": ..., "epoch": ...}
        if "model" in checkpoint:
            state_dict = checkpoint["model"]

        # 3. Apply weights to model
        model.load_state_dict(state_dict)

        # 4. Switch to eval mode
        model.eval()
        quick_profile(model, image_size=160, run_dir=run_dir, 
            device=("cuda" if torch.cuda.is_available() else "cpu"), extra=None)
            
    
    """ ==== Profiling MCUNet ==== """
    # for grid_num in [6]:
    #     for image_size in [192]:

    #         run_dir = f"runs/mcunet_S{grid_num}_res{image_size}_pkg_lrdecay"
    #         extra = {"backbone": "mcunet-in4", "head": "yolov2", "S": grid_num}

    #         """===== Model Evaluation ====="""
    #         checkpoint = torch.load(os.path.join(run_dir, "best.pth"), map_location='cuda')
    #         print(" Converting PyTorch keys to PyTorch NAS keys...")
    #         fixed_state_dict = fix_state_dict_keys(checkpoint['model'])

    #         with open("./quanmain/model_config/mcunetYolo_config.json", "r") as f:
    #             mcunetYolo_config = json.load(f)

    #         pytorch_model = ProxylessNASNets.build_from_config(mcunetYolo_config)
    #         pytorch_model.load_state_dict(fixed_state_dict)
    #         pytorch_model.eval()

    #         quick_profile(pytorch_model, image_size=image_size, run_dir=run_dir, 
    #                     device=("cuda" if torch.cuda.is_available() else "cpu"), extra=extra)
            
  