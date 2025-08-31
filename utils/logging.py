import os
import json
import csv
import time
import torch
from contextlib import nullcontext

class RunManager:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.log_path = os.path.join(run_dir, "train_log.csv")
        self._log_init = not os.path.exists(self.log_path)
        self.best_map = -1.0
        self.best_epoch = -1

    # logging
    def log_epoch(self, **row):
        # Ensure consistent column order
        cols = ["epoch","lr","train_loss","val_map50"]
        for c in cols:
            row.setdefault(c, None)
        with open(self.log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if self._log_init:
                w.writeheader(); self._log_init = False
            w.writerow({k: row.get(k) for k in cols})

    # profiling (once per run) 
    def write_profile(self, flops=None, params=None, peak_ram=None, extra=None):
        prof = {"FLOPs": flops, "Params": params, "PeakActsRAM": peak_ram}
        if extra: prof.update(extra)
        with open(os.path.join(self.run_dir, "profile.json"), "w") as f:
            json.dump(prof, f, indent=2)

    # checkpoints
    def save_ckpt(self, name, model, optimizer, epoch, best_val_map, config=None):
        path = os.path.join(self.run_dir, name)
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "best_val_map": best_val_map,
            "config": config or {},
        }
        torch.save(ckpt, path)

    def save_best(self, val_map, epoch, model, optimizer, config=None):
        if val_map > self.best_map:
            self.best_map, self.best_epoch = val_map, epoch
            self.save_ckpt("best.pth", model, optimizer, epoch, self.best_map, config)
            return True
        return False