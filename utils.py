import os

# from data_loader import load_split
import numpy as np
import torch
import sys


# Saving and Loading Functions
def save_model_state(state_dict, filepath):
    """Save only the model weights (most common approach)"""
    torch.save(state_dict, filepath)
    print(f"Model weights saved to {filepath}")


def load_model_weights(model, filepath):
    """Load weights into an existing model"""
    model.load_state_dict(torch.load(filepath))
    print(f"Model weights loaded from {filepath}")
    return model


def validate_paths(storage_path, creator_name, model_name):
    """Validate the paths for storage, creator, and model"""
    if not os.path.exists(storage_path):
        raise ValueError(f"Storage path '{storage_path}' does not exist.")

    creator_path = os.path.join(storage_path, creator_name)
    if not os.path.exists(creator_path):
        os.makedirs(creator_path)

    model_dir = os.path.join(creator_path, model_name)
    if os.path.exists(model_dir):
        raise ValueError(f"Model directory '{model_dir}' already exists.")
    else:
        os.makedirs(model_dir, exist_ok=True)

    return model_dir


def print_available_memory():
    """Prints available system memory in GB (Linux only)."""
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        meminfo = {line.split(":")[0]: line.split(":")[1].strip() for line in lines}
        mem_free_kb = int(meminfo.get("MemAvailable", meminfo.get("MemFree", "0 kB")).split()[0])
        mem_free_gb = mem_free_kb / 1024 / 1024
        print(f"Available memory: {mem_free_gb:.2f} GB", flush=True)
    except Exception as e:
        print(f"Could not determine available memory: {e}", flush=True)
