import torch


def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS backend for PyTorch.")
        return "mps"
    elif torch.cuda.is_available():
        print("Using CUDA backend for PyTorch.")
        return "cuda"
    else:
        print("Using CPU backend for PyTorch.")
        return "cpu"
