import torch
import sys
import platform
import multiprocessing

def check_environment():
    """Check if the environment is properly set up"""
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
    print("Platform:", platform.platform())
    print("CPU count:", multiprocessing.cpu_count())

if __name__ == "__main__":
    check_environment() 