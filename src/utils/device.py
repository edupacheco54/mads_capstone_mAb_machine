import torch


def get_best_device(prefer_mps=True):
    """
    Determine the best available computation device for PyTorch.

    The function prioritizes GPU acceleration when available. The selection
    order is:

    1. CUDA GPU (NVIDIA GPUs)
    2. MPS (Apple Metal Performance Shaders, typically on Apple Silicon)
    3. CPU fallback

    Parameters
    ----------
    prefer_mps : bool, optional
        If True, allow the function to return Apple's MPS backend when CUDA
        is not available. If False, the function will fall back directly to
        CPU instead of using MPS. This can be useful when MPS memory or
        compatibility issues are expected.

    Returns
    -------
    str
        The name of the device to use with PyTorch models. One of:
        - "cuda" : NVIDIA GPU
        - "mps"  : Apple Silicon GPU backend
        - "cpu"  : Standard CPU execution

    Notes
    -----
    This helper centralizes device selection for the project so that all
    models use a consistent strategy when choosing where to run inference.
    """

    # Prefer CUDA if an NVIDIA GPU is available
    if torch.cuda.is_available():
        return "cuda"

    # Optionally use Apple's Metal backend on Apple Silicon
    if prefer_mps and torch.backends.mps.is_available():
        return "mps"

    # Default fallback when no GPU acceleration is available
    return "cpu"
