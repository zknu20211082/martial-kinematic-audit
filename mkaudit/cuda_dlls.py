"""Windows helper: expose torch's bundled cuDNN/cuBLAS DLLs to onnxruntime's CUDA execution provider.

onnxruntime-gpu on Windows looks for the CUDA 12 / cuDNN 9 DLLs on the DLL search path; the PyTorch wheel already ships
them in ``torch/lib``. Call :func:`expose_torch_cuda_dlls` before ``onnxruntime`` / ``rtmlib`` are imported. On other
platforms this is a no-op (onnxruntime finds the CUDA libraries through the usual library path)."""
import os


def expose_torch_cuda_dlls():
    if os.name != "nt":
        return
    import torch  # expose torch's bundled cuDNN/cuBLAS DLLs to onnxruntime's CUDA provider
    _TL = os.path.join(os.path.dirname(torch.__file__), "lib")
    os.environ["PATH"] = _TL + ";" + os.environ["PATH"]; os.add_dll_directory(_TL)
