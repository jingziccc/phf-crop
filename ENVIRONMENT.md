# Environment

The paper's runs used NVIDIA L40 GPUs with Python 3.10.19, PyTorch 2.6.0,
CUDA 12.4, torchvision 0.21.0, Transformers 4.50.0, NumPy 2.2.6, SciPy 1.15.3,
Pillow 12.1.0, and scikit-image 0.25.2.

Install a PyTorch build that supports your CUDA driver before installing the
remaining dependencies in [requirements.txt](requirements.txt). The
implementation checks use synthetic inputs and do not require model weights.

For offline inference, download the model files first or use an existing local
model directory. Then set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

The visualization notebooks additionally require Jupyter and Matplotlib.
