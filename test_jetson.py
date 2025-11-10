import sys, torch, torchvision
print("Python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA in torch:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())