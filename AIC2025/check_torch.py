import torch, torchvision
print(torch.__version__)       # 2.7.1+cu118
print(torchvision.__version__) # 0.22.1+cu118
print(torch.version.cuda)      # 11.8

from torchvision.ops import nms
print("NMS available ✅")   