from utils import *
import torch
import torch.nn as nn




# 예제 사용
img = load_image()
img_tensor = torch.tensor(img).permute(2, 0, 1)  # (C, H, W)
patches = patchify(img_tensor)


# 패치를 덩어리로 전송
send_patches_in_chunks(patches, delay=0.02)  # 0.5초 지연