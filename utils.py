import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
sys.path.append('..')


import requests
import random 
import time
import socket
import struct



imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
mask_token = torch.zeros(1, 3, 16, 16)
torch.nn.init.normal_(mask_token, std=.02)


def show_image(image, title=''):
    # image is [H, W, 3]
    print(image.shape)
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return



#img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg' # fox, from ILSVRC2012_val_00046145
# img_url = 'https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba.jpg' # cucumber, from ILSVRC2012_val_00047851
def load_image():
    img_url = 'https://s3.ap-northeast-2.amazonaws.com/img.kormedi.com/news/article/__icsFiles/artimage/2015/09/30/c_km601/face_540.jpg'
    #img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg'
    #img_url = 'https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba.jpg'
    img = Image.open(requests.get(img_url, stream=True).raw)
    img = img.resize((224, 224))
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std

    return img


# 이미지를 패치 단위로 분할하는 함수
def patchify(img, patch_size=16):
    C, H, W = img.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch size"
    patches = img.reshape(C, H // patch_size, patch_size, W // patch_size, patch_size)
    patches = patches.permute(1, 3, 0, 2, 4).reshape(-1, C, patch_size, patch_size)
    return patches

def unpatchify(patches, patch_size=16, image_size=224):
    num_patches_per_row = image_size // patch_size
    patches = patches.reshape(num_patches_per_row, num_patches_per_row, 3, patch_size, patch_size)
    patches = patches.permute(2, 0, 3, 1, 4).contiguous().view(3, image_size, image_size)
    return patches

# 패치를 랜덤하게 하나씩 전송하는 함수
def send_patches_in_chunks(patches, delay=0.5):
    patch_indices = list(range(patches.shape[0]))
    
    
    random.shuffle(patch_indices)  # 랜덤하게 패치 인덱스 섞기
    
    all_patches = [None] * 196
    for idx in patch_indices:
        
        patch = patches[idx, :, :, :].unsqueeze(0)  # 패치를 하나씩 선택
        
        send_to_server(patch,idx)  # B 디바이스로 패치 전송
        
        all_patches[idx] = patch
        time.sleep(delay)  # 전송 사이에 지연 추가

    
    
# B 디바이스로 데이터를 전송하는 함수 (구현 필요)
def send_to_server(patch, idx):
    # 네트워크를 통해 데이터를 전송하거나 공유 메모리를 사용하는 로직을 구현합니다.

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Server address and port
    server_address = ('localhost', 8020)

    try:
        # Convert patch and index to bytes
        patch_bytes = patch.numpy().tobytes()
        idx_bytes = idx.to_bytes(4, byteorder='big')

        # Combine index and patch bytes
        message = idx_bytes + patch_bytes

        # Send combined message to server
        sock.sendto(message, server_address)

    finally:
        # Close the socket
        sock.close()



