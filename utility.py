import os
from PIL import Image
import numpy as np
import cv2

def getFiles(path, getPath = False):
    img_path = []

    for (root, directories, files) in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            img_path.append(file_path)

    imgs = []
    for i in range(len(img_path)):
        img = Image.open(img_path[i])
        img = np.array(img, dtype=np.float32)
        imgs.append(img)

    # print("images: ",len(imgs))

    if getPath:
        return imgs, img_path
    else:
        return imgs

def checkSize(sr,hr):
    if len(hr.shape) != 3:
        hr = np.expand_dims(hr,axis=2)
        hr = np.concatenate([hr]*3,2)

    new_hr = hr[0:sr.shape[0],0:sr.shape[1],:]
    # print(new_hr.shape)
    return new_hr

def calculatePSNR(sr, hr, scale = 2):
    if np.size(hr) == 1: return 0
    hr = checkSize(sr,hr)
    diff = (sr - hr) / 256
    shave = scale
    diff[:, :, 0] = diff[:, :, 0] * 65.738 / 256
    diff[:, :, 1] = diff[:, :, 1] * 129.057 / 256
    diff[:, :, 2] = diff[:, :, 2] * 25.064 / 256

    diff = np.sum(diff, axis=2)

    valid = diff[shave:-shave, shave:-shave]
    mse = np.mean(valid ** 2)

    return -10 * np.log10(mse)

def getbicubic(lr,scale,path):
    b_list = []
    for i in range(len(lr)):
        bicubic = cv2.cvtColor(lr[i], cv2.COLOR_RGB2BGR)
        bicubic = cv2.resize(bicubic,dsize=(bicubic.shape[1]*scale,bicubic.shape[0]*scale),interpolation=cv2.INTER_CUBIC)
        newPath = path[i].replace("lr","bicubic")
        newPath = newPath.replace(".png","_bicubic.png")
        cv2.imwrite(newPath,bicubic)
        bicubic = cv2.cvtColor(bicubic, cv2.COLOR_BGR2RGB)
        bicubic = np.clip(bicubic,0.0,255.0)
        b_list.append(bicubic)

    return b_list
def getPSNR(o_path, sr_path, b_path,lr_path, scale):
    ori = getFiles(o_path)
    sr = getFiles(sr_path)
    bi = getFiles(b_path)

    # bicubic 파일이 없을때
    # lr,img_path = getFiles(lr_path,getPath=True)
    # bi = getbicubic(lr,scale,img_path)

    o_sr_PSNR = 0.0
    o_bi_PSNR = 0.0
    for i in range(len(ori)):
        o_sr_PSNR += calculatePSNR(sr[i], ori[i], scale=scale)
        o_bi_PSNR += calculatePSNR(bi[i], ori[i], scale=scale)

    return o_sr_PSNR/len(ori), o_bi_PSNR/len(ori)
