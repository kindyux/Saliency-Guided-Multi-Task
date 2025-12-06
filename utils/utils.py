import numpy as np
from PIL import Image


#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BILINEAR)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BILINEAR)
    return new_image

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#   获得先验框
#---------------------------------------------------#
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#归一化mask
def preprocess_input(mask):
    mask = mask / 255.0
    return mask

#归一化mask
def Normed(image,dataset_name):
    assert dataset_name in ['SIRST', 'mixedall', 'ExtIRShip','IHAST','IRDST','Small-SSDD','IRSTD-1K'], "please check your dataset name"
    if dataset_name == 'SIRST':
        img_norm_cfg = dict(mean=111.86589813232422, std=27.563417434692383)
        return image * img_norm_cfg['std'] + img_norm_cfg['mean']
    elif dataset_name == 'mixedall':
        img_norm_cfg = {'mean': 99.99104309082031, 'std': 55.49182891845703}
        return image * img_norm_cfg['std'] + img_norm_cfg['mean']
    elif dataset_name == 'ExtIRShip':
        img_norm_cfg = {'mean': 65.75875854492188, 'std': 53.26555252075195}
        return image * img_norm_cfg['std'] + img_norm_cfg['mean']
    elif dataset_name == 'IHAST':
        img_norm_cfg = {'mean': 89.90446472167969, 'std': 46.753150939941406}
        return image * img_norm_cfg['std'] + img_norm_cfg['mean']
    elif dataset_name == 'IRDST':
        img_norm_cfg = {'mean': 101.5417251586914, 'std': 56.49897384643555}
        return image * img_norm_cfg['std'] + img_norm_cfg['mean']
    elif dataset_name == 'Small-SSDD':
        img_norm_cfg = {'mean': 29.209943771362305, 'std': 23.89558219909668}
        return image * img_norm_cfg['std'] + img_norm_cfg['mean']
    elif dataset_name == 'IRSTD-1K':
        img_norm_cfg = {'mean': 87.41641235351562, 'std': 39.69350051879883}
        return image * img_norm_cfg['std'] + img_norm_cfg['mean']