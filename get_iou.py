from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from metrics_my import SigmoidMetric,SamplewiseSigmoidMetric,PD_FA
from utils.utils import preprocess_input

def resize_image_NEAR(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.NEAREST)
        new_image = Image.new('L', size, (0))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.NEAREST)
    return new_image

def resize_image_BiLi(image, size, letterbox_image):
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

def resize_image_BiCU(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def calculate_ious(net,test_txt_path,img_root,png_root,input_shape,n_thre):

    iou_metric = SigmoidMetric(score_thresh=n_thre)
    nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=n_thre)
    eval_PD_FA = PD_FA()
    iou_metric.reset()
    nIoU_metric.reset()

    test_ids = [i.split(' ')[0].split('\\')[-1].split('.')[0] for i in open(test_txt_path, 'r', encoding='utf-8').readlines()]

    tbar = tqdm(test_ids)

    for _,i in enumerate(tbar):
        img_path = img_root + i + '.jpg'
        png_path = png_root + i + '.png'

        image_data = Image.open(img_path).convert('RGB')
        image_data = resize_image_BiLi(image_data, input_shape, True)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        gt_img = Image.open(png_path)
        gt_img = resize_image_NEAR(gt_img, input_shape, True)
        gt_img = np.expand_dims(preprocess_input(np.array(gt_img, dtype='float32')), [0,1])
        gt_img = torch.from_numpy(gt_img)

        with torch.no_grad():
            outputs = net(torch.from_numpy(image_data).cuda())[-1]

        iou_metric.update(outputs.cpu(), gt_img)
        nIoU_metric.update(outputs.cpu(), gt_img)
        eval_PD_FA.update(outputs.cpu()[0, 0, :, :] > n_thre, gt_img[0, 0, :, :], input_shape)

        # _, IoU = iou_metric.get()
        # _, nIoU = nIoU_metric.get()
        # tbar.set_description('IoU:%f, nIoU:%f' % (IoU, nIoU))
    _, IoU = iou_metric.get()
    _, nIoU = nIoU_metric.get()
    PD,FA = eval_PD_FA.get()
    return IoU, nIoU, PD, FA