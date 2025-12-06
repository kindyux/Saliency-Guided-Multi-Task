import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm
from torchstat import stat
from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from yolo import YOLO

if __name__ == "__main__":
    '''
    Recall和Precision不像AP是一个面积的概念，在门限值不同时，网络的Recall和Precision值是不同的。
    map计算结果中的Recall和Precision代表的是当预测时，门限置信度为0.5时，所对应的Recall和Precision值。

    此处获得的./map_out_IRDST_992_2_1/detection-results/里面的txt的框的数量会比直接predict多一些，这是因为这里的门限低，
    目的是为了计算不同门限条件下的Recall和Precision值，从而实现map的计算。
    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表整个map计算流程，包括获得预测结果、获得真实框、计算VOC_map。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅获得真实框。
    #   map_mode为3代表仅仅计算VOC_map。
    #   map_mode为4代表利用COCO工具箱计算当前数据集的0.50:0.95map。需要获得预测结果、获得真实框后并安装pycocotools才行
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 0
    #-------------------------------------------------------#
    #   此处的classes_path用于指定需要测量VOC_map的类别
    #   一般情况下与训练和预测所用的classes_path一致即可
    #-------------------------------------------------------#
    classes_path    = 'model_data/small.txt'
    #-------------------------------------------------------#
    #   MINOVERLAP用于指定想要获得的mAP0.x
    #   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
    #-------------------------------------------------------#
    MINOVERLAP      = 0.38
    #-------------------------------------------------------#
    #   map_vis用于指定是否开启VOC_map计算的可视化
    #-------------------------------------------------------#
    map_vis         = False
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    #-------------------------------------------------------#
    #   结果输出的文件夹，默认为map_out
    #-------------------------------------------------------#
    map_out_path    = 'map_out_SIRST_320_sailence_SWID_0.38shrehold'


    path = r"F:\红外小目标数据集\sirst-master\test.txt"
    img_root = r"F:/红外小目标数据集/sirst-master/JPEGImages/"
    #image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt"), encoding='utf-8').read().strip().split()
    image_ids = [i.split(' ')[0].split('\\')[-1].split('.')[0] for i in open(path, 'r',encoding='utf-8').readlines()]

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(confidence = 0.1, nms_iou = 0.5)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            #image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image_path = img_root+image_id+'.jpg'
            image       = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            yolo.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")
        
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        difficult_flag = False
        obj_name = 'target'
        txt_content = open(path,encoding='utf-8').readlines()
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w", encoding='utf-8') as new_f:
                for i in txt_content:
                    if image_id == i.split()[0].split('\\')[-1].split('.')[0]:
                        datas = i.split()
                        with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "a",
                                  encoding='utf-8') as new_f:
                            for k in datas[1:]:
                                data = k.split(',')
                                left = data[0]
                                top = data[1]
                                right = data[2]
                                bottom = data[3]
                                new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
                    else:
                        continue
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
