import torch

from nets.swin_transformer import *
from nets.yolo_salience_only_swin import YoloBody_Salience_Swin
from utils.utils import get_classes, get_anchors


def load_concat_pretrained(net,hjr_weight_path,microsoft_backbone_path):
    net_dict = net.state_dict()
    filted_dict = {}
    filted_dict.update({key: value for key, value in torch.load(hjr_weight_path, map_location=torch.device('cuda')).items() if
                   key in net_dict.keys() and net_dict[key].size() == value.size()})

    filted_dict.update(
        {'backbone.' + key: value for key, value in
         torch.load(microsoft_backbone_path, map_location=torch.device('cuda'))['model'].items() if
         'backbone.' + key in net_dict.keys() and net_dict['backbone.' + key].size() == value.size()})

    print('该模型共有{}层，其中有{}层是公共的。'.format(len(net_dict.keys()),len(filted_dict.keys())))
    net_dict.update(filted_dict)
    net.load_state_dict(net_dict)
    print('完成更新！')

if __name__ == '__main__':

    classes_path = 'model_data/small.txt'
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    class_names, num_classes = get_classes(classes_path)
    model = YoloBody_Salience_Swin(anchors_mask, num_classes)

    hjr_pth = r"F:\best.pth"
    micro_pth = r"D:\postgraduate\hjr\yolo\logs\swin_tiny_patch4_window7_224_22k.pth"
    load_concat_pretrained(model,hjr_pth,micro_pth)






