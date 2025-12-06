import torch
from tqdm import tqdm
from utils.utils import *
from utils.utils_bbox import DecodeBox
from utils.utils_map import *
from get_iou import calculate_ious
from datetime import datetime

def fit_one_epoch_salience(model, yolo_loss, salience_loss, optimizer, epoch, epoch_step, gen, Epoch, cuda, save_dir,
                           best_map, best_iou, logsigma):
    object_loss_total = 0
    salience_loss_total = 0

    # 创建txt用来保存训练时meta_weights的变化情况
    current_time = datetime.now().strftime("%Y-%m-%d")
    txt_file_path_weights = os.path.join(save_dir, f"{current_time}_weights_change.txt")

    # 创建txt用来保存训练时loss的变化情况
    txt_file_path_losses = os.path.join(save_dir, f"{current_time}_loss_change.txt")

    # 创建txt用来保存训练时mAP和mIoU的变化情况
    txt_file_path_metrics = os.path.join(save_dir, f"{current_time}_metric_change.txt")

    model.train()
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            images, pnges, targets = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    pnges = torch.from_numpy(pnges).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    pnges = torch.from_numpy(pnges).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]

            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model(images)

            loss_value_all = 0  # loss_value_all单次迭代的目标检测损失
            # ----------------------#
            #   计算损失
            # ----------------------#

            for l in range(len(outputs[:-1])):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item

            # 显著性损失
            # salience_result = torch.clamp(outputs[-1],min=0,max=1)
            salience_loss_value = salience_loss(outputs[-1], pnges)  # salience_loss_value单次迭代的显著性检测损失

            salience_loss_total = salience_loss_total + salience_loss_value  # salience_loss_total已经过的所有迭代的总显著性检测损失
            object_loss_total = object_loss_total + loss_value_all  # object_loss_total已经过的所有迭代的总目标检测损失

            train_loss = [loss_value_all,salience_loss_value]

            with open(txt_file_path_weights, mode='a') as file:
                file.write('epoch:{}, iter:{}, object_det_weight:{}, salience_weight:{}\n'.format(epoch + 1, iteration,
                                                                                                  logsigma[0],
                                                                                                  logsigma[1]))

            with open(txt_file_path_losses, mode='a') as file:
                file.write('epoch:{}, iter:{}, object_det_loss:{}, salience_loss:{}\n'.format(epoch + 1, iteration,
                                                                                              loss_value_all,
                                                                                              salience_loss_value))

            train_loss_tmp = [1 / (2 * torch.exp(w)) * train_loss[i] + w / 2 for i, w in enumerate(logsigma)]

            # ----------------------#
            #   反向传播
            # ----------------------#
            sum(train_loss_tmp).backward()
            optimizer.step()

            pbar.set_postfix(**{'lr': get_lr(optimizer),
                                'object_loss': object_loss_total.item() / (iteration + 1),
                                'salience_loss': salience_loss_total.item() / (iteration + 1),
                                })
            pbar.update(1)

    print('Finish Train')

    model.eval()

    print('开始在测试集上测试！')

    MINOVERLAP = 0.5
    input_shape = [512, 512]
    map_out_path = 'temp_ep{}'.format(epoch + 1)
    path = "/root/autodl-tmp/yolo/dataset/IRSTD-1k/test.txt"
    img_root = '/root/autodl-tmp/yolo/dataset/IRSTD-1k/JPEGImages/'
    png_root = '/root/autodl-tmp/yolo/dataset/IRSTD-1k/masks/'
    image_ids = [i.split(' ')[0].split('\\')[-1].split('.')[0] for i in open(path, 'r', encoding='utf-8').readlines()]

    with torch.no_grad():

        if not os.path.exists(map_out_path):
            os.makedirs(map_out_path)
        if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
            os.makedirs(os.path.join(map_out_path, 'ground-truth'))
        if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
            os.makedirs(os.path.join(map_out_path, 'detection-results'))

        class_names, num_classes = get_classes('model_data/small.txt')
        anchors, _ = get_anchors('model_data/yolo_anchors.txt')
        for image_id in tqdm(image_ids):
            image_path = img_root + image_id + '.jpg'
            image = Image.open(image_path)
            get_map_txt(model, image_id, image, class_names, map_out_path, num_classes, anchors, input_shape)

        obj_name = 'target'
        txt_content = open(path, encoding='utf-8').readlines()
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w", encoding='utf-8') as new_f:
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

        current_map = get_map_for_train(MINOVERLAP, False, path=map_out_path)

        current_iou, current_niou, current_PD, current_FA = calculate_ious(model, path, img_root, png_root,
                                                                           input_shape=input_shape, n_thre=0.5)

        with open(txt_file_path_metrics, mode='a') as file:
            file.write('epoch:{}, mAP:{}, mIoU:{}\n'.format(epoch + 1, current_map, current_iou))

        # if current_map >= best_map:
        #     print('之前最好的mAP为：',best_map,'\n','当前轮次的mAP为：',current_map,'\n','满足条件，需要保存!')
        #     torch.save(model.state_dict(), os.path.join(save_dir, 'epoch_{}_AP50_{}_IoU_{}_nIoU_{}_PD_{}_FA_{}.pth'.format((epoch + 1),round(current_map,5),round(current_iou,5),round(current_niou,5),round(current_PD,5),round(current_FA,5))))
        #     return current_map
        # else:
        #     print('之前最好的mAP为：',best_map,'\n','当前轮次的mAP为：',current_map,'\n','不满足条件!')
        #     return best_map

        # if current_iou >= best_iou:
        #     print('之前最好的IoU为：',best_iou,'\n','当前轮次的IoU为：',current_iou,'\n','满足条件，需要保存!')
        #     torch.save(model.state_dict(), os.path.join(save_dir, 'epoch_{}_IoU_{}_nIoU_{}_PD_{}_FA_{}.pth'.format((epoch + 1),round(current_iou,5),round(current_niou,5),round(current_PD,5),round(current_FA,5))))
        #     return current_iou
        # else:
        #     print('之前最好的IoU为：',best_iou,'\n','当前轮次的IoU为：',current_iou,'\n','不满足条件!')
        #     return best_iou

        if current_iou >= best_iou and current_map >= best_map:
            print('之前最好的IoU为：', best_iou, '\n', '当前轮次的IoU为：', current_iou, '\n', '之前最好的mAP为：',
                  best_map, '\n', '当前轮次的mAP为：', current_map, '\n', '满足条件，需要保存!')
            torch.save(model.state_dict(), os.path.join(save_dir,
                                                        'epoch_{}_mAP_{}_IoU_{}_nIoU_{}_PD_{}_FA_{}.pth'.format(
                                                            (epoch + 1), round(current_map, 5), round(current_iou, 5),
                                                            round(current_niou, 5), round(current_PD, 5),
                                                            round(current_FA, 5))))
            return current_map, current_iou
        else:
            print('之前最好的IoU为：', best_iou, '\n', '当前轮次的IoU为：', current_iou, '\n', '之前最好的mAP为：',
                  best_map, '\n', '当前轮次的mAP为：', current_map, '\n', '不满足条件!')
            return best_map, best_iou


def get_map_txt(net, image_id, image, class_names, map_out_path, num_classes, anchors, input_shape):
    confidence = 0.1
    nms_iou = 0.5
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w", encoding='utf-8')
    image_shape = np.array(np.shape(image)[0:2])
    # ---------------------------------------------------------#
    #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    # ---------------------------------------------------------#
    image = cvtColor(image)
    # ---------------------------------------------------------#
    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    # ---------------------------------------------------------#
    image_data = resize_image(image, (input_shape[1], input_shape[0]), False)
    # ---------------------------------------------------------#
    #   添加上batch_size维度
    # ---------------------------------------------------------#
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

    images = torch.from_numpy(image_data)
    images = images.cuda()
    # ---------------------------------------------------------#
    #   将图像输入网络当中进行预测！
    # ---------------------------------------------------------#
    outputs = net(images)[:-1]
    bbox_decode = DecodeBox(anchors, num_classes, (input_shape[0], input_shape[1]), anchors_mask)
    outputs = bbox_decode.decode_box(outputs)
    # ---------------------------------------------------------#
    #   将预测框进行堆叠，然后进行非极大抑制
    # ---------------------------------------------------------#
    results = bbox_decode.non_max_suppression(torch.cat(outputs, 1), num_classes, input_shape,
                                              image_shape, False, conf_thres=confidence,
                                              nms_thres=nms_iou)

    if results[0] is None:
        return

    top_label = np.array(results[0][:, 6], dtype='int32')
    top_conf = results[0][:, 4] * results[0][:, 5]
    top_boxes = results[0][:, :4]

    for i, c in list(enumerate(top_label)):
        predicted_class = class_names[int(c)]
        box = top_boxes[i]
        score = str(top_conf[i])

        top, left, bottom, right = box
        if predicted_class not in class_names:
            continue

        f.write("%s %s %s %s %s %s\n" % (
            predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

    f.close()