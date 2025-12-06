import torch
from tqdm import tqdm
from utils.utils import *
from utils.utils_bbox import DecodeBox
from utils.utils_map import *
from get_iou import calculate_ious
from datetime import datetime
from scipy.optimize import minimize
from copy import deepcopy

def graddrop(grads):
    P = 0.5 * (1. + grads.sum(1) / (grads.abs().sum(1) + 1e-8))
    U = torch.rand_like(grads[:, 0])
    M = P.gt(U).view(-1, 1) * grads.gt(0) + P.lt(U).view(-1, 1) * grads.lt(0)
    g = (grads * M.float()).mean(1)
    return g

def pcgrad(grads, rng, num_tasks):
    grad_vec = grads.t()

    shuffled_task_indices = np.zeros((num_tasks, num_tasks - 1), dtype=int)
    for i in range(num_tasks):
        task_indices = np.arange(num_tasks)
        task_indices[i] = task_indices[-1]
        shuffled_task_indices[i] = task_indices[:-1]
        rng.shuffle(shuffled_task_indices[i])
    shuffled_task_indices = shuffled_task_indices.T

    normalized_grad_vec = grad_vec / (grad_vec.norm(dim=1, keepdim=True) + 1e-8)  # num_tasks x dim
    modified_grad_vec = deepcopy(grad_vec)
    for task_indices in shuffled_task_indices:
        normalized_shuffled_grad = normalized_grad_vec[task_indices]  # num_tasks x dim
        dot = (modified_grad_vec * normalized_shuffled_grad).sum(dim=1, keepdim=True)   # num_tasks x dim
        modified_grad_vec -= torch.clamp_max(dot, 0) * normalized_shuffled_grad
    g = modified_grad_vec.mean(dim=0)
    return g

def cagrad(grads, num_tasks, alpha=0.5, rescale=1):
    GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
    g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

    x_start = np.ones(num_tasks) / num_tasks
    bnds = tuple((0, 1) for x in x_start)
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (x.reshape(1, num_tasks).dot(A).dot(b.reshape(num_tasks, 1)) + c * np.sqrt(
            x.reshape(1, num_tasks).dot(A).dot(x.reshape(num_tasks, 1)) + 1e-8)).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grads.device)
    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        return g
    elif rescale == 1:
        return g / (1 + alpha ** 2)
    else:
        return g / (1 + alpha)



def grad2vec(gradients, grads, grad_dims, task):

    assert len(grad_dims) == len(gradients), "Lengths are not equal!"

    # store the gradients
    grads[:, task].fill_(0.0)

    cnt = 0

    for grad in gradients:
        if grad is not None:
            grad_cur = grad.data.detach().clone()
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg:en, task].copy_(grad_cur.data.view(-1))
        cnt += 1

    # cnt = 0
    # for mm in m.shared_modules():
    #     for p in mm.parameters():
    #         grad = p.grad
    #         if grad is not None:
    #             grad_cur = grad.data.detach().clone()
    #             beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
    #             en = sum(grad_dims[:cnt + 1])
    #             grads[beg:en, task].copy_(grad_cur.data.view(-1))
    #         cnt += 1


def overwrite_grad(m, newgrad, grad_dims, num_tasks):
    newgrad = newgrad * num_tasks  # to match the sum loss
    cnt = 0
    for mm in m.shared_modules():
        for param in mm.parameters():
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

def fit_one_epoch_salience(model, yolo_loss, salience_loss, optimizer, epoch, epoch_step, gen, Epoch, cuda, save_dir,
                           best_map, best_iou, meta, meta_optimizer, meta_loader, optimizer_current_lr,grads,grad_dims,rng,grad_method):
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
        for (iteration, (batch, meta_batch)) in enumerate(zip(gen, meta_loader)):
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

            # 更新权重值并记录到txt
            meta_optimizer.zero_grad()
            meta.unrolled_backward(batch, meta_batch, optimizer_current_lr, optimizer, yolo_loss, salience_loss)
            meta_optimizer.step()
            with open(txt_file_path_weights, mode='a') as file:
                file.write('epoch:{}, iter:{}, object_det_weight:{}, salience_weight:{}\n'.format(epoch + 1, iteration,
                                                                                                  meta.meta_weights[0],
                                                                                                  meta.meta_weights[1]))

            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model(images)

            object_loss_value = 0  # loss_value_all单次迭代的目标检测损失
            # ----------------------#
            #   计算损失
            # ----------------------#

            for l in range(len(outputs[:-1])):
                loss_item = yolo_loss(l, outputs[l], targets)
                object_loss_value += loss_item

            # 显著性损失

            salience_loss_value = salience_loss(outputs[-1], pnges)  # salience_loss_value单次迭代的显著性检测损失

            salience_loss_total = salience_loss_total + salience_loss_value  # salience_loss_total已经过的所有迭代的总显著性检测损失
            object_loss_total = object_loss_total + object_loss_value  # object_loss_total已经过的所有迭代的总目标检测损失

            train_loss_tmp = [meta.meta_weights[0] * object_loss_value, meta.meta_weights[1] * salience_loss_value]

            with open(txt_file_path_losses, mode='a') as file:
                file.write('epoch:{}, iter:{}, object_det_loss:{}, salience_loss:{}\n'.format(epoch + 1, iteration,
                                                                                              object_loss_value,
                                                                                              salience_loss_value))
            if grad_method == 'none':
                sum(train_loss_tmp).backward()
                optimizer.step()
            else:
                for i in range(len(meta.meta_weights)):

                    if i == len(meta.meta_weights) - 1:
                        gradient = torch.autograd.grad(train_loss_tmp[i],[param for module in model.shared_modules() for param in module.parameters()],retain_graph=False,create_graph=False)
                    else:
                        gradient = torch.autograd.grad(train_loss_tmp[i],[param for module in model.shared_modules() for param in module.parameters()],retain_graph=True,create_graph=False)

                    # 赋值grads中每个任务的共享梯度值
                    grad2vec(gradient, grads, grad_dims, i)

                if grad_method == 'graddrop':
                    g = graddrop(grads)
                elif grad_method == 'pcgrad':
                    g = pcgrad(grads, rng, len(meta.meta_weights))
                elif grad_method == 'cagrad':
                    g = cagrad(grads, len(meta.meta_weights), 0.4, rescale=1)
                else:
                    print('please choose a correct grad_method!')
                    break

                sum(train_loss_tmp).backward()

                # 将共享网络层的梯度清零
                model.zero_grad_shared_modules()

                overwrite_grad(model, g, grad_dims, len(meta.meta_weights))

                optimizer.step()

                # gradients_det = torch.autograd.grad(train_loss_tmp[0], model.shared_parameters(), retain_graph=True,
                #                                     create_graph=False)
                #
                # gradients_seg = torch.autograd.grad(train_loss_tmp[1], model.shared_parameters(), retain_graph=True,
                #                                     create_graph=False)
                #
                # # 遍历两个任务的共享参数的梯度
                # for grad_det, grad_seg in zip(gradients_det, gradients_seg):
                #     # 将任务1和任务2的梯度转换为向量
                #     grad_vec = torch.stack([grad_det, grad_seg], dim=0)  # shape: [2, param_dim]
                #
                #     # 计算梯度向量的归一化形式
                #     grad_norm = grad_vec.norm(dim=1, keepdim=True) + 1e-8  # 防止除零
                #     normalized_grad_vec = grad_vec / grad_norm  # shape: [2, param_dim]
                #
                #     # 计算两个梯度的点积
                #     dot = (normalized_grad_vec[0] * normalized_grad_vec[1]).sum()
                #
                #     # 如果点积为负，则进行调整
                #     if dot < 0:
                #
                #         # 投影 task1 的梯度到 task2 的法向量方向
                #         grad_det_adjusted = grad_vec[0] - dot * normalized_grad_vec[1]
                #
                #         # 投影 task2 的梯度到 task1 的法向量方向
                #         grad_seg_adjusted = grad_vec[1] - dot * normalized_grad_vec[0]
                #
                #     else:
                #         # 如果没有冲突，保留原始梯度
                #         grad_det_adjusted = grad_vec[0]
                #         grad_seg_adjusted = grad_vec[1]
                #
                #     shared_modules_final_gradients.append((grad_det_adjusted + grad_seg_adjusted) / 2)
                #
                # # # 清空共享网络部分的梯度列表
                # # shared_modules_final_gradients.clear()
                # #
                # # # 遍历每对梯度
                # # for index,(pri_g, aux_g) in enumerate(zip(gradients_det_new, gradients_seg_new)):
                # # #for pri_g, aux_g in zip(gradients_pri, gradients_aux):
                # #
                # #     if pri_g is None or aux_g is None:
                # #         print("梯度存在None值，无法进行balance操作")
                # #         break
                # #
                # #     if pri_g.is_sparse or aux_g.is_sparse:
                # #         raise RuntimeError('主任务梯度或辅助任务梯度是稀疏的，无法进行balance操作')
                # #
                # #     # 计算m和范数并加偏置
                # #     norm_aux = torch.norm(aux_g)
                # #     norm_pri = torch.norm(pri_g)
                # #
                # #     # 修正 aux_gradient
                # #     if norm_aux > norm_pri:
                # #         aux_g = aux_g * (norm_pri / norm_aux) * relax_factor + aux_g * (1 - relax_factor)
                # #
                # #     # 直接计算最终梯度并加入列表
                # #     final_g = pri_g + aux_g
                # #     shared_modules_final_gradients.append(final_g)
                #
                # sum(train_loss_tmp).backward()
                #
                # # # 将共享网络层的梯度清零
                # # model.zero_grad_shared_modules()
                #
                # # 遍历共享网络的参数及其对应的梯度
                # for param, shared_grads in zip(model.shared_parameters(), shared_modules_final_gradients):
                #
                #     # 如果参数当前没有梯度，直接赋值
                #     if param.grad is None:
                #         param.grad = shared_grads.detach()
                #     else:
                #         param.grad.copy_(shared_grads.detach())
                #
                # shared_modules_final_gradients.clear()

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