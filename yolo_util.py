from __future__ import division
from torch.autograd import Variable
import torch
import numpy as np
import cv2


def parse_cfg(cfgfile):
    '''
    Parse the net structure
    '''
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def unique(tensor):
    '''
    Return classes in detection
    '''
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # transform from center and width to exact coordinates
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
        torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=False):
    '''
    convert the model output to (batch_size, grid_size*grid_size*num_anchors, 5+classes)
    '''
    # origin output (batch_size, num_anchors*(5+classes), grid_size, grid_size)
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    # convert dim (batch_size, bbox_attrs*num_anchors, grid_size*grid_size) (1, 45, 169)
    prediction = prediction.view(
        batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    # convert dim (batch_size, grid_size*grid_size, bbox_attrs*num_anchors) (1, 169, 45)
    prediction = prediction.transpose(1, 2).contiguous()
    # convert dim (batch_size, grid_size*grid_size*num_anchors, bbox_attrs) (1, 507, 15)
    prediction = prediction.view(
        batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    # anchors in gird scale
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # x, y, conf sigmoid
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # list xy coordinate in grid scale
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    # list all girds
    '''
    for example:
    xy ==> [1,4] [2,5] [3,6]
    xy.repeat(1, 3).view(-1, 2).unsqueeze(0) ==>
        [[[1, 4],
          [1, 4],
          [1, 4],
          [2, 5],
          [2, 5],
          [2, 5],
          [3, 6],
          [3, 6],
          [3, 6]]]
    '''
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(
        1, num_anchors).view(-1, 2).unsqueeze(0)

    # origin xy in 0-1 add grid coordinate
    prediction[:, :, :2] += x_y_offset

    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    # origin wh(0-1) exp and times anchors in grid scale
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    # print(anchors)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # origin cls sigmoid
    prediction[:, :, 5:5 +
               num_classes] = torch.sigmoid((prediction[:, :, 5:5+num_classes]))

    # origin xywh times stride
    prediction[:, :, :4] *= stride

    # (batch_size, grid_size*grid_size*num_anchors, bbox_attrs) x、y、w、h 416 scale
    # [sig(tx)+cx, sig(ty)+cy, pwe^tw, phe^th, cf, cls...]
    return prediction


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    '''
    NMS and draw bounding box in origin image
    '''
    # clear the anchors that conf<thresh
    # print(prediction[:, :, 4].max())
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    # prediction[:,:,:4] (center x, center y, h, w, ...) ==> (topleft x, topleft y, bottomright x, bottomright y, ...)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2]/2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3]/2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2]/2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3]/2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)
    write = False

    # handle images in batch
    for ind in range(batch_size):
        image_pred = prediction[ind]
        # get the max probability and index in anchors
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        # image_pred (topleft x, topleft y, bottomright x, bottomright y, conf, max score, max index)
        image_pred = torch.cat(seq, 1)
        # clear the anchors that conf==0
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))

        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue

        # if no anchor, detect the next
        if image_pred_.shape[0] == 0:
            continue

        # get the class
        img_classes = unique(image_pred_[:, -1])

        # NMS in class
        for cls in img_classes:
            # one class
            cls_mask = image_pred_ * \
                (image_pred_[:, -1] == cls).float().unsqueeze(1)
            # clear the anchors that conf==0 in this class
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            # image_pred_class (topleft x, topleft y, bottomright x, bottomright y, conf, max score, max index)
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)
            # rank with conf
            conf_sort_index = torch.sort(
                image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            # idx is the number of anchors belong to the class
            idx = image_pred_class.size(0)

            # NMS
            for i in range(idx):
                try:
                    # the shape of return value is same with the second param
                    ious = bbox_iou(image_pred_class[i].unsqueeze(
                        0), image_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break

                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                # clear the anchor in class if iou>thresh
                image_pred_class[i+1:] *= iou_mask
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            batch_ind = image_pred_class.new(
                image_pred_class.size(0), 1).fill_(ind)
            # seq (topleft x, topleft y, bottomright x, bottomright y, conf, max score, max index)
            seq = batch_ind, image_pred_class

            # ensure output is not None
            if not write:
                # (1, 8)
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

        print(output[:, -1].data.cpu().int().numpy())

    try:
        # output(1, 8) ==> (index in batch, topleft x, topleft y, bottomright x, bottomright y, conf, max score, max index)
        return output
    except:
        return 0


def load_classes(namesfile):
    '''
    Return the list with all classes
    '''
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def letterbox_image(img, inp_dim):
    '''
    Zoom the image to 416*416, fill blank with (128, 128, 128)
    '''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(
        img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h, (w-new_w) //
           2:(w-new_w)//2 + new_w, :] = resized_image

    return canvas


def prep_image(img, inp_dim):
    '''
    Convert numpy to tensor
    '''
    # img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = cv2.resize(img, (inp_dim, inp_dim))
    # H*W*C(BGR) to C(RGB)*H*W
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    # normalization add dim 0 1*3*416*416
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def get_test_input():
    '''
    Test input
    '''
    img = cv2.imread("dogs.jpg")
    img = cv2.resize(img, (416, 416))  # Resize to the input dimension
    # BGR -> RGB | H X W C -> C X H X W
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    # Add a channel at 0 (for batch) | Normalise
    img_ = img_[np.newaxis, :, :, :]/255.0
    img_ = torch.from_numpy(img_).float()  # Convert to float
    img_ = Variable(img_)  # Convert to Variable
    return img_


def train_transform(prediction, inp_dim, anchors, num_classes):
    '''
    Convert the model output to: (batch_size, grid_size*grid_size*num_anchors, 5+classes)
    '''
    # origin output: (batch_size, num_anchors*(5+classes), grid_size, grid_size)
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    # convert dim (batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.view(
        batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    # convert dim (batch_size, grid_size*grid_size, bbox_attrs*num_anchors)
    prediction = prediction.transpose(1, 2).contiguous()
    # convert dim (batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    prediction = prediction.view(
        batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    # x, y, conf sigmoid
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    anchors = torch.FloatTensor(anchors)

    # cls sigmoid
    prediction[:, :, 5:5 +
               num_classes] = torch.sigmoid((prediction[:, :, 5:5+num_classes]))

    # (batch_size, grid_size*grid_size*num_anchors, bbox_attrs) sig(x)、sig(y)、w、h
    # x、y、w、h are the origin outputs of the model
    return prediction
