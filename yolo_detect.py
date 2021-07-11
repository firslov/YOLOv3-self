from __future__ import division
from yolo_util import predict_transform, prep_image, write_results, load_classes
import torch
import cv2
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import random


ANCHORS = torch.tensor([[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]])
classes, _ = load_classes('./cfg/classes.names')


def arg_parse():
    '''
    Command line
    '''
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--images", dest='images', help="Image / Directory containing images to perform detection upon",
                        default="eval/", type=str)
    parser.add_argument("--det", dest='det', help="Image / Directory to store detections to",
                        default="output", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence",
                        help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh",
                        help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default="weight/yo.pth", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=int)

    return parser.parse_args()


def write_box(x, img):
    '''
    draw bounding box in origin image(cv2)
    '''
    for i in range(x.size(0)):  # x (1, 8)
        c1 = tuple((int(x[i][1]), int(x[i][2])))
        c2 = tuple((int(x[i][3]), int(x[i][4])))
        cls = int(x[i][-1])
        color = random.choice(colors)
        label = "{0}".format(classes[cls])
        cv2.rectangle(img, c1, c2, color, 5)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 5, 2)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                    cv2.FONT_HERSHEY_PLAIN, 5, [225, 255, 255], 1)
    return img


def draw(origin_img, img, name, size, reso):
    # predict
    with torch.no_grad():
        # [1, 3, 416, 416]
        _prediction = model(img)

    # monitor confidence
    '''
    GRID = torch.tensor([13, 26, 52])
    outputs = _prediction
    conf = 0
    max_conf = 0
    for i in range(3):
            bs = outputs[i].size(0)
            gd = GRID[i]
            _temp = outputs[i]
            _temp = _temp.view(bs, 45, gd*gd).transpose(1, 2).contiguous().view(bs, 3*gd*gd, 15)
            max_conf = torch.sigmoid(_temp[..., 4].max()) if max_conf<torch.sigmoid(_temp[..., 4].max()) else max_conf
            conf += (torch.sigmoid(_temp[..., 4]) > 0.5).sum()
            
    print("positive:",conf)
    print("max_conf:", max_conf)
    '''

    write = 0

    for i in range(3):
        x = predict_transform(_prediction[i], reso, ANCHORS[i], 10)
        if write:
            prediction = torch.cat((prediction, x), 1)
        else:
            write = 1
            prediction = x

    prediction = write_results(prediction, 0.6, 10, nms_conf=0.4)
    # (index in batch, topleft x, topleft y, bottomright x, bottomright y, confidence, max score, max index) 416 scale

    # convert into image scale
    prediction[:, 1] *= size[0]/416
    prediction[:, 3] *= size[0]/416
    prediction[:, 2] *= size[1]/416
    prediction[:, 4] *= size[1]/416

    img = write_box(prediction, origin_img)

    # write file
    if not os.path.isdir("./output"):
        os.mkdir("./output")
    cv2.imwrite("./output/{}".format(name), img)


'''
main process
'''
# arg parse
args = arg_parse()
images = args.images
cfg = args.cfgfile
weights = args.weightsfile
inp_dim = args.reso

# load model
model = Darknet(cfg)
# pthfile = r'./weight/yo.pth'
model.load_state_dict(torch.load(weights, map_location='cpu'))
model.eval()

# get images path list
try:
    # images name list
    namelist = [img for img in os.listdir(images)]
    # images path list
    imlist = [osp.join(osp.realpath('.'), images, img)
              for img in namelist]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print("No file or directory with the name {}".format(images))
    exit()

# make output directory
if not os.path.exists(args.det):
    os.makedirs(args.det)

# images list(numpy)
loaded_ims = [cv2.imread(x) for x in imlist]
# images list(tensor)
im_batches = list(map(prep_image, loaded_ims, [
                  inp_dim for x in range(len(imlist))]))
# size list(W, H)
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
# im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

colors = pkl.load(open("pallete", "rb"))

for i in range(len(imlist)):
    draw(loaded_ims[i], im_batches[i], namelist[i], im_dim_list[i], inp_dim)
