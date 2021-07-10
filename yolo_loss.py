import torch
import torch.nn as nn
from yolo_util import bbox_iou
import numpy as np
import math


class Loss_func(nn.Module):
    def __init__(self, anchors, num_classes, img_size, device=torch.device('cpu')):
        super(Loss_func, self).__init__()
        self.device = device
        self.anchors = anchors  # list
        self.num_anchors = len(anchors)  # int
        self.num_classes = num_classes  # int
        self.bbox_attrs = 5 + num_classes  # int
        self.img_size = img_size  # (int, int)

        self.ignore_threshold = 0.5
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, input, targets):
        '''
        calculate losses
        '''
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h)
                          for a_w, a_h in self.anchors]

        # (bs, num_anchors, gird, grid, attrs)
        prediction = input.to(self.device).view(bs,  self.num_anchors,
                                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # get outputs
        x = torch.sigmoid(prediction[..., 0])          # Center x
        y = torch.sigmoid(prediction[..., 1])          # Center y
        w = prediction[..., 2]                         # Width
        h = prediction[..., 3]                         # Height
        conf = torch.sigmoid(prediction[..., 4])       # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # build targets
        mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.get_target(targets,
                                                                        scaled_anchors,
                                                                        in_w,
                                                                        in_h,
                                                                        self.ignore_threshold,
                                                                        device=self.device)

        # losses
        loss_x = self.bce_loss(x * mask, tx * mask)
        loss_y = self.bce_loss(y * mask, ty * mask)
        loss_w = self.mse_loss(w * mask, tw * mask)
        loss_h = self.mse_loss(h * mask, th * mask)
        loss_conf = self.bce_loss(conf * mask, mask)
        loss_nconf = 0.5 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0)
        loss_cls = self.bce_loss(pred_cls[mask == 1], tcls[mask == 1])

        # total loss = losses * weight
        loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
            loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
            10 * loss_conf * self.lambda_conf + 3 * loss_nconf * \
            self.lambda_conf + 20 * loss_cls * self.lambda_cls

        # print("loss_x:", loss_x)
        # print("loss_y:", loss_y)
        # print("loss_w:", loss_w)
        # print("loss_h:", loss_h)
        # print("loss_conf:", loss_conf)
        # print("loss_nconf:", loss_nconf)
        # print("loss_cls:", loss_cls)

        return loss, loss_x.item(), loss_y.item(), loss_w.item(),\
            loss_h.item(), loss_conf.item(), loss_cls.item()

    def get_target(self, target, anchors, in_w, in_h, ignore_threshold, device=torch.device('cpu')):
        '''
        convert label format to calculate loss
        '''
        bs = target.size(0)

        mask = torch.zeros(bs, self.num_anchors, in_h, in_w,
                           requires_grad=False).to(device)
        noobj_mask = torch.ones(bs, self.num_anchors,
                                in_h, in_w, requires_grad=False).to(device)
        tx = torch.zeros(bs, self.num_anchors, in_h, in_w,
                         requires_grad=False).to(device)
        ty = torch.zeros(bs, self.num_anchors, in_h, in_w,
                         requires_grad=False).to(device)
        tw = torch.zeros(bs, self.num_anchors, in_h, in_w,
                         requires_grad=False).to(device)
        th = torch.zeros(bs, self.num_anchors, in_h, in_w,
                         requires_grad=False).to(device)
        tconf = torch.zeros(bs, self.num_anchors, in_h,
                            in_w, requires_grad=False).to(device)
        tcls = torch.zeros(bs, self.num_anchors, in_h, in_w,
                           self.num_classes, requires_grad=False).to(device)

        for b in range(bs):
            for t in range(target.shape[1]):
                if target[b, t].sum() == 0:
                    continue
                # central point coordinate and width hight in grid scale
                gx = target[b, t, 1] * in_w
                gy = target[b, t, 2] * in_h
                gw = target[b, t, 3] * in_w
                gh = target[b, t, 4] * in_h
                # grid coordinate
                gi = gx.floor().long()
                gj = gy.floor().long()
                # Get shape of gt box [0, 0, gt_w, gt_h] grid scale
                gt_box = torch.FloatTensor(np.array(
                    [0, 0, gw.data.cpu().numpy(), gh.data.cpu().numpy()])).unsqueeze(0).to(device)
                # Get shape of anchor box
                '''
                [
                    [0, 0, a1_w, a1_h],
                    [0, 0, a2_w, a2_h],
                    [0, 0, a3_w, a3_h]
                ]
                '''
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                                  np.array(np.array(anchors))), 1)).to(device)
                # Calculate iou between gt and anchor shapes
                anch_ious = bbox_iou(gt_box, anchor_shapes)
                # Where the overlap is larger than threshold set mask to zero (ignore)
                noobj_mask[b,
                           (anch_ious > ignore_threshold).long(), gj, gi] = 0
                # Find the best matching anchor box
                best_n = torch.argmax(anch_ious, dim=0)

                # Masks
                mask[b, best_n, gj, gi] = 1
                # Coordinates
                tx[b, best_n, gj, gi] = gx - gi
                f = tx[0]
                ty[b, best_n, gj, gi] = gy - gj
                # Width and height
                tw[b, best_n, gj, gi] = math.log(gw/anchors[best_n][0] + 1e-16)
                th[b, best_n, gj, gi] = math.log(gh/anchors[best_n][1] + 1e-16)
                # object
                tconf[b, best_n, gj, gi] = 1
                # One-hot encoding of label
                tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1

        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls
