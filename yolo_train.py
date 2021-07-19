import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
from darknet import Darknet
from yolo_util import load_classes
from yolo_dataset import data_set
from yolo_loss import Loss_func


if torch.cuda.is_available():
    device = torch.device('cuda:0')
    # cudnn.benchmark = True
else:
    device = torch.device('cpu')

# meta params
LR = 1e-2
BATCH_SIZE = 10
EPOCH = 500
ANCHORS = [[(116, 90), (156, 198), (373, 326)],
           [(30, 61), (62, 45), (59, 119)],
           [(10, 13), (16, 30), (33, 23)]]
GRID = [13, 26, 52]
SIZE = 416
_, CLASSES = load_classes('./cfg/classes.names')
THRESH = 0.5
RESUME = True


def train():
    '''
    train function：GPU train, checkpoint, scheduler update
    '''
    # model initialize
    model = Darknet("cfg/yolov3.cfg")
    model.to(device)
    # model.to(device)
    model.train()
    # weight initialize
    model.weight_init()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 100, 0.5)
    start_epoch = -1
    learn_data = [[] for i in range(7)]

    ''' check model parameters requires_grad=True
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    '''

    # load dataset
    train_data = data_set("./data/VOCdevkit/train", SIZE, train=True)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, drop_last=True)

    # make loss function in 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(Loss_func(ANCHORS[i],
                                     CLASSES, (SIZE, SIZE), device=device))

    # checkpoint
    if RESUME:
        # path
        path_checkpoint = "./ckpt/ckpt.pth"
        # load checkpoint
        checkpoint = torch.load(path_checkpoint)
        # load model parameters
        model.load_state_dict(checkpoint['net'])
        # load optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])
        # load scheduler
        scheduler.load_state_dict(checkpoint['scheduler'])
        # set start epoch
        start_epoch = checkpoint['epoch']

    # start train
    print("Start training...")
    # epochs
    for epoch in range(start_epoch+1, EPOCH):
        # iterations
        for images, labels in tqdm(train_loader, leave=False):
            images = images.to(device).detach()
            labels = labels.to(device)
            # forward
            optimizer.zero_grad()
            outputs = model(images)

            # monitor the confidence
            '''
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

            # losses list
            losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
            losses = []
            for _ in range(len(losses_name)):
                losses.append([])

            # calculate losses in 3 scales
            for i in range(3):
                _loss_item = yolo_losses[i](outputs[i], labels)
                for j, l in enumerate(_loss_item):
                    losses[j].append(l)

            # sum the values of the three scales
            losses = [sum(l) for l in losses]
            loss = losses[0]    # losses[0] total loss
            # print losses
            for i in range(len(losses)):
                print("X loss:", losses[1])
                print("Y loss:", losses[2])
                print("W loss:", losses[3])
                print("H loss:", losses[4])
                print("Conf loss:", losses[5])
                print("Cls loss:", losses[6])
                print("Total loss:", losses[0].item())

            # backward
            loss.backward()
            # update grad
            optimizer.step()

        # save learn data
        for i in range(7):
            if i == 0:
                learn_data[i].append(losses[i].item())
            else:
                learn_data[i].append(losses[i])

        # updata scheduler
        scheduler.step()

        # checkpoint
        if epoch != 0 and epoch % 100 == 0:
            print('epoch:', epoch)
            print('learning rate:', optimizer.state_dict()
                  ['param_groups'][0]['lr'])
            checkpoint = {
                "net": model.state_dict(),
                "epoch": epoch,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            # save checkpoint
            if not os.path.isdir("./ckpt"):
                os.mkdir("./ckpt")
            torch.save(checkpoint, './ckpt/ckpt.pth')

            # save learn data
            if not os.path.isdir("./lrd"):
                os.mkdir("./lrd")
            learn_data = np.array(learn_data)
            np.savetxt('./lrd/{0}-{1}.csv'.format(epoch-99, epoch),
                       learn_data, delimiter=',')

            # 100 epochs
            break

        print("Epoch {}".format(epoch))

    # save weight
    if not os.path.isdir("./weight"):
        os.mkdir("./weight")
    torch.save(model.state_dict(), "./weight/yo.pth")


if __name__ == "__main__":
    train()
