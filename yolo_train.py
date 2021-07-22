import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import yaml
import numpy as np
from tqdm import tqdm
from model.darknet import Darknet
from model.yolo_util import load_classes
from model.yolo_dataset import data_set
from model.yolo_loss import Loss_func

_, CLASSES = load_classes('./cfg/classes.names')


def train(cfg):
    """
    train function：GPU train, checkpoint, scheduler update
    """
    # model initialize
    global losses
    model = Darknet(cfg['net_structure'])
    model.to(cfg['device'])
    model.train()
    # weight initialize
    model.weight_init()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, 100, 0.5)
    start_epoch = -1
    learn_data = [[] for i in range(7)]

    ''' check model parameters requires_grad=True
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    '''

    # load dataset
    train_data = data_set(cfg['train_dir'], cfg['size'], train=True)
    train_loader = DataLoader(train_data, batch_size=cfg['batch_size'],
                              shuffle=True, num_workers=0, drop_last=True)

    # make loss function in 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(Loss_func(cfg['anchors'][i],
                                     CLASSES, (cfg['size'], cfg['size']), device=cfg['device']))

    # checkpoint
    if cfg['resume']:
        # path
        path_checkpoint = cfg['ckpt_dir'] + "/ckpt.pth"
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
    for epoch in range(start_epoch + 1, cfg['epoch']):
        # iterations
        for images, labels in tqdm(train_loader, leave=False):
            images = images.to(cfg['device']).detach()
            labels = labels.to(cfg['device'])
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
            loss = losses[0]  # losses[0] total loss
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
            if not os.path.isdir(cfg['ckpt_dir']):
                os.mkdir(cfg['ckpt_dir'])
            torch.save(checkpoint, cfg['ckpt_dir'] + "/ckpt.pth")

            # save learn data
            if not os.path.isdir(cfg['lrd_dir']):
                os.mkdir(cfg['lrd_dir'])
            learn_data = np.array(learn_data)
            np.savetxt(cfg['lrd_dir'] + '/{0}-{1}.csv'.format(epoch - 99, epoch),
                       learn_data, delimiter=',')

            # 100 epochs
            break

        print("Epoch {}".format(epoch))

    # save weight
    if not os.path.isdir(cfg['wt_dir']):
        os.mkdir(cfg['wt_dir'])
    torch.save(model.state_dict(), cfg['wt_dir'] + "/yo.pth")


if __name__ == "__main__":
    with open('cfg/cfg.yaml', 'r') as loadfile:
        config = yaml.load_all(loadfile, Loader=yaml.FullLoader)
        config_all = [x for x in config]

    # train mode
    config = config_all[0]

    train(config)
