from __future__ import print_function
import os
import gc
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from tqdm import tqdm



def test_one_epoch(args, net, test_loader):
    net.eval()

    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0

    for src, target, gt_flow in tqdm(test_loader):
        src = src.cuda()
        target = target.cuda()
        gt_flow = gt_flow.cuda()

        batch_size = src.size(0)
        num_examples += batch_size
        gt_flow_pred = net(src, target)
        ###########################
        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        loss = F.mse_loss(gt_flow_pred, gt_flow)
        total_loss += loss.item() * batch_size



    return total_loss * 1.0 / num_examples


def train_one_epoch(args, net, train_loader, opt):
    net.train()

    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0


    for src, target, gt_flow in tqdm(train_loader):
        src = src.cuda()
        target = target.cuda()
        gt_flow = gt_flow.cuda()

        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size
        gt_flow_pred = net(src, target)

        ###########################
        loss = F.mse_loss(gt_flow_pred, gt_flow) 

        loss.backward()
        opt.step()
        total_loss += loss.item() * batch_size


    return total_loss * 1.0 / num_examples


def test_flow(args, net, test_loader, boardio, textio):

    test_loss = test_one_epoch(args, net, test_loader)


    textio.cprint('==FINAL TEST==')
    textio.cprint('EPOCH:: %d, Loss: %f'% (-1, test_loss))


def train_flow(args, net, train_loader, test_loader, boardio, textio):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)


    best_test_loss = np.inf

    for epoch in range(args.epochs):
        scheduler.step()
        train_loss = train_one_epoch(args, net, train_loader, opt)
        
        test_loss = test_one_epoch(args, net, test_loader)


        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)

        textio.cprint('==TRAIN==')
        textio.cprint('EPOCH:: %d, Loss: %f'% (epoch, train_loss))


        textio.cprint('==TEST==')
        textio.cprint('EPOCH:: %d, Loss: %f'% (epoch, test_loss))
    

        textio.cprint('==BEST TEST==')
        textio.cprint('EPOCH:: %d, Loss: %f'% (epoch, best_test_loss))
        
        boardio.add_scalar('A->B/train/loss', train_loss, epoch)


        ############TEST
        boardio.add_scalar('A->B/test/loss', test_loss, epoch)

        if torch.cuda.device_count() > 1:
            torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        else:
            torch.save(net.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        gc.collect()