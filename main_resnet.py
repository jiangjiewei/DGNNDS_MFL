import argparse
import os, sys
import shutil
import time, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader

from utils.transforms import get_train_test_set
from networks.resnet import resnet101
from utils.load_pretrain_model import load_pretrain_model
from utils.metrics import voc12_mAP, AveragePrecisionMeter
from models import Resnet_GNN
from networks.resnet_origi import resnet101_origi
from thop import profile


global best_prec1
best_prec1 = 0


def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch multi label Training')
    parser.add_argument('dataset', metavar='DATASET',
                        help='path to train dataset')
    parser.add_argument('train_data', metavar='DIR',
                        help='path to train dataset')
    parser.add_argument('test_data', metavar='DIR',
                        help='path to test dataset')
    parser.add_argument('trainlist', metavar='DIR',
                        help='path to train list')
    parser.add_argument('testlist', metavar='DIR',
                        help='path to test list')
    parser.add_argument('-pm', '--pretrain_model', default='', type=str, metavar='PATH',
                        help='path to latest pretrained_model (default: none)')
    parser.add_argument('-train_label', default='', type=str, metavar='PATH',
                        help='path to train label (default: none)')
    parser.add_argument('-graph_file', default='', type=str, metavar='PATH',
                        help='path to graph (default: none)')
    parser.add_argument('-word_file', default='', type=str, metavar='PATH',
                        help='path to word feature')
    parser.add_argument('-test_label', default='', type=str, metavar='PATH',
                        help='path to test label (default: none)')
    parser.add_argument('--print_freq', '-p', default=100, type=int, metavar='N',
                        help='number of print_freq (default: 100)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--model_name', default="SSGRL", type=str, metavar='M',
                        help='model name')
    parser.add_argument('--word_feature_dim', default=300, type=int, metavar='N',
                        help='number of total word_feature_dim to run')
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--step_epoch', default=30, type=int, metavar='N',
                        help='decend the lr in epoch number')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', type=int, default=0,
                        help='use pre-trained model')
    parser.add_argument('--crop_size', dest='crop_size', default=224, type=int,
                        help='crop size')
    parser.add_argument('--scale_size', dest='scale_size', default=448, type=int,
                        help='the size of the rescale image')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--post', dest='post', type=str, default='',
                        help='postname of save model')
    parser.add_argument('--num_classes', '-n', default=80, type=int, metavar='N',
                        help='number of classes (default: 80)')
    args = parser.parse_args()
    return args


def print_args(args):
    print("==========================================")
    print("==========       CONFIG      =============")
    print("==========================================")
    for arg, content in args.__dict__.items():
        print("{}:{}".format(arg, content))
    print("\n")


def main():
    global best_prec1
    args = arg_parse()
    print_args(args)

    # Create dataloader
    print("==> Creating dataloader...")
    train_data_dir = args.train_data
    test_data_dir = args.test_data
    train_list = args.trainlist
    test_list = args.testlist
    train_label = args.train_label
    test_label = args.test_label
    train_loader, test_loader = get_train_test_set(train_data_dir, test_data_dir, train_list, test_list, train_label,
                                                   test_label, args)

    # load the network
    print("==> Loading the network ... ", args.model_name)
    if args.model_name == "Resnet_GNN":
        model = Resnet_GNN(image_feature_dim=2048,
                      output_dim=2048, time_step=6,
                      word_feature_dim=args.word_feature_dim,
                      adjacency_matrix=args.graph_file,
                      word_features=args.word_file,
                      num_classes=args.num_classes)
    elif args.model_name == "Resnet101":
        model = resnet101_origi(num_classes=args.num_classes)
    else:
        exit("please input model name!!!!")
        model = 0

    if args.pretrained:
        if args.model_name == "Resnet101":
            inchannel = model.fc.in_features
            model.fc = nn.Linear(inchannel, 1000)
            model.load_state_dict(torch.load(args.pretrain_model), strict=False)
            inchannel = model.fc.in_features
            model.fc = nn.Linear(inchannel, args.num_classes)
        else:
            model = load_pretrain_model(model, args)

    # 多GPU训练选择
    if args.model_name != "Resnet101":
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model.cuda())

            for p in model.module.resnet_101.parameters():
                p.requires_grad = False
            for p in model.module.resnet_101.layer4.parameters():
                p.requires_grad = True
        else:
            for p in model.resnet_101.parameters():
                p.requires_grad = False
            for p in model.resnet_101.layer4.parameters():
                p.requires_grad = True
            model.cuda()
    else:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model.cuda())
        else:
            model.cuda()
    print("use: ", torch.cuda.device_count(), "GPU train......")

    criterion = nn.BCEWithLogitsLoss(reduce=True, size_average=True).cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_mAP']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            # checkpoint = torch.load(args.model_load_path, map_location='cpu')
            # model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        with torch.no_grad():
            validate(test_loader, model, criterion, 0, args)
        return


    for epoch in range(args.start_epoch, args.epochs):

        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        with torch.no_grad():
            mAP = validate(test_loader, model, criterion, epoch, args)
        # remember best prec@1 and save checkpoint
        is_best = mAP > best_prec1
        best_prec1 = max(mAP, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mAP': mAP,
        }, is_best, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()
    if args.model_name == "Resnet101":
        if torch.cuda.device_count() > 1:
            model.module.train()
        else:
            model.train()
    else:
        if torch.cuda.device_count() > 1:
            model.module.resnet_101.eval()
            model.module.resnet_101.layer4.train()
        else:
            model.resnet_101.eval()
            model.resnet_101.layer4.train()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = torch.tensor(target).cuda()
        input_var = torch.tensor(input).cuda()
        # compute output

        t1 = time.time()
        output = model(input_var)
        target = target.float()
        target = target.cuda()
        target_var = torch.autograd.Variable(target)
        loss = criterion(output, target_var)
        losses.update(loss.data.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses))


def on_end_print(ap_meter):
    map = 100 * ap_meter.value().mean()
    OP, OR, OF1, CP, CR, CF1 = ap_meter.overall()
    OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = ap_meter.overall_topk(3)
    print('Test: \t mAP {map:.3f}'.format(map=map))
    print('OP: {OP:.4f}\t'
          'OR: {OR:.4f}\t'
          'OF1: {OF1:.4f}\t'
          'CP: {CP:.4f}\t'
          'CR: {CR:.4f}\t'
          'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
    print('OP_3: {OP:.4f}\t'
          'OR_3: {OR:.4f}\t'
          'OF1_3: {OF1:.4f}\t'
          'CP_3: {CP:.4f}\t'
          'CR_3: {CR:.4f}\t'
          'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))

def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    ap_meter = AveragePrecisionMeter(difficult_examples=False)
    ap_meter.reset()
    # ap_list = []

    # switch to evaluate mode
    model.eval()
    end = time.time()
    x = []
    for i, (input, target) in enumerate(val_loader):
        target = torch.tensor(target).cuda()
        input_var = torch.tensor(input).cuda()
        output = model(input_var)
        target = target.float()
        target = target.cuda()
        target_var = torch.autograd.Variable(target)
        ap_meter.add(output.data, target)
        loss = criterion(output, target_var)
        losses.update(loss.data.item(), input.size(0))

        mask = (target > 0).float()
        v = torch.cat((output, mask), 1)
        x.append(v)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses))

    x = torch.cat(x, 0)
    x = x.cpu().detach().numpy()
    print(x.shape)
    np.savetxt(args.post + '_score', x)
    mAP = voc12_mAP(args.post + '_score', args.num_classes)
    print(' * mAP {mAP:.3f}'.format(mAP=mAP))
    on_end_print(ap_meter)
    return mAP


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    filename = 'checkpoint_{}.pth.tar'.format(args.post)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_{}.pth.tar'.format(args.post))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    main()
