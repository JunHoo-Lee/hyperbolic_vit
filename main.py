from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime

from vit_pytorch import ViT
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch, num_epochs, batch_size, optim_type, is_hyp, c, l_reg, c_add = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type, cf.is_hyp, cf.c, cf.l_reg, cf.c_add

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

transform_test_stl = transforms.Compose([
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['stl10'], cf.std['stl10']),
])

if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    num_classes = 10
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root='/data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='/data', train=False, download=True, transform=transform_test)
    num_classes = 100

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Return network & file name
def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = LeNet(num_classes)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, num_classes)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, num_classes)
        file_name = 'resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes, is_hyp, c, c_add)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)+'_hyp:'+str(is_hyp)+str(c)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name

# Test only option
if (args.testOnly):
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    file_name = "vit"
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        acc = 100.*correct/total
        print("| Test Result\tAcc@1: %.2f%%" %(acc))

    sys.exit(0)

# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    file_name = "vit"
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('| Building net type [' + args.net_type + ']...')
    net = ViT(
            image_size = 32,
            patch_size = 4,
            num_classes = 100,
            dim = 256,
            depth = 6,
            heads = 16,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1
            )
    file_name = "vit"

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

# Training
def train(epoch):
    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0
    total_norm = 0
    total_normlog = 0
    total_var = 0
    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        out = net(inputs)               # Forward Propagation
        loss = criterion(out, targets)
        loss.backward()   # Backward Propagation
        optimizer.step()  # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f'
                %(epoch, num_epochs, batch_idx+1,
                    (len(trainset)//batch_size)+1, loss.item(), 100.*correct/total))
        sys.stdout.flush()

def test(epoch):
    global best_acc
    net.eval()
    net.training = False
    test_loss = 0
    test_loss_stl = 0
    correct = 0
    correct_stl = 0
    total = 0
    total_stl = 0
    norm_stl = 0
    normlog_stl = 0
    var_stl = 0
    norm_val = 0
    normlog_val = 0
    var_val = 0
    stl_idx = 0
    val_idx = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            val_idx += 1
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100.*correct/total
        var_val = var_val / val_idx
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f" %(epoch, loss.item(), acc))

    if acc > best_acc:
            print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
            state = {
                    'net':net.module if use_cuda else net,
                    'acc':acc,
                    'epoch':epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_point = './checkpoint/'+args.dataset+os.sep
            if not os.path.isdir(save_point):
                os.mkdir(save_point)
            best_acc = acc

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))

print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))
