import os
import sys
import ast
import argparse
import utils
from tqdm import tqdm
import torch.nn as nn
from utils import *
from scipy.stats import kendalltau
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from networks import FBNASNet
import matplotlib.pyplot as plt
import torch.optim as optim
from CenterLoss import CenterLoss

def get_args():
    parser = argparse.ArgumentParser("MixPath")
    # parser.add_argument('--exp_name', type=str, required=True, help='search model name')
    parser.add_argument('--m', type=int, default=2, help='num of selected paths as most') #default=2
    parser.add_argument('--shadow_bn', action='store_false', default=True, help='shadow bn or not, default: True')
    # parser.add_argument('--data_dir', type=str, default='/home/work/dataset/cifar', help='dataset dir')
    # parser.add_argument('--classes', type=int, default=10, help='classes')
    # parser.add_argument('--layers', type=int, default=12, help='num of MB_layers')
    # parser.add_argument('--kernels', type=list, default=[3, 5, 7], help='selective kernels')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='num of epochs') #default=200
    parser.add_argument('--seed', type=int, default=20190821, help='seed') #default=20190821
    parser.add_argument('--search_num', type=int, default=20, help='num of epochs') #default=1000
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--pm', type=float, default=0.2, help='probability of mutation')
    parser.add_argument('--pc', type=float, default=0.8, help='probability of corssover')
    parser.add_argument('--N', type=int, default=20, help='num of iterations')
    parser.add_argument('--popsize', type=int, default=50, help='num of populations')
    parser.add_argument('--centerLossEnable', type=bool, default=False, help='use center Loss or not')
    parser.add_argument('--Lambda', type=float, default=0.0005, help='center Loss coefficient')
    parser.add_argument('--AugumentEnable', type=bool, default=False, help='use data augument or not')
    parser.add_argument('--repeat_num', type=int, default=5, help='number of repeated trials')
    # parser.add_argument('--learning_rate_min', type=float, default=1e-8, help='min learning rate')
    # parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    # parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    # parser.add_argument('--train_interval', type=int, default=1, help='train to print frequency')
    # parser.add_argument('--val_interval', type=int, default=5, help='evaluate and save frequency')
    # parser.add_argument('--dropout_rate', type=float, default=0.2, help='drop out rate')
    # parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop_path_prob')
    # parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    # parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    # parser.add_argument('--resume', type=bool, default=False, help='resume')
    # # ******************************* dataset *******************************#
    # parser.add_argument('--data', type=str, default='cifar10', help='[cifar10, imagenet]')
    # parser.add_argument('--cutout', action='store_false', default=True, help='use cutout')
    # parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    # parser.add_argument('--resize', action='store_true', default=False, help='use resize')

    arguments = parser.parse_args()

    return arguments



def validate_cali(args, val_data, model, choice):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.0
    val_top1 = AvgrageMeter()
    val_top5 = AvgrageMeter()
    criterion = nn.NLLLoss() 
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs = inputs.to(device)
            targets = targets.type(torch.LongTensor).to(device)                
            outputs,_ = model(inputs, choice)

            loss = criterion(outputs, targets)
            val_loss += loss.item()
            prec1 = accuracy(outputs, targets, topk=(1,))
            n = inputs.size(0)
            val_top1.update(prec1[0], n)
    return val_top1.avg, val_loss / (step + 1)

def validate_cali_aug(args, inputs, targets, model, choice):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.0
    val_top1 = AvgrageMeter()
    val_top5 = AvgrageMeter()
    criterion = nn.NLLLoss()  
    with torch.no_grad():
        inputs = inputs.to(device)
        targets = targets.type(torch.LongTensor).to(device)
        # print(inputs.size())
        outputs,_ = model(inputs, choice)

        loss = criterion(outputs, targets)
        val_loss += loss.item()
        prec1 = accuracy(outputs, targets, topk=(1,))
        n = inputs.size(0)
        val_top1.update(prec1[0], n)
    return val_top1.avg, val_loss




check_dict = []
def validate_search(args, val_data, model):
    model.eval()
    choice_dict = {}
    val_loss = 0.0
    val_top1 = AvgrageMeter()
    val_top5 = AvgrageMeter()
    criterion = nn.NLLLoss()
    choice = random_choice(m=args.m)

    while choice in check_dict:
        print('Duplicate Index !')
        choice = random_choice(m=args.m)
    check_dict.append(choice)
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            outputs,_ = model(inputs, choice)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            prec1 = accuracy(outputs, targets, topk=(1,))
            n = inputs.size(0)
            val_top1.update(prec1[0], n)
    choice_dict['Low'] = choice['Low']
    choice_dict['Mid'] = choice['Mid']
    choice_dict['High'] = choice['High']
    choice_dict['val_loss'] = val_loss / (step + 1)
    choice_dict['val_top1'] = val_top1.avg

    return choice_dict


def train(args, epoch, train_data, model, criterion, optimizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    train_loss = 0.0
    top1 = AvgrageMeter()
    top5 = AvgrageMeter() 
    transform = build_tranforms()
    rta = RepeatedTrialAugmentation(transform, m=args.repeat_num)
    
    for step, (inputs, targets) in enumerate(train_data):
        inputs = inputs.to(device)
        targets = targets.type(torch.LongTensor).to(device)
        if args.AugumentEnable:
            inputs, targets = rta(inputs, targets)
        optimizer.zero_grad()
        choice = random_choice(m=args.m)
        outputs, _ = model(inputs, choice)

        loss = criterion(outputs, targets)
        loss.backward()

        for p in model.parameters():
            if p.grad is not None and p.grad.sum() == 0:
                p.grad = None

        # nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        prec1 = accuracy(outputs, targets, topk=(1,))
        # print(prec1)
        n = inputs.size(0)
        top1.update(prec1[0], n)
        optimizer.step()
        train_loss += loss.item()

        postfix = {'loss': '%.6f' % (train_loss / (step + 1)), 'top1': '%.3f' % top1.avg}

        train_data.set_postfix(postfix)


def validate(args, val_data, model):
    model.eval()
    val_loss = 0.0
    val_top1 = AvgrageMeter()
    val_top5 = AvgrageMeter()
    criterion = nn.NLLLoss()

    with torch.no_grad():
        top1_m = []
        top5_m = []
        loss_m = []
        for _ in range(20):
            choice = random_choice(m=args.m)
            for step, (inputs, targets) in enumerate(val_data):
                outputs,_ = model(inputs, choice)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                prec1 = accuracy(outputs, targets, topk=(1,))
                n = inputs.size(0)
                val_top1.update(prec1[0], n)
            top1_m.append(val_top1.avg), loss_m.append(val_loss / (step + 1))

    return np.mean(top1_m), np.mean(loss_m)


def separate_bn_params(model):
    bn_index = []
    bn_params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_index += list(map(id, m.parameters()))
            bn_params += m.parameters()
    base_params = list(filter(lambda p: id(p) not in bn_index, model.parameters()))
    return base_params, bn_params

class eeg2normalLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, eegdatset):
        self.eegdatset = eegdatset
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        if self.eegdatset[index]['data'].dtype == 'float32':
            data = torch.tensor(self.eegdatset[index]['data']).unsqueeze(0)
        else:
            data = self.eegdatset[index]['data'].unsqueeze(0)
        labels = np.uint8(self.eegdatset[index]['label'])
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.eegdatset)


def nas_phase(net,
    trainData,
    valData,
    classes=4,
    sampler=None,
    supernet_path=None):
    
    args = get_args()
    # print(args)
    set_seed(args.seed)    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.NLLLoss()
    base_params, bn_params = separate_bn_params(net)   
    optimizer = torch.optim.Adam([
        {'params': base_params, 'weight_decay': 0.0},
        {'params': bn_params, 'weight_decay': 0.0}],
        lr=args.learning_rate)    
    start_epoch = 0     
    
    trainData = eeg2normalLoader(trainData)     
       
    valData = eeg2normalLoader(valData)
    
    train_queue = torch.utils.data.DataLoader(trainData, batch_size= args.batch_size,
                            shuffle= True, num_workers=8)
    valid_queue = torch.utils.data.DataLoader(valData, batch_size= len(valData),
                            shuffle= False, num_workers=8) 
    
    
    top1 = []
    loss = []
    for epoch in range(start_epoch, args.epochs):
        train_data = tqdm(train_queue)
        train(args, epoch, train_data, net, criterion=criterion, optimizer=optimizer)   
    # print(supernet_path)
    torch.save(net.state_dict(), supernet_path)
    
    candidate_list = []
    cali_bn_acc= []
    cali_bn_loss= []
    choice_list = traverse_choice(args.m)
    # net.load_state_dict(torch.load(supernet_path))
    # net = net.to(device)
    # with tqdm(total=len(choice_list)) as pbar:
    #     for epoch in range(len(choice_list)):           
    #         choice = choice_list[epoch]
    #         top1_acc, val_loss = validate_cali(args, valid_queue, net, choice)
    #         cali_bn_acc.append(top1_acc)
    #         cali_bn_loss.append(val_loss)
    #         candidate_list.append(choice)
    #         pbar.update(1)
    # transform = build_tranforms()
    # rta = RepeatedTrialAugmentation(transform, m=args.repeat_num)
    with tqdm(total=len(choice_list)) as pbar:
        for epoch in range(len(choice_list)):           
            net.load_state_dict(torch.load(supernet_path))
            net = net.to(device)
            with torch.no_grad():
                choice = choice_list[epoch]
                net.train()
                for inputs, targets in valid_queue:
                    inputs = inputs.to(device)
                    targets = targets.type(torch.LongTensor).to(device)
                    # if args.AugumentEnable:
                    #     inputs, targets = rta(inputs, targets)
                    net(inputs, choice)
                    # print(inputs.size())
                    top1_acc, val_loss = validate_cali_aug(args, inputs, targets, net, choice)
                # top1_acc, val_loss = validate_cali(args, valid_queue, net, choice)
                cali_bn_acc.append(top1_acc)
                cali_bn_loss.append(val_loss)
                candidate_list.append(choice)
            pbar.update(1)
    # with torch.no_grad():
    #     choice = random_choice(m=args.m)
    #     net.train()
    #     for inputs, targets in valid_queue:
    #         net(inputs, choice)
       
    # sort candidate_dict
    
    sorted_index = sort_list_with_index(cali_bn_acc)
    # sorted_index = sort_list_with_index(cali_bn_loss)
    # print(cand_dict[1])
    opt_cand = candidate_list[sorted_index[-1]]
    
    for i in range(len(cali_bn_acc)):
        cali_bn_acc[i] = cali_bn_acc[i].detach().cpu().numpy()
    ######  获取center loss 初始中心点 ######
    # net.load_state_dict(torch.load(supernet_path))
    # net.eval()
    # dataLoader = torch.utils.data.DataLoader(trainData, batch_size= len(trainData), shuffle= False)
    # choice = random_choice(m=args.m)
    # with torch.no_grad():
    #     for inputs, targets in dataLoader:
    #         # print(data.shape)
    #         label_list = targets.numpy()
    #         num_classes = len(set(label_list))
    #         init_centers = np.zeros([num_classes,2304])
    #         inputs = inputs.to(device)
    #         preds,feats = net(inputs, opt_cand)
    #         for i in range(num_classes):
    #             idx = np.where(label_list==list(set(label_list))[i])[0]
    #             init_centers[i,:] = np.mean(feats[idx,:].detach().cpu().numpy(),axis=0)
    ######     
    
    
    # opt_cand = candidate_list[sorted_index[0]]
    # opt_choice = {}
    # opt_choice['low'] = opt_cand['low']
    # opt_choice['Mid'] = opt_cand['Mid']
    # opt_choice['High'] = opt_cand['High']
    return opt_cand, cali_bn_acc
        
    
    
    
    
    
        
    
