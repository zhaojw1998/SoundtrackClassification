import os
import glob
import time
import torch
import numpy as np
from config import params
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from AudioDataLoader import audioDataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

#from dataloaders.dataset import VideoDataset

from network import logistic, neural_network, resnet, vgg
from torchsummary import summary


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


def train(model, train_dataloader, epoch, criterion, optimizer, writer, acclist):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()
    for step, (inputs, labels) in enumerate(train_dataloader):
        data_time.update(time.time() - end)
        inputs = inputs.cuda().float()
        labels = labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if (step + 1) % params['display'] == 0:
            print('-------------------------------------------------------')
            for param in optimizer.param_groups:
                print('lr1: ', param['lr'])
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(train_dataloader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)
            print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
            print(print_string)
            print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                top1_acc=top1.avg,
                top5_acc=top5.avg)
            print(print_string)
    acclist.append('Top-1: ' + str(top1.avg) + '; Top-5: ' + str(top5.avg) + '\n')
    #writer.add_scalar('train/loss_epoch', losses.avg, epoch)
    #writer.add_scalar('train/top1_acc_epoch', top1.avg, epoch)
    #writer.add_scalar('train/top5_acc_epoch', top5.avg, epoch)



def validation(model, val_dataloader, epoch, criterion, writer, acclist):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()
    with torch.no_grad():
        total_outputs=torch.empty(0,params['num_classes'])
        for step, (inputs, labels) in enumerate(val_dataloader):
            data_time.update(time.time() - end)
            inputs = inputs.cuda().float()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if (step + 1) % params['display'] == 0:
                print('----validation----')
                print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(val_dataloader))
                print(print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val,
                    batch_time=batch_time.val)
                print(print_string)
                print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
                print(print_string)
                print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                    top1_acc=top1.avg,
                    top5_acc=top5.avg)
                print(print_string)
            if (epoch+1) % 1 == 0:
                total_outputs = torch.cat((total_outputs, outputs.detach().cpu()), dim=0)
    #if (epoch+1) % 5 == 1:#?????????????????????????????????????????????????????????????????????????????????????????????????????????
        #print(total_outputs.shape)
    np.save('\\npy_files\\resNet_output_Kinetics600.npy', total_outputs)
    acclist.append('Top-1: ' + str(top1.avg) + '; Top-5: ' + str(top5.avg) + '\n')
        #np.save('new_output_labels_Kinetics32.npy', F.softmax(outputs.detach()).cpu())
    #writer.add_scalar('val/loss_epoch', losses.avg, epoch)
    #writer.add_scalar('val/top1_acc_epoch', top1.avg, epoch)
    #writer.add_scalar('val/top5_acc_epoch', top5.avg, epoch)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

def main():
    cudnn.benchmark = True
    """
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logdir = os.path.join(params['log'], cur_time)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    """
    writer = None#SummaryWriter(logdir)

    gpu_num = torch.cuda.device_count()

    print("Loading " + " dataset")
    x_train = np.load('datalists\\train_audio_list.npy')
    y_train = np.load('datalists\\train_label_list.npy')
    x_val = np.load('datalists\\val_audio_list.npy')
    y_val = np.load('datalists\\val_label_list.npy')
    y_train, y_val = map(torch.tensor, (y_train, y_val))
    y_train = y_train.long()
    y_val = y_val.long()
    """    
    x_train, y_train, x_val, y_val = map(torch.tensor, (x_train, y_train, x_val, y_val))
    x_train = x_train.float()
    x_val = x_val.float()
    """

    train_dataloader = DataLoader(audioDataLoader('train', x_train, y_train),
                                  batch_size=params['batch_size'] * gpu_num,
                                  shuffle=True,
                                  num_workers=params['num_workers'],
                                  drop_last=False)
    val_dataloader = DataLoader(audioDataLoader('val', x_val, y_val),
                                batch_size=params['batch_size'] * gpu_num,
                                num_workers=params['num_workers'],
                                drop_last=False)

    print("load model")
    #model = logistic.Logistic()
    #model = neural_network.NeuralNetwork()
    model = resnet.resnet50()
    #model = vgg.vgg16()
    #model.apply(weight_init)

    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9)#, weight_decay=5e-4)
    #optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])#, weight_decay=5e-4)

    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    # exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
    
    if params['resume_epoch'] != 0:
        models = sorted(glob.glob(os.path.join(save_dir_root, 'model', 'model_*')))
        model_id = int(models[-1].split('_')[-1]) if models else 0
    else:
        models = sorted(glob.glob(os.path.join(save_dir_root, 'model', 'model_*')))
        model_id = int(models[-1].split('_')[-1]) + 1 if models else 0


    save_dir = os.path.join(save_dir_root, 'model', 'model_' + '%04d' % (model_id))
    saveName = params['model_name'] + '-' + params['dataset'] #+ '-' + params['stream']
    
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    model = model.cuda()

    if params['check_model']:
        summary(model, (3,params['clip_len'],params['crop_size'], params['crop_size']))
        return

    model = nn.DataParallel(model, device_ids=list(range(gpu_num)))  # multi-Gpu

    if params['resume_epoch'] == 0:
        print("Training {} from scratch...".format(params['model_name']))
    else:
        checkpoint = torch.load(os.path.join(save_dir, saveName + '_epoch-' + str(params['resume_epoch']-1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, saveName + '_epoch-' + str(params['resume_epoch'] - 1) + '.pth.tar')))

        # model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['opt_dict'])
        # model.load_state_dict(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss().cuda()  # standard crossentropy loss for classification
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['step'], gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=params['milestones'], gamma=0.1, last_epoch=last_epoch)
    train_record = []
    val_record = []
    for epoch in range(params['resume_epoch'],params['epoch_num']):
        train(model, train_dataloader, epoch, criterion, optimizer, writer, train_record)
        with open('train_record.txt', 'w') as f:
            f.writelines(train_record)
        if (epoch+1) % 1 == 0:
            validation(model, val_dataloader, epoch, criterion, writer, val_record)
        with open('val_record.txt', 'w') as f:
            f.writelines(val_record)
        scheduler.step()
        if (epoch+1) % 1 == 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            checkpoint = os.path.join(save_dir, saveName + '_epoch-' + str(epoch) + '.pth.tar')
            torch.save({'epoch':epoch,
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict()},
                        checkpoint)
    



if __name__ == "__main__":
    main()