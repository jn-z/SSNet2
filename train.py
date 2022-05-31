import os
import pdb
import sys
from timeit import default_timer as timer
import torch
import numpy as np
from loss import CtdetLoss
from torch.utils.data import DataLoader
from dataset import ctDataset
from DLAnet import DlaNet
import torch.nn as nn
import torch.optim as optim
import argparse
from utils import snapshot, setup_logs
run_name = "zjn-cennet"
print(run_name)
class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def state_dict(self):
        self.optimizer.state_dict()

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr
def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--n-warmup-steps', type=int, default=50)
    parser.add_argument('--logging-dir', required=True,
                        help='model save directory')
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()
    print("Use CUDA? ", use_gpu)
    logger = setup_logs(args.logging_dir, run_name)  # setup logs
    model = DlaNet(34)

    if (use_gpu):
        # os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
        # model = nn.DataParallel(model)
        # print('Using ', torch.cuda.device_count(), "CUDAs")
        print('cuda', torch.cuda.current_device(), torch.cuda.device_count())
        device = torch.device("cuda")
        model.cuda()
    else:
        device = torch.device("cpu")

    loss_weight = {'hm_weight':1, 'wh_weight':0.1, 'reg_weight':0.1}
    criterion = CtdetLoss(loss_weight)
    
    model.train()

    #learning_rate = 5e-4
    #num_epochs = args.epochs

    # different learning rate
    # params=[]
    # params_dict = dict(model.named_parameters())
    # for key, value in params_dict.items():
    #     params += [{'params':[value], 'lr':learning_rate}]

    #optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=1e-4)
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),args.n_warmup_steps)
    # split into training and testing set
    full_dataset = ctDataset()
    full_dataset_len = full_dataset.__len__()
    print("Full dataset has ",  full_dataset_len, " images.")
    train_size = int(0.8 * full_dataset_len)
    test_size = full_dataset_len - train_size
    torch.manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    print("Training set and testing set has: ", train_dataset.__len__(), \
            " and ", test_dataset.__len__(), " images respectively.")
    logger.info('===> loading train, validation and eval dataset')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('### Model summary below###\n {}\n'.format(str(model)))
    logger.info('===> Model total parameter: {}\n'.format(model_params))
    best_test_loss = np.inf 

    loss_log = np.empty((0, 3))

    best_acc = 0
    best_loss = np.inf
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        epoch_timer = timer()
        model.train()
        # if epoch == 45:
        #     learning_rate = learning_rate * 0.1
        # if epoch == 60:
        #     learning_rate = learning_rate * (0.1 ** 2)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = learning_rate

        total_loss = 0.0
        #pdb.set_trace()
        for i, sample in enumerate(train_loader):
            for k in sample:
                if k != 'ori_index':
                   sample[k] = sample[k].to(device=device, non_blocking=True)

            pred = model(sample['input'])
            loss = criterion(pred, sample)    
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr = optimizer.update_learning_rate()

            if (i+1) % 5 == 0:

                print ('Epoch [%d/%d], Iter [%d/%d] lr: %.4f, Loss: %.4f, average_loss: %.4f'
                %(epoch, args.epochs, i+1, len(train_loader), lr, loss.data, total_loss / (i+1)))

        # validation
        validation_loss = 0.0
        model.eval()
        for i, sample in enumerate(test_loader):
            if use_gpu:
                for k in sample:
                    if k != 'ori_index':
                       sample[k] = sample[k].to(device=device, non_blocking=True)
            
            pred = model(sample['input'])
            loss = criterion(pred, sample)   
            validation_loss += loss.item()
        validation_loss /= len(test_loader)
        
        print('Epoch [%d/%d] Validation loss %.5f' % (epoch, args.epochs, validation_loss))

        loss_log = np.append(loss_log, [[epoch, total_loss / len(train_loader), validation_loss]], axis=0)
        np.savetxt('../loss_log.csv', loss_log, delimiter=',')
        # Save
        # if validation_loss > best_acc:
        #     best_acc = max(validation_loss, best_acc)
        if validation_loss < best_loss:
            best_loss = min(validation_loss, best_loss)
            snapshot(args.logging_dir, run_name, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'validation_loss': validation_loss,
                'optimizer': optimizer.state_dict(),
            })
            best_epoch = epoch + 1
        elif epoch - best_epoch > 2:
            optimizer.increase_delta()
            best_epoch = epoch + 1
        end_epoch_timer = timer()
        logger.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, args.epochs, end_epoch_timer - epoch_timer))
            

if __name__ == "__main__":
    main()
