# noinspection PyUnresolvedReferences
import os
import datetime
from hat_block_distillation import HAT
import numpy as np
from torch.utils.data import Dataset
# noinspection PyUnresolvedReferences
from torch.utils import data as data
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.data.data_util import paired_paths_from_lmdb
from basicsr.data.transforms import augment, paired_random_crop
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as f
from utils import calc_psnr, fun_convert, data_Augmentation, convert_rgb_to_y, rgb2ycbcr, EvalDataset
import torchvision.transforms
from SCAN_for_distillation import SCAN, CALayer_noSigmoid
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch


gt_folder = "dataset/DF_HR_sub.lmdb"
lq_folder = "dataset/DF_LR_bicubic_X3_sub.lmdb"
db_paths = [lq_folder, gt_folder]
client_keys = ['lq', 'gt']
file_client = FileClient(db_paths=db_paths, client_keys=client_keys, backend='lmdb')


class dataset_lmdb(Dataset):
    def __init__(self, gt_size, scale):
        super(dataset_lmdb).__init__()
        self.paths = paired_paths_from_lmdb(db_paths, client_keys)
        self.scale = scale
        self.gt_size = gt_size

    def __getitem__(self, index):
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        img_bytes = file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        img_bytes = file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        # random crop
        img_gt, img_lq = paired_random_crop(img_gt, img_lq, self.gt_size, self.scale, gt_path)
        # data augmentation
        img_gt, img_lq = augment([img_gt, img_lq], hflip=True, rotation=True)
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        return img_lq, img_gt

    def __len__(self):
        return len(self.paths)


if __name__ == '__main__':
    epoch_num = 500
    scale = 3
    batch_size = 64
    continue_train = False
    trans = torchvision.transforms.ToTensor()
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
    train_dataset = dataset_lmdb(gt_size=144,
                                 scale=scale)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=20,
                                  drop_last=True,
                                  pin_memory=True)
    eval_dataset = EvalDataset('dataset/DIV2K_eval_x3.h5')
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    torch.cuda.manual_seed(seed=123)
    iter_number_epoch = (len(train_dataset) // batch_size) * epoch_num

    model_teacher = HAT(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    )
    state_dic = torch.load('HAT_pth\\HAT_SRx{}_ImageNet-pretrain.pth'.format(scale))
    model_teacher.load_state_dict(state_dic['params_ema'])
    model_teacher.upsample = nn.Identity()
    model_teacher.conv_last = nn.Identity()

    model_student = SCAN(num_in_ch=3, num_out_ch=3, scale=3, num_feat=48, depths=[2,2,2,2,2,2,2], d_atten=64, conv_groups=2)

    proj1, proj2, proj3, proj4, proj5, proj6 = (CALayer_noSigmoid(channel=180), CALayer_noSigmoid(channel=180), CALayer_noSigmoid(channel=180),
                                                CALayer_noSigmoid(channel=180), CALayer_noSigmoid(channel=180), CALayer_noSigmoid(channel=180))

    proj1, proj2, proj3, proj4, proj5, proj6 = proj1.to(device), proj2.to(device), proj3.to(device), proj4.to(device), proj5.to(device), proj6.to(device)
    proj = CALayer_noSigmoid(channel=64)
    proj = proj.to(device)
    optimizer_teacher = optim.Adam(params=proj.parameters(), lr=1e-6, betas=(0.9, 0.99), eps=1e-8)
    optimizer_student = optim.Adam(params=model_student.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-8)
    scheduler_teacher = optim.lr_scheduler.ConstantLR(optimizer=optimizer_teacher, factor=1.)
    '''chose your optim method, Cosdecay learning rate is recommended'''
    scheduler_student = optim.lr_scheduler.MultiStepLR(optimizer=optimizer_student,
                                                       milestones=[iter_number_epoch // 4, (iter_number_epoch // 5) * 2,
                                                                   (iter_number_epoch // 20) * 9,
                                                                   (iter_number_epoch // 40) * 19],
                                                       gamma=0.5)
    for param in model_teacher.parameters():
        param.requires_grad = False
    model_student = model_student.to(device)
    model_teacher = model_teacher.to(device)
    loss_function = nn.L1Loss()
    loss_function = loss_function.to(device)
    print("总迭代次数：{}".format(iter_number_epoch))
    print("训练数据集长度：{}".format(len(train_dataset)))
    print("验证数据集长度：{}".format(len(eval_dataset)))
    epoch_factor = 1
    # epoch_factor is named distillation interval in our papper, when the total iteration number is 1000000, 15 is recommended
    alpha = 1
    # alpha = any other number smaller than 1
    iter_num = 0
    for epoch in range(0, epoch_num):
        model_student.train()
        model_teacher.train()
        print("\n")
        print("\n---------------------第{}轮训练开始---------------------".format(epoch + 1))
        print("------------时间{}------------".format(datetime.datetime.now()))
        for data in train_dataloader:
            inputs, labels = data
            inputs, labels = data_Augmentation(inputs, labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            if epoch % epoch_factor == 0 and epoch > 1:
                preds, SCAN_feat = model_student(inputs)
                inter1, inter2, inter3, inter4, inter5, inter6, inter7 = model_teacher(inputs)
                inter1, inter2, inter3, inter4, inter5, inter6, inter7 = proj1(inter1), proj2(inter2), proj3(inter3), proj4(inter4), proj5(inter5), proj6(inter6), proj(inter7)
                loss_co = (alpha * (loss_function(torch.mean(SCAN_feat[0], dim=1, keepdim=True), torch.mean(inter1, dim=1, keepdim=True)) +
                                   loss_function(torch.mean(SCAN_feat[1], dim=1, keepdim=True), torch.mean(inter2, dim=1, keepdim=True)) +
                                   loss_function(torch.mean(SCAN_feat[2], dim=1, keepdim=True), torch.mean(inter3, dim=1, keepdim=True)) +
                                   loss_function(torch.mean(SCAN_feat[3], dim=1, keepdim=True), torch.mean(inter4, dim=1, keepdim=True)) +
                                   loss_function(torch.mean(SCAN_feat[4], dim=1, keepdim=True), torch.mean(inter5, dim=1, keepdim=True)) +
                                   loss_function(torch.mean(SCAN_feat[5], dim=1, keepdim=True), torch.mean(inter6, dim=1, keepdim=True)) +
                                   loss_function(torch.mean(SCAN_feat[6], dim=1, keepdim=True), torch.mean(inter7, dim=1, keepdim=True))) +
                           (1 - alpha) * loss_function(preds, labels))
                optimizer_student.zero_grad()
                loss_co.backward()
                optimizer_student.step()
                scheduler_student.step()
            else:
                preds, SCAN_feat = model_student(inputs)
                loss1 = loss_function(labels, preds)
                optimizer_student.zero_grad()
                loss1.backward()
                optimizer_student.step()
                scheduler_student.step()
            iter_num = iter_num + 1
        psnr_in_batch_log = 0
        model_student.eval()
        for data in eval_dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                preds = model_student(inputs)[0]
            output = preds.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            labels = labels.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            output = rgb2ycbcr(np.array(output).astype('float32'), y_only=True)
            labels = rgb2ycbcr(np.array(labels).astype('float32'), y_only=True)
            labels, output = trans(labels), trans(output)
            psnr = calc_psnr(output[:, scale:-scale, scale:-scale], labels[:, scale:-scale, scale:-scale])
            psnr_in_batch_log = psnr + psnr_in_batch_log

        psnr_in_eval = psnr_in_batch_log / len(eval_dataset)
        print('eval_psnr_every_epoch:{:.6f}'.format(psnr_in_eval))
        print("---------------------第{}轮训练结束---------------------".format(epoch + 1))
        print("------------时间{}------------".format(datetime.datetime.now()))

        '''save your model'''
        # if epoch > (epoch_num - 5):
        #     torch.save(model_student,
        #                "Result_distill\\{}_{}.pth".format(writer_name, epoch))


