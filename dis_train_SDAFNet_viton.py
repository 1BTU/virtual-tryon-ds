import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import tqdm
import random
import torch
import argparse
from torch import nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from datasets import VITONDataset
from models.sdafnet import SDAFNet_Tryon
from models.ds_studentmodel import student_SDAFet_Tryon
from models import external_function
from utils import lpips
from utils.utils import AverageMeter
from torchvision import transforms, utils
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from test_SDAFNet_viton import test
from distill.dsloss import KDLoss
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='VITON')
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--ratio', type=int, default=0.5)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--load_height', type=int, default=256)
    parser.add_argument('--load_width', type=int, default=192)

    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--iter', type=int, default=30001)
    parser.add_argument('--shuffle', action='store_false')
    parser.add_argument('--mode', type=str, default='train')
    # parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--dataset_imgpath', type=str, default='data/VITON/VITON_train')
    parser.add_argument('--dataset_list', type=str, default='data/VITON/train.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/0.8init_pruned_model.pt')
    parser.add_argument('--save_dir', type=str, default='./results/')
    parser.add_argument('--display_freq', type=int, default=200)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--multi_flows', type=int, default=6)
    parser.add_argument('--alpha', type=int, default=0.3)
    parser.add_argument('--temperture', type=int, default=1)

    opt = parser.parse_args()
    return opt

def train(opt,teacher_model ,student_model):
    train_dataset = VITONDataset(opt)
    train_loader = data.DataLoader(train_dataset, batch_size=opt.batch_size,drop_last=True, shuffle=opt.shuffle,num_workers=opt.workers)

    #criterion
    criterion_L1 = nn.L1Loss()
    criterion_percept = lpips.exportPerceptualLoss(model="net-lin", net="vgg", use_gpu=True)
    criterion_style = external_function.VGGLoss().cuda()
    #optimizer
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=opt.lr)
    #scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
    epoches = 200
    iterations = 0
    # torch.save(net.state_dict(), "savemodel.pt")
    parmcage = f'ds_b_{opt.batch_size}l{opt.lr}r{opt.ratio}t{opt.temperture}'
    save_dir = f'results/{parmcage}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    writer = SummaryWriter(f'./logs/{parmcage}')
    # 断点续训
    if opt.resume:
        ptname = f'ds_b_4l2e-05r0.25iter169001opm.pt'
        # ptname = f'{parmcage}iter{opt.iter}opm.pt'
        resume_path = opt.save_dir+f"{parmcage}/{ptname}"
        if os.path.isfile(resume_path):
            print(f"resume form checkpoint:",resume_path)
            checkpoint = torch.load(resume_path)
            student_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim'])
            # opt.load_state_dict(checkpoint['opt'])
            start_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            # iterations = checkpoint['iterations']
            print('Resuming from epoch %d with loss %.3f' % (start_epoch, loss))
        else:
            start_epoch = 0
    else:
        start_epoch = 0


    for epoch in range(start_epoch,epoches):
        loss_l1_avg = AverageMeter()
        loss_vgg_avg = AverageMeter()


        for i, inputs in enumerate(tqdm.tqdm(train_loader)):
            # with torch.no_grad():
            iterations += 1
            img_names = inputs['img_name']
            c_names = inputs['c_name']['paired']
            img = inputs['img'].cuda()
            img_agnostic = inputs['img_agnostic'].cuda()
            pose = inputs['pose'].cuda()
            cloth_img = inputs['cloth']['paired'].cuda()
            img =  F.interpolate(img, size=(256, 192), mode='bilinear')
            ref_input = torch.cat((pose, img_agnostic), dim=1)
            result_tryon, results_all = student_model(ref_input, cloth_img, img_agnostic, return_all=True)
            # result_tryon, results_all = student_model(ref_input, cloth_img, img_agnostic, return_all=True)
            teacher_rf_fpns = teacher_model.reference_FPN(teacher_model.reference_features(ref_input))
            teacher_sf_fpns = teacher_model.source_FPN(teacher_model.source_features(cloth_img))
            student_rf_fpns = student_model.reference_FPN(student_model.reference_features(ref_input))
            student_sf_fpns = student_model.source_FPN(student_model.source_features(cloth_img))
            epsilon = 0.001
            rf_fpn_loss = 0
            sf_fpn_loss = 0
            loss_all = 0
            num_layer = 5
            for num in range(num_layer):
                #kd loss
                rf_fpn_loss += KDLoss(teacher_rf_fpns[num],student_rf_fpns[num],opt.temperture)
                sf_fpn_loss += KDLoss(teacher_sf_fpns[num],student_sf_fpns[num],opt.temperture)
                #orginal loss
                cur_img = F.interpolate(img, scale_factor=0.5**(4-num), mode='bilinear')
                loss_l1 = criterion_L1(results_all[num], cur_img.cuda())
                if num == 0:
                    cur_img = F.interpolate(cur_img, scale_factor=2, mode='bilinear')
                    results_all[num] = F.interpolate(results_all[num], scale_factor=2, mode='bilinear')
                loss_perceptual = criterion_percept(cur_img.cuda(),results_all[num]).mean()
                loss_content, loss_style = criterion_style(results_all[num], cur_img.cuda())
                loss_vgg = loss_perceptual+100*loss_style+0.1*loss_content
                loss_all = loss_all + (num+1) * loss_l1 + (num + 1)  * loss_vgg

            loss = opt.alpha*loss_all+(1-opt.alpha)/2*(opt.temperture**2)*(rf_fpn_loss)+(1-opt.alpha)/2*(opt.temperture**2)*(sf_fpn_loss)
            loss_l1_avg.update(loss.item())
            loss_vgg_avg.update(loss_vgg.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("\n[%d %d][%d] l1_loss:%.4f %.4f vgg_loss:%.4f %.4f "%(epoch,epoches,iterations,loss_l1.item(),loss_l1_avg.avg,loss_vgg.item(),loss_vgg_avg.avg))

        # if iterations%opt.display_freq ==1:
        #     parse_pred = torch.cat([result_tryon,img],2)
        #     parse_pred = F.interpolate(parse_pred, size=(512, 192), mode='bilinear')
        #     utils.save_image(
        #             parse_pred,
        #             f"{os.path.join(opt.save_dir, opt.name)}/log_sample/{str(iterations).zfill(6)}_daf_viton_s1fine_{str(opt.name)}.jpg",
        #             nrow=6,
        #             normalize=True,
        #             range=(-1, 1),
        #         )
            if iterations % 1000 == 1:
                torch.save(
                    {
                        "state_dict": student_model.state_dict(),
                        "optim": optimizer.state_dict(),
                        "opt": opt,
                        "epoch": epoch,
                        'loss': loss_vgg_avg.avg,
                        'iterations':iterations

                    },
                    save_dir + f"/{parmcage}iter{iterations}opm.pt",
                )
                print('save_path:', opt.save_dir+f"/{parmcage}iter{iterations}opm.pt")
            if iterations%opt.save_freq == 1:

                writer.add_scalar(f"{parmcage}", loss_vgg_avg.avg, iterations)

    #plot loss

        # save_path = f"{save_dir}/{parmcage}e{epoch}"
        # torch.save(net.state_dict(),save_path)
        # print('save_path:',save_path)
        scheduler.step()
    writer.close()

def main():
    opt = get_opt()

    if not os.path.exists(os.path.join(opt.save_dir, opt.name)):
        os.makedirs(os.path.join(opt.save_dir, opt.name))
        os.makedirs(os.path.join(opt.save_dir, opt.name,'log_sample'))

    teacher_model = SDAFNet_Tryon(ref_in_channel=6)  # 网络
    teacher_model = teacher_model.cuda()
    teacher_model.load_state_dict(torch.load("./checkpoint/model.pt"))

    student_model = student_SDAFet_Tryon(ref_in_channel=6)
    student_model = student_model.cuda()
    student_model.train()


    train(opt, teacher_model,student_model)




if __name__ == '__main__':
    main()