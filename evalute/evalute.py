import os

import yaml

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
from models.never_parse_ds_studentmodel import NPDNet
from models.sdafnet import SDAFNet_Tryon
from models import external_function
from utils import lpips
from utils.utils import AverageMeter
from torchvision import transforms, utils
from torch.utils import data
from config.load_config import load_config
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='VITON')
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--load_height', type=int, default=256)
    parser.add_argument('--load_width', type=int, default=192)
    parser.add_argument('--shuffle', action='store_false')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--dataset_imgpath', type=str, default='data/VITON/VITON_test')
    parser.add_argument('--dataset_list', type=str, default='data/VITON/test_pairs2.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/model.pt')
    parser.add_argument('--save_weights', type=str, default='/home/xxfs/document/ai/fuchuang/DAFlow/results/pf_b_2l5e-05_0.5_')
    parser.add_argument('--save_dir', type=str, default='./results/')
    parser.add_argument('--display_freq', type=int, default=200)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--multi_flows', type=int, default=6)
    # parser.add_argument('--config_path', type=str, default='/home/xxfs/document/ai/fuchuang/DAFlow/config/evalute/evalute1.yaml')
    opt = parser.parse_args()
    return opt

def evalute(opt, net):
    test_dataset = VITONDataset(opt)
    test_loader = data.DataLoader(test_dataset, batch_size=opt.batch_size,drop_last=True, shuffle=opt.shuffle,num_workers=opt.workers)

    #criterion
    criterion_L1 = nn.L1Loss()
    criterion_percept = lpips.exportPerceptualLoss(model="net-lin", net="vgg", use_gpu=True)
    criterion_style = external_function.VGGLoss().cuda()
    iterations = 0

    loss = 0.0
    for i, inputs in enumerate(tqdm.tqdm(test_loader)):
        with torch.no_grad():
            iterations+=1
            img = inputs['img'].cuda()
            cloth_img = inputs['cloth']['paired'].cuda()
            pose = inputs['pose'].cuda()
            img_agnostic = inputs['img_agnostic'].cuda()
            cloth_img_unpair = inputs['cloth']['unpaired'].cuda()
            img = F.interpolate(img, size=(256, 192), mode='bilinear')
            ref_input = torch.cat((pose, img_agnostic), dim=1)
            result_tryon, results_all = net(img, cloth_img, return_all=True)
            # result_tryon, results_all = net(ref_input, cloth_img, img_agnostic, return_all=True)
            loss_all = 0
            num_layer = 5
            for num in range(num_layer):
                cur_img = F.interpolate(img, scale_factor=0.5**(4-num), mode='bilinear')
                loss_l1 = criterion_L1(results_all[num], cur_img.cuda())
                if num == 0:
                    cur_img = F.interpolate(cur_img, scale_factor=2, mode='bilinear')
                    results_all[num] = F.interpolate(results_all[num], scale_factor=2, mode='bilinear')
                loss_perceptual = criterion_percept(cur_img.cuda(),results_all[num]).mean()
                loss_content, loss_style = criterion_style(results_all[num], cur_img.cuda())
                loss_vgg = loss_perceptual+100*loss_style+0.1*loss_content
                loss_all = loss_all + (num+1) * loss_l1 + (num + 1)  * loss_vgg
            loss += loss_all
    mean_loss = loss/len(test_dataset)
    return mean_loss
def evalute_train(opt, net):
    test_dataset = VITONDataset(opt)
    test_loader = data.DataLoader(test_dataset, batch_size=opt.batch_size,drop_last=True, shuffle=opt.shuffle,num_workers=opt.workers)

    #criterion
    criterion_L1 = nn.L1Loss()
    criterion_percept = lpips.exportPerceptualLoss(model="net-lin", net="vgg", use_gpu=True)
    criterion_style = external_function.VGGLoss().cuda()
    iterations = 0

    loss = 0.0
    for i, inputs in enumerate(tqdm.tqdm(test_loader)):
        with torch.no_grad():
            iterations += 1
            img = inputs['img'].cuda()
            img_agnostic = inputs['img_agnostic'].cuda()
            pose = inputs['pose'].cuda()
            cloth_img = inputs['cloth']['paired'].cuda()
            img = F.interpolate(img, size=(256, 192), mode='bilinear')
            ref_input = torch.cat((pose, img_agnostic), dim=1)
            result_tryon, results_all = net(ref_input, cloth_img, img_agnostic, return_all=True)
            loss_all = 0
            num_layer = 5
            for num in range(num_layer):
                cur_img = F.interpolate(img, scale_factor=0.5**(4-num), mode='bilinear')
                loss_l1 = criterion_L1(results_all[num], cur_img.cuda())
                if num == 0:
                    cur_img = F.interpolate(cur_img, scale_factor=2, mode='bilinear')
                    results_all[num] = F.interpolate(results_all[num], scale_factor=2, mode='bilinear')
                loss_perceptual = criterion_percept(cur_img.cuda(),results_all[num]).mean()
                loss_content, loss_style = criterion_style(results_all[num], cur_img.cuda())
                loss_vgg = loss_perceptual+100*loss_style+0.1*loss_content
                loss_all = loss_all + (num+1) * loss_l1 + (num + 1)  * loss_vgg
            loss += loss_all
    mean_loss = loss/len(test_dataset)
    return mean_loss
def build_model(weight_path,if_ori=True):
    if if_ori:
        sdafnet = SDAFNet_Tryon(ref_in_channel=6).cuda()
        sdafnet.load_state_dict(torch.load(weight_path), strict=False)

    else:
        sdafnet = NPDNet().cuda()
        # sdafnet.load_state_dict(torch.load(weight_path)['state_dict'], strict=False)
        sdafnet.load_state_dict(torch.load(weight_path)['state_dict'], strict=False)
    sdafnet.train()
    return sdafnet

    # sdafnet = torch.nn.DataParallel(sdafnet, device_ids=range(torch.cuda.device_count()))
def get_train_loss(file_paths):
    file_path = os.path.join('/DAFlow/config/evalute', file_paths[0].split('/')[-2] + '.txt')
    weight_list = []

    # 先读取txt文件内容
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            existing_content = file.read().strip()

        # 如果txt文件已有内容，则直接使用txt文件中的内容
        if existing_content:
            weight_list = [tuple(line.split(',')) for line in existing_content.split('\n')]

    # 如果weight_list为空，则读取文件路径并将其保存到txt文件
    if not weight_list:
        for weight_path in file_paths:
            # loss = torch.load(weight_path)['loss']
            loss = 1
            weight_list.append((weight_path, loss))
            print(weight_path, loss)
        weight_list = sorted(weight_list, key=lambda x: x[1])

        # 将weight_list的内容写入txt文件
        with open(file_path, 'w') as file:
            for item in weight_list:
                file.write(f"{item[0]},{item[1]}\n")
    cfg = [0,3,7,9,12,20,30,40]
    # weight_list = weight_list[cfg]
    # weigth_cfg = [weight_list[i] for i in cfg]
    weigth_cfg = weight_list[:50]
    print('cfg_weight_list:',weigth_cfg)
    return weigth_cfg
def get_test_loss(opt,weight_paths):
    weight_list = []

    for weight_path in weight_paths:
        dis_model = build_model(weight_path[0], False)
        # min_test_loss = evalute_train(opt, dis_model)
        min_test_loss = evalute(opt, dis_model)
        weight_list.append((weight_path[0], min_test_loss))
    weight_list = sorted(weight_list, key=lambda x: x[1])

    best_model_path = weight_list[0][0]
    save_path = best_model_path.replace('results','checkpoint')

    best_model_loss = weight_list[0][1]
    print("best_loss:",best_model_loss)
    best_model = build_model(best_model_path,False)
    best_model_parm = torch.load(best_model_path)
    # 使用os.path.dirname()获取路径中除最后一个斜杠之前的部分
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 将最后一个斜杠后的部分替换为新的名称
    model_name = opt.save_weights.split("/")[-1]
    best_model_data_pth = os.path.join(directory, f'{model_name}.pt')
    best_model_pth = os.path.join(directory, f'{model_name}_data.pt')
    best_model_parm_pth = os.path.join(directory, f'{model_name}_parms.pt')

    torch.save(best_model.state_dict(),best_model_data_pth)
    torch.save(best_model,best_model_pth)
    torch.save(best_model_parm,best_model_parm_pth)
    model_yaml_path = os.path.join(os.path.dirname(directory),'model.yaml')
    with open(model_yaml_path,'r') as file:
        data = yaml.safe_load(file)
        if data is None:
            data = {}
    data[f'{save_path}'] = f'loss={best_model_loss}'
    with open(model_yaml_path,'w') as file:
        yaml.dump(data,file)

def main():
    # opt = load_config(get_opt().config_path)
    opt = get_opt()
    print(opt)
    if not os.path.exists(os.path.join(opt.save_dir, opt.name)):
        os.makedirs(os.path.join(opt.save_dir, opt.name))
        os.makedirs(os.path.join(opt.save_dir, opt.name,'log_sample'))


    # = os.path.join(opt.save_weights,[0])

    cudnn.benchmark = True
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    weight_path_list = list(os.path.join(opt.save_weights,i) for i in os.listdir(opt.save_weights))
    # Computing loss of orignal model
    # ori_model = build_model(opt.checkpoint_dir)
    # loss_ori = evalute_train(opt, ori_model)
    # print(f"loss_ori={loss_ori}")
    min_train_loss = 100
    weight_paths = get_train_loss(weight_path_list)
    get_test_loss(opt, weight_paths)

    # print(f'loss_dis={loss_dis}')



if __name__ == '__main__':
    #该代码用于评估训练后的所有模型在测试集的损失，保存一个损失最小的模型到checkpoint中。
    main()