import argparse
import os

import cv2
import matplotlib.pyplot as plt
import torchvision.utils as vutils
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torch.nn import functional as F
import tqdm
import numpy as np
from PIL import Image
from datasets import VITONDataset
from models.never_parse_ds_studentmodel import NPDNet
from torchvision import transforms, utils
from utils import lpips
from utils.utils import AverageMeter
from torchvision import transforms, utils
from torch.utils import data
from torchvision.utils import save_image
cudnn.benchmark = True



def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str,default="TEST_PAIR")
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--ratio', type=int, default=0)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--load_height', type=int, default=256)
    parser.add_argument('--load_width', type=int, default = 192)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--add_compare', action='store_true')
    # parser.add_argument('--dataset_dir', type=str, default='data')
    parser.add_argument('--dataset_imgpath', type=str, default='data/test_data')
    parser.add_argument('--dataset_list', type=str, default=r'data/test_data/test_pairs2.txt')
    parser.add_argument('--checkpoint_dir', type=str, default=r'checkpoint/pf_b_2l2e-05m_35/pf_b_2l2e-05m_35_data.pt')
    parser.add_argument('--save_dir', type=str, default='./results/')
    parser.add_argument('--multi_flows', type=int, default=6)
    parser.add_argument('--display_freq', type=int, default=100)
    opt = parser.parse_args()
    return opt

def atest(opt, net):
    test_dataset = VITONDataset(opt)
    test_loader = data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle,num_workers=opt.workers)
    with torch.no_grad():
        for i, inputs in enumerate(tqdm.tqdm(test_loader)):
            img_names = inputs['img_name']
            # cloth_names = inputs['c_name']['paired']
            img = inputs['img'].cuda()
            cloth_img = inputs['cloth'].cuda()
            img = F.interpolate(img, size=(256, 192), mode='bilinear')
            cloth_img = F.interpolate(cloth_img, size=(256, 192), mode='bilinear')
            # img_agnostic = F.interpolate(img_agnostic, size=(256, 192), mode='bilinear')
            # pose = F.interpolate(pose, size=(256, 192), mode='bilinear')
            # ref_input = torch.cat((pose, img_agnostic), dim=1)
            tryon_result = net(img,cloth_img).detach()
            for j in range(tryon_result.shape[0]):
                save_image(tryon_result[j:j+1], os.path.join(opt.save_dir, opt.name, "vis_viton_out", img_names[j]), nrow=1, normalize=True, range=(-1,1))
                print(f'save_dir:{os.path.join(opt.save_dir, opt.name, "vis_viton_out", img_names[j])}')

def return_tryon(img_path,cloth_img):
    img = Image.open()

def transformdata(arr_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    arr = Image.open(arr_path).convert('RGB')
    arr = transforms.Resize(192, interpolation=2)(arr)
    arr = transform(arr)
    arr = arr.unsqueeze(0)
    arr = F.interpolate(arr, size=(256, 192), mode='bilinear')
    return arr
def main():
    opt = get_opt()
    print(opt)
    if not os.path.exists(os.path.join(opt.save_dir, opt.name, "vis_viton_out")):
        os.makedirs(os.path.join(opt.save_dir, opt.name,"vis_viton_out"))

    sdafnet = torch.load(opt.checkpoint_dir).to('cpu')
    # img = cv2.imread('data/my_data/clothes/000010_1.jpg')
    # cloth = cv2.imread('data/my_data/test_img/000048_0.jpg')
    img_path = r'./data\test_data\clothes\000057_1.jpg'
    cloth_path = r'./data\test_data\test_img\000118_0.jpg'
    # img = transformdata('data/my_data/test_img/000048_0.jpg')
    img = transformdata(img_path)
    cloth = transformdata(cloth_path)

    tryon_result = sdafnet(img, cloth).detach()
    # save_image(tryon_result,'1.png', nrow=1, normalize=True, range=(-1,1))
    # 对图像数据进行归一化处理
    range = (-1,1)
    tryon_result = (tryon_result - range[0]) / (range[1] - range[0])

    # 创建图像的网格，一行显示 nrow 张图像
    grid_img = vutils.make_grid(tryon_result, nrow=1, normalize=True)
    # 将 Tensor 数据转换为 NumPy 数组，并交换通道维度顺序
    grid_img = grid_img.permute(1, 2, 0).numpy()

    # 显示图像
    plt.imshow(grid_img)
    plt.axis('off')
    plt.show()
    # img1 = tryon_result.squeeze(0).permute(1,2,0).cpu().numpy()
    # img1 = np.round(img1/np.max(img1)*255)
    # img1 = np.uint8(img1)
    # plt.imshow(img1)
    # plt.show()
    # print(cloth.shape)
    # print(tryon_result)

if __name__ == '__main__':
    main()