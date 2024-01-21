import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torch.nn import functional as F
import tqdm
import numpy as np
from PIL import Image
from datasets import VITONDataset
from models.sdafnet import SDAFNet_Tryon
from torchvision import transforms, utils
from utils import lpips
from utils.utils import AverageMeter
from torchvision import transforms, utils
from torch.utils import data
from torchvision.utils import save_image
from adaptive_pruning_daflow import run
cudnn.benchmark = True



def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str,default="TEST_PAIR")
    parser.add_argument('-b', '--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--load_height', type=int, default=256)
    parser.add_argument('--load_width', type=int, default = 192)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--add_compare', action='store_true')
    # parser.add_argument('--dataset_dir', type=str, default='data')
    parser.add_argument('--dataset_imgpath', type=str, default='data/VITON/VITON_test')
    parser.add_argument('--dataset_list', type=str, default=r'data/VITON/test_pairs2.txt')
    parser.add_argument('--checkpoint_dir', type=str, default=r'checkpoint/model.pt')
    parser.add_argument('--ad_checkpoint_dir', type=str, default=r'results/ad_b4l2.4e-05ts0.8/ad_b4l2.4e-05ts0.8iter487001opm.pt')
    parser.add_argument('--save_dir', type=str, default='./results/')
    parser.add_argument('--multi_flows', type=int, default=6)
    parser.add_argument('--display_freq', type=int, default=100)
    parser.add_argument('--output_dir',type=str, default='checkpoint')
    parser.add_argument('--threshold', type=float, default=0.8, required=False,
                        help='Pruning threshold for the adaptive part')
    opt = parser.parse_args()
    return opt

def test(opt, net,save_dir='./results/'):
    test_dataset = VITONDataset(opt)
    test_loader = data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle,num_workers=opt.workers)
    with torch.no_grad():
        for i, inputs in enumerate(tqdm.tqdm(test_loader)):
            img_names = inputs['img_name']
            cloth_names = inputs['c_name']['paired']
            img = inputs['img'].cuda()
            img_agnostic = inputs['img_agnostic'].cuda() #Masked model image
            pose = inputs['pose'].cuda()
            cloth_img = inputs['cloth']['paired'].cuda()
            img =  F.interpolate(img, size=(256, 192), mode='bilinear')
            cloth_img = F.interpolate(cloth_img, size=(256, 192), mode='bilinear')
            img_agnostic = F.interpolate(img_agnostic, size=(256, 192), mode='bilinear')
            pose = F.interpolate(pose, size=(256, 192), mode='bilinear')
            ref_input = torch.cat((pose, img_agnostic), dim=1)
            tryon_result = net(ref_input, cloth_img, img_agnostic).detach()
            for j in range(tryon_result.shape[0]):
                save_image(tryon_result[j:j+1], os.path.join(opt.save_dir, opt.name, "vis_viton_out", img_names[j]), nrow=1, normalize=True, range=(-1,1))
                print(f'save_dir:{os.path.join(save_dir, opt.name, "vis_viton_out", img_names[j])}')
            if opt.add_compare:
                tryon_result = torch.cat([img_agnostic, cloth_img, tryon_result],2)
                save_image(tryon_result, os.path.join(opt.save_dir, opt.name, "vis_viton_out", img_names[0]), nrow=10, normalize=True, range=(-1,1))


def main():
    opt = get_opt()
    print(opt)
    if not os.path.exists(os.path.join(opt.save_dir, opt.name, "vis_viton_out")):
        os.makedirs(os.path.join(opt.save_dir, opt.name,"vis_viton_out"))
    ratio = 0
    ori_cfg = [64, 128, 256, 256, 256]
    cfg = [int(x * (1 - ratio)) for x in ori_cfg]

    sdafnet = SDAFNet_Tryon(cfg=cfg).cuda()


    # sdafnet = sdafnet.cuda()
    # sdafnet.load_state_dict(torch.load(opt.checkpoint_dir),strict=False)
    sdafnet.load_state_dict(torch.load(opt.checkpoint_dir),strict=False)
    # sdafnet.load_state_dict(torch.load(opt.checkpoint_dir)['state_dict'],strict=False)

    # sdafnet.eval()
    prun_net = run(sdafnet, opt.output_dir,threshold=opt.threshold)
    prun_net.load_state_dict(torch.load(opt.ad_checkpoint_dir)['state_dict'],strict=False)
    torch.save(prun_net.state_dict(),f'checkpoint/ad_pruned_model_ts{opt.threshold}_no_opm.pt')
    test(opt, prun_net)

if __name__ == '__main__':
    main()
