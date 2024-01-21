import argparse
import os

import matplotlib.pyplot as plt

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
cudnn.benchmark = True
from models.sdafnet import ResBlock,MFEBlock,DAFlowNet,DownSample
# from prune.pruner_daflow import prunner
# from prune.adaptive_pruner_daflow import prunner
from prune.blocks import ResBlock_ifo,DownSample_ifo,MFEBlock_ifo
from models.refactor_sdanet import prun_SDAFNet_Tryon
from prune.adaptive_pruner_daflow import prunner
from nni.compression.pytorch.utils import count_flops_params
from puned_train_SDAFNet_viton import train
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def compute_weights(m, threshold):
    """
    Adaptive pruning is used to calculate the channels that meet the
    requirements according to the threshold
    """
    assert threshold != 0, "threshold must > 0"
    # return m.shape[0]
    weight = m.abs().cpu().numpy()
    L1_norm = np.sum(weight, axis=(1, 2, 3))
    x = sorted(L1_norm, reverse=True)
    normalized = (x - np.min(x)) / (np.max(x) - np.min(x))

    idx = 0
    flag = True
    for filter_idx in range(len(normalized)):
        if normalized[filter_idx] < threshold:
            idx = filter_idx
            break
            # flag = False

    nums_keep = None
    per = (idx + 1) / len(normalized)
    if int(per * len(normalized)) != 1:
        nums_keep = _make_divisible(per * len(normalized), 8)
    else:
        nums_keep = 1
    return nums_keep

def get_block(model):
    resblock = []
    mfeblock = []
    downsample = []
    # refinepyramid = []
    for i,(name,module) in enumerate(model.named_modules()):
        if isinstance(module,ResBlock):
            resblock.append(ResBlock_ifo(name,list(module.state_dict().values()),list(module.state_dict().keys())))
        elif isinstance(module,DownSample):
            downsample.append(DownSample_ifo(name,list(module.state_dict().values()),list(module.state_dict().keys())))
        elif isinstance(module,MFEBlock):
            mfeblock.append(MFEBlock_ifo(name,list(module.state_dict().values()),list(module.state_dict().keys())))
    blocks = downsample+resblock+mfeblock
    return blocks
def get_pruning_cfg(blocks,threshold):
    FeatureEncoder_cfg = []
    mfeblock_cfg = []
    refinepyramid_cfg = []
    for block in blocks:
        if isinstance(block,DownSample_ifo):
            FeatureEncoder_cfg.append(compute_weights(block.weight,threshold))
        elif isinstance(block,ResBlock_ifo):
            FeatureEncoder_cfg.append(compute_weights(block.out_weight1,threshold))
            FeatureEncoder_cfg.append(compute_weights(block.out_weight2,threshold))
        elif isinstance(block,MFEBlock_ifo):
            mfeblock_cfg.append(compute_weights(block.weight1,threshold))
            mfeblock_cfg.append(compute_weights(block.weight2,threshold))
            mfeblock_cfg.append(compute_weights(block.weight3,threshold))
    cfg = FeatureEncoder_cfg+mfeblock_cfg
    return cfg
def show_params(net,prun_net,ref_input,source_image,ref_image):
    flops_ori, params_ori, ori_dict = count_flops_params(net, (ref_input, source_image, ref_image), verbose=False)
    # print(_)
    print(f"pruned model :\nFLOPS:{flops_ori / 1e6:.2f}M,params {params_ori / 1e6:.2f}M")
    flops, params, prun_dict = count_flops_params(prun_net, (ref_input, source_image, ref_image), verbose=False)
    print(f"model :\nFLOPS:{flops / 1e6:.2f}M,params {params / 1e6:.2f}M")
    # 获取数据
    x = np.linspace(1, len(prun_dict), len(prun_dict))
    y1 = []
    y2 = []
    names = []
    for i in range(len(prun_dict)):
        # x.append(dic['flops'])
        y1.append(ori_dict[i]['params'])
        y2.append(prun_dict[i]['params'])
        names.append(prun_dict[i]['name'].split('.')[0])

    # 绘制散点图
    plt.plot(x, [y1[i] - y2[i] for i in range(len(y1))], c='r', label='ori_params')
    # plt.scatter(x, y2,c='g',label='prun_params')

    # 添加标签
    # for i, name in enumerate(names):
    #     plt.annotate(i, (x[i], y1[i]-y2[i]))
    # plt.annotate(i, (x[i], y2[i]))

    # 添加标题和轴标签
    plt.title('Params error')
    plt.xlabel('xi')
    plt.ylabel('Params')
    # 剪枝效率
    print(f'prun_flops_rate:{flops/flops_ori}%，prun_params_rate:{params/params_ori}%')
    # 显示图形
    plt.show()


def run(model,save_dir,threshold):
    blocks = get_block(model)
    cfg = get_pruning_cfg(blocks,threshold)
    #sourse_feature cfg
    scfg = cfg[:5]+cfg[10:30]
    scfg_cfg = []
    for i in range(5):
        scfg_cfg += scfg[i:i+1]+scfg[i*4+5:(i+1)*4+5]
    #reference_feature cfg
    rcfg = cfg[5:10]+cfg[30:50]
    rcfg_cfg = []
    for i in range(5):
        rcfg_cfg += rcfg[i:i + 1] + rcfg[i * 4 + 5:(i + 1) * 4 + 5]
    danet_cfg = cfg[50:]
    # net = SDAFNet_Tryon(ref_in_channel=6)
    scfg_FPN = [scfg_cfg[i] for i in range(4, len(scfg_cfg), 5)]
    rcfg_FPN = [rcfg_cfg[i] for i in range(4, len(rcfg_cfg), 5)]
    prun_net = prun_SDAFNet_Tryon(scfg_cfg=scfg_cfg,
                                  rcfg_cfg=rcfg_cfg,
                                  danet_cfg=danet_cfg,
                                  scfg_FPN=scfg_FPN,
                                  rcfg_FPN=rcfg_FPN).cuda()
    #更新信息
    prunner(model,prun_net,save_dir,threshold)

    #测试prun效果
    # test(opt, prun_net)

    return prun_net



    #测试网路是否构建成功
    # test(opt, prun_net)
    # prunner(net,prun_net,opt.output_dir)
    # with torch.no_grad():
    #     ref_input = torch.ones(8, 6, 128, 128).cuda()
    #     source_image = torch.ones(8, 3, 128, 128).cuda()
    #     ref_image = torch.ones(8, 3, 128, 128).cuda()
    #     out = prun_net(ref_input,source_image,ref_image)
        # show_params(model,prun_net,ref_input,source_image,ref_image)


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str,default="TEST_PAIR")
    parser.add_argument('-b', '--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--ratio', type=int, default=0)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--load_height', type=int, default=256)
    parser.add_argument('--load_width', type=int, default = 192)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--add_compare', action='store_true')
    parser.add_argument('--dataset_imgpath', type=str, default=r'data/VITON/VITON_test')
    parser.add_argument('--dataset_list', type=str, default=r'data/VITON/test_pairs2.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/model.pt')
    parser.add_argument('--save_dir', type=str, default='./results/')
    parser.add_argument('--multi_flows', type=int, default=6)
    parser.add_argument('--display_freq', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=0.9, required=False,
                        help='Pruning threshold for the adaptive part')

    parser.add_argument('--output_dir',type=str, default='checkpoint')
    opt = parser.parse_args()
    return opt

def test(opt, net):
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
            if opt.add_compare:
                tryon_result = torch.cat([img_agnostic, cloth_img, tryon_result],2)
                save_image(tryon_result, os.path.join(opt.save_dir, opt.name, "vis_viton_out", img_names[0]), nrow=10, normalize=True, range=(-1,1))


def main():
    opt = get_opt()
    print(opt)
    if not os.path.exists(os.path.join(opt.save_dir, opt.name, "vis_viton_out")):
        os.makedirs(os.path.join(opt.save_dir, opt.name,"vis_viton_out"))
    sdafnet = SDAFNet_Tryon(ref_in_channel=opt.multi_flows).cuda()
    sdafnet = sdafnet.cuda()
    sdafnet.load_state_dict(torch.load(opt.checkpoint_dir))
    prun_net = run(sdafnet,opt.output_dir,threshold=opt.threshold)
    # pruned_model = prunner(sdafnet,opt.output_dir,opt.ratio).cuda()
    #

    # test(opt, sdafnet)
if __name__ == '__main__':
    main()
