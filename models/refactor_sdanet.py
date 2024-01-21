import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from nni.compression.pytorch.utils import count_flops_params
from tensorboardX import SummaryWriter
def apply_offset(offset):
    sizes = list(offset.size()[2:])
    grid_list = torch.meshgrid([torch.arange(size, device=offset.device) for size in sizes])
    grid_list = reversed(grid_list)
    # apply offset
    grid_list = [grid.float().unsqueeze(0) + offset[:, dim, ...]
        for dim, grid in enumerate(grid_list)]
    # normalize
    grid_list = [grid / ((size - 1.0) / 2.0) - 1.0
        for grid, size in zip(grid_list, reversed(sizes))]

    return torch.stack(grid_list, dim=-1)




# backbone
class ResBlock(nn.Module):
    def __init__(self,in_channels1,out_channels1,out_channels2):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels1, out_channels1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels1, out_channels2, kernel_size=3, padding=1, bias=False)
            )

    def forward(self, x):
        # return self.block(x) + x #orignal
        return self.block(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            )

    def forward(self, x):
        return self.block(x)



class FeatureEncoder(nn.Module):
    def __init__(self, in_channels, chns=[64, 64,64,64,64,128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,256, 256, 256]):
        # in_channels = 3 for images, and is larger (e.g., 17+1+1) for agnositc representation
        super(FeatureEncoder, self).__init__()
        self.encoders = []
        iters = 0
        while iters<len(chns):
            sub_encoder = []
            out_chns = chns[iters]
            iters += 1
            sub_encoder.append(DownSample(in_channels, out_chns))
            in_channels1 = out_chns

            out_chns1 = chns[iters]
            iters += 1
            out_chns2 = chns[iters]
            iters += 1
            sub_encoder.append(ResBlock(in_channels1,out_chns1,out_chns2))
            in_channels3 = out_chns2
            out_chns3 = chns[iters]
            iters+=1
            out_chns4 = chns[iters]
            iters+=1
            sub_encoder.append(ResBlock(in_channels3,out_chns3,out_chns4))
            in_channels = out_chns4
            self.encoders.append(nn.Sequential(*sub_encoder))
        self.encoders = nn.ModuleList(self.encoders)


    def forward(self, x):
        encoder_features = []
        for encoder in self.encoders:
            # for e in encoder:
            x = encoder(x)
            encoder_features.append(x)
        return encoder_features

class RefinePyramid(nn.Module):
    def __init__(self, chns=None, fpn_dim=256):
        super(RefinePyramid, self).__init__()
        self.chns = chns

        # adaptive
        #in chanle
        if chns is None:
            chns = [256,256,256,128,64]
        self.adaptive = []
        for in_chns in chns:
            adaptive_layer = nn.Conv2d(in_chns, fpn_dim, kernel_size=1)
            self.adaptive.append(adaptive_layer)
        self.adaptive = nn.ModuleList(self.adaptive)
        # output conv
        cfg_smooth = [256,256,256,256,256]
        self.smooth = []
        for i in cfg_smooth:
            smooth_layer = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)
            self.smooth.append(smooth_layer)
        self.smooth = nn.ModuleList(self.smooth)

    def forward(self, x):
        conv_ftr_list = x

        feature_list = []
        last_feature = None
        for i, conv_ftr in enumerate(list(reversed(conv_ftr_list))):
            # adaptive
            feature = self.adaptive[i](conv_ftr)
            # fuse
            if last_feature is not None:
                feature = feature + F.interpolate(last_feature, scale_factor=2, mode='nearest')
            # smooth
            feature = self.smooth[i](feature)
            last_feature = feature
            feature_list.append(feature)

        return tuple(reversed(feature_list))
def DAWarp(feat, offsets, att_maps, sample_k, out_ch,detach = False):
    att_maps = torch.repeat_interleave(att_maps, out_ch, 1)
    B,C,H,W = feat.size()
    multi_feat = torch.repeat_interleave(feat, sample_k, 0)
    if detach:
       multi_warp_feat = F.grid_sample(multi_feat, offsets.detach().permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
    else:
       multi_warp_feat = F.grid_sample(multi_feat, offsets.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
    multi_att_warp_feat = multi_warp_feat.reshape(B,-1,H,W)*att_maps
    att_warp_feat = sum(torch.split(multi_att_warp_feat,out_ch,1))
    return att_warp_feat



class MFEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_filters=None):
        super(MFEBlock, self).__init__()
        layers = []
        if num_filters == None:
            num_filters = [128,64,32]
        for i in range(len(num_filters)):
            if i==0:
                layers.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=num_filters[i], kernel_size=3, stride=1, padding=1))
            else:
                layers.append(torch.nn.Conv2d(in_channels=num_filters[i-1], out_channels=num_filters[i], kernel_size=kernel_size, stride=1, padding=kernel_size//2))
            layers.append(torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
        layers.append(torch.nn.Conv2d(in_channels=num_filters[-1], out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class DAFlowNet(nn.Module):
    def __init__(self, num_pyramid, danet_cfg, fpn_dim=256, head_nums=1):
        super(DAFlowNet, self).__init__()
        self.Self_MFEs = []

        self.Cross_MFEs = []
        self.Refine_MFEs = []
        self.k = head_nums
        self.out_ch = fpn_dim
        # out_cfg = [3,3,3,3,3]
        Self_MFE_layer_cfg = [512,512,512,512,512]
        Cross_MFE_layer_cfg = [512,512,512,512,512]
        Refine_MFE_layer_cfg = [512,512,512,512,512]
        iters_index = 0
        for i in Self_MFE_layer_cfg:
            # self-MFE for model img 2k:flow 1k:att_map
            Self_MFE_layer = MFEBlock(in_channels=i, out_channels=self.k*3, kernel_size=7, num_filters=danet_cfg[iters_index:iters_index + 3])
            self.Self_MFEs.append(Self_MFE_layer)
            iters_index += 3

        for i in Cross_MFE_layer_cfg:
            # cross-MFE for cloth img
            Cross_MFE_layer = MFEBlock(in_channels=i, out_channels=self.k * 3, num_filters=danet_cfg[iters_index:iters_index + 3])
            self.Cross_MFEs.append(Cross_MFE_layer)
            iters_index += 3

        for i in Refine_MFE_layer_cfg:
            # refine-MFE for cloth and model imgs
            Refine_MFE_layer = MFEBlock(in_channels=i, out_channels=self.k * 6, num_filters=danet_cfg[iters_index:iters_index + 3])
            self.Refine_MFEs.append(Refine_MFE_layer)
            iters_index += 3

        self.Self_MFEs = nn.ModuleList(self.Self_MFEs)
        self.Cross_MFEs = nn.ModuleList(self.Cross_MFEs)
        self.Refine_MFEs = nn.ModuleList(self.Refine_MFEs)

        cfg_light = [64,32,3,32,64]
        self.lights_decoder = torch.nn.Sequential(torch.nn.Conv2d(cfg_light[0], out_channels=cfg_light[1], kernel_size=1, stride=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=cfg_light[1], out_channels=cfg_light[2], kernel_size=3, stride=1,padding=1))
        self.lights_encoder = torch.nn.Sequential(torch.nn.Conv2d(cfg_light[2], out_channels=cfg_light[3], kernel_size=3, stride=1,padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=cfg_light[3], out_channels=cfg_light[4], kernel_size=1, stride=1))


    def forward(self,
                source_image,
                reference_image,
                source_feats,
                reference_feats,
                return_all=False,
                warp_feature=True,
                use_light_en_de=True):

        r"""
        Args:
            source_image: cloth rgb image for tryon
            reference_image: model rgb image for try on
            source_feats: cloth FPN features
            reference_feats: model and pose features
            return_all: bool return all intermediate try-on results in training phase
            warp_feature: use DAFlow for both features and images
            use_light_en_de: use shallow encoder and decoder to project the images from RGB to high dimensional space

        """

        #reference branch inputs model img using self-DAFlow
        last_multi_self_offsets = None
        #source branch inputs cloth img using cross-DAFlow
        last_multi_cross_offsets = None

        if return_all:
            results_all = []


        for i in range(len(source_feats)):

            feat_source = source_feats[len(source_feats) - 1 - i]
            feat_ref = reference_feats[len(reference_feats) - 1 - i]
            B,C,H,W = feat_source.size()

            ## Pre-DAWarp for Pyramid feature
            if last_multi_cross_offsets is not None and warp_feature:
                att_source_feat = DAWarp(feat_source, last_multi_cross_offsets, cross_att_maps, self.k, self.out_ch, detach=True)
                att_reference_feat = DAWarp(feat_ref, last_multi_self_offsets, self_att_maps, self.k, self.out_ch, detach=True)
            else:
                att_source_feat = feat_source
                att_reference_feat = feat_ref
            ## Cross-MFE
            input_feat = torch.cat([att_source_feat,feat_ref],1)
            offsets_att = self.Cross_MFEs[i](input_feat)
            cross_att_maps = F.softmax(offsets_att[:,self.k*2:,:,:],dim=1)
            offsets = apply_offset(offsets_att[:,:self.k*2,:,:].reshape(-1,2,H,W))
            if last_multi_cross_offsets is not None:
                offsets = F.grid_sample(last_multi_cross_offsets, offsets, mode='bilinear', padding_mode='border')
            else:
                offsets = offsets.permute(0, 3, 1, 2)
            last_multi_cross_offsets = offsets
            att_source_feat = DAWarp(feat_source, last_multi_cross_offsets, cross_att_maps, self.k, self.out_ch)

            ## Self-MFE
            input_feat = torch.cat([att_source_feat,att_reference_feat],1)
            offsets_att = self.Self_MFEs[i](input_feat)
            self_att_maps = F.softmax(offsets_att[:,self.k*2:,:,:],dim=1)
            offsets = apply_offset(offsets_att[:,:self.k*2,:,:].reshape(-1,2,H,W))
            if last_multi_self_offsets is not None:
                offsets = F.grid_sample(last_multi_self_offsets, offsets, mode='bilinear', padding_mode='border')
            else:
                offsets = offsets.permute(0, 3, 1, 2)
            last_multi_self_offsets = offsets
            att_reference_feat = DAWarp(feat_ref, last_multi_self_offsets, self_att_maps , self.k, self.out_ch)

            ##Refine-MFE
            input_feat =  torch.cat([att_source_feat,att_reference_feat],1)
            offsets_att = self.Refine_MFEs[i](input_feat)
            att_maps = F.softmax(offsets_att[:,self.k*4:,:,:],dim=1)
            cross_offsets = apply_offset(offsets_att[:,:self.k*2,:,:].reshape(-1,2,H,W))
            self_offsets = apply_offset(offsets_att[:,self.k*2:self.k*4,:,:].reshape(-1,2,H,W))
            last_multi_cross_offsets = F.grid_sample(last_multi_cross_offsets, cross_offsets, mode='bilinear', padding_mode='border')
            last_multi_self_offsets = F.grid_sample(last_multi_self_offsets, self_offsets, mode='bilinear', padding_mode='border')

            ##Upsampling
            last_multi_cross_offsets = F.interpolate(last_multi_cross_offsets, scale_factor=2, mode='bilinear')
            last_multi_self_offsets = F.interpolate(last_multi_self_offsets, scale_factor=2, mode='bilinear')
            self_att_maps = F.interpolate(att_maps[:,:self.k,:,:], scale_factor=2, mode='bilinear')
            cross_att_maps = F.interpolate(att_maps[:,self.k:,:,:], scale_factor=2, mode='bilinear')

            ## Post-DAWarp for source and reference images
            if return_all:
                cur_source_image = F.interpolate(source_image, (H*2,W*2), mode='bilinear')
                cur_reference_image = F.interpolate(reference_image,(H*2,W*2), mode='bilinear')
                if use_light_en_de:
                    cur_source_image = self.lights_encoder(cur_source_image)
                    cur_reference_image = self.lights_encoder(cur_reference_image)
                    # the feat dim in light encoder is 64
                    warp_att_source_image = DAWarp(cur_source_image, last_multi_cross_offsets, cross_att_maps, self.k, 64)
                    warp_att_reference_image = DAWarp(cur_reference_image, last_multi_self_offsets, self_att_maps, self.k, 64)
                    result_tryon = self.lights_decoder(warp_att_source_image+warp_att_reference_image)
                else:
                    warp_att_source_image = DAWarp(cur_source_image, last_multi_cross_offsets, cross_att_maps, self.k, 3)
                    warp_att_reference_image = DAWarp(cur_reference_image, last_multi_self_offsets, self_att_maps, self.k, 3)
                    result_tryon = warp_att_source_image+warp_att_reference_image
                results_all.append(result_tryon)

        last_multi_self_offsets = F.interpolate(last_multi_self_offsets, reference_image.size()[2:], mode='bilinear')
        last_multi_cross_offsets = F.interpolate(last_multi_cross_offsets, source_image.size()[2:], mode='bilinear')
        self_att_maps = F.interpolate(self_att_maps, reference_image.size()[2:], mode='bilinear')
        cross_att_maps = F.interpolate(cross_att_maps, source_image.size()[2:], mode='bilinear')
        if use_light_en_de:
            source_image = self.lights_encoder(source_image)
            reference_image = self.lights_encoder(reference_image)
            warp_att_source_image = DAWarp(source_image, last_multi_cross_offsets, cross_att_maps, self.k, 64)
            warp_att_reference_image = DAWarp(reference_image, last_multi_self_offsets, self_att_maps, self.k, 64)
            result_tryon = self.lights_decoder(warp_att_source_image+warp_att_reference_image)
        else:
            warp_att_source_image = DAWarp(source_image, last_multi_cross_offsets, cross_att_maps, self.k, 3)
            warp_att_reference_image = DAWarp(reference_image, last_multi_self_offsets, self_att_maps, self.k, 3)
            result_tryon = warp_att_source_image+warp_att_reference_image

        if return_all:
            return result_tryon, results_all
        return result_tryon


import os
class prun_SDAFNet_Tryon(nn.Module):
    def __init__(self,scfg_cfg=None,rcfg_cfg=None,danet_cfg = None,scfg_FPN=None,rcfg_FPN=None,ref_in_channel = 6,source_in_channel = 3,head_nums = 6):
        super(prun_SDAFNet_Tryon, self).__init__()
        if scfg_cfg is None:
            scfg_cfg = [64, 64,64,64,64,128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,256, 256, 256]
            rcfg_cfg = [64, 64,64,64,64,128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,256, 256, 256]
            scfg_FPN = [256, 256, 256, 128, 64]
            rcfg_FPN = [256, 256, 256, 128, 64]



        self.source_features = FeatureEncoder(source_in_channel,scfg_cfg)
        self.reference_features = FeatureEncoder(ref_in_channel, rcfg_cfg)
        self.source_FPN = RefinePyramid(reversed(scfg_FPN))
        self.reference_FPN = RefinePyramid(reversed(rcfg_FPN))
        self.dafnet = DAFlowNet(5,head_nums = head_nums,danet_cfg = danet_cfg)

    def forward(self, ref_input, source_image, ref_image,use_light_en_de=True,return_all=False,warp_feature=True):
        reference_feats = self.reference_FPN(self.reference_features(ref_input))
        source_feats = self.source_FPN(self.source_features(source_image))
        result = self.dafnet(source_image, ref_image, source_feats, reference_feats,use_light_en_de=use_light_en_de,return_all=return_all,warp_feature=warp_feature)
        return result

from torchviz import make_dot
import netron
if __name__ == '__main__':
    # path = 'F:\学习\服创\pruning model\DAFlow\model.pt'
    path_new = 'F:\学习\服创\pruning model\DAFlow'
    # netron.start(path)
    net = prun_SDAFNet_Tryon(ref_in_channel=6).cuda()
    ori_cfg = [32, 128, 256, 256, 256]
    # new_cfg = [int(x * (1 - 0.5)) for x in ori_cfg]
    # new_net = SDAFNet_Tryon(cfg=new_cfg).cuda()

    torch.save(net.state_dict(),os.path.join(path_new,'new_ckpt_viton.pt'))
    path = r'F:\学习\服创\pruning model\DAFlow\new_ckpt_viton.pt'
    net.load_state_dict(torch.load(path))
    # pruned_model = pruner(net, op.output_dir, args.ratio)
    # for k,v in net.state_dict().items():
    #     print(k,v.shape)
    # flops, params, _ = count_flops_params(new_net.dafnet, ())
    # print(f"Pruned Model:\nFLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M")

    with torch.no_grad():
        ref_input = torch.ones(8, 6, 128, 128).cuda()
        source_image = torch.ones(8, 3, 128, 128).cuda()
        ref_image = torch.ones(8, 3, 128, 128).cuda()
        out = net(ref_input,source_image,ref_image)
        # g = make_dot(out)
        # g.view()

    # print(out.shape)
