import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F



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
#ghost
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1):
        super(GhostBottleneck, self).__init__()
        self.in_chs = in_chs
        self.mid_chs = mid_chs
        self.out_chs = out_chs
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        # if self.stride > 1:
        self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                 padding=(dw_kernel_size - 1) // 2,
                                 groups=mid_chs, bias=False)
        self.bn_dw = nn.BatchNorm2d(mid_chs)



        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        # if self.stride > 1:
        x = self.conv_dw(x)
        x = self.bn_dw(x)


        # 2nd ghost bottleneck
        x = self.ghost2(x)
        if self.in_chs == self.out_chs:
            x += self.shortcut(residual)
        return x

# backbone



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

class DownSample_ghost(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample_ghost, self).__init__()
        init_channels = math.ceil(out_channels / 2)
        new_channels = out_channels - init_channels
        # Point-wise expansion
        self.primary = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, init_channels, kernel_size=1, padding=0, stride=2,bias=False),
        )
        self.cheap_operation = nn.Sequential(
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_channels, new_channels, kernel_size=3, padding=1, stride=1, bias=False, groups=init_channels),
        )
        # Squeeze-and-excitation


    def forward(self, x):
        x1 = self.primary(x)
        x2 = self.cheap_operation(x1)
        x_add = torch.cat([x1, x2], dim=1)
        return x_add

class FeatureEncoder(nn.Module):
    def __init__(self, in_channels, chns=[64,128,256,256,256]):
        # in_channels = 3 for images, and is larger (e.g., 17+1+1) for agnositc representation
        super(FeatureEncoder, self).__init__()

        self.encoders = []
        for i, out_chns in enumerate(chns):
            if i == 0:
                encoder = nn.Sequential(DownSample(in_channels, out_chns),
                                        GhostBottleneck(out_chns,out_chns,out_chns))
            else:
                encoder = nn.Sequential(GhostBottleneck(chns[i-1], out_chns,out_chns,stride=2),
                                         GhostBottleneck(out_chns,out_chns,out_chns))

            self.encoders.append(encoder)

        self.encoders = nn.ModuleList(self.encoders)


    def forward(self, x):
        encoder_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
        return encoder_features

class RefinePyramid(nn.Module):
    def __init__(self, chns=[64,128,256,256,256], fpn_dim=256):
        super(RefinePyramid, self).__init__()
        self.chns = chns

        # adaptive
        self.adaptive = []
        for in_chns in list(reversed(chns)):
            adaptive_layer = nn.Conv2d(in_chns, fpn_dim, kernel_size=1)
            self.adaptive.append(adaptive_layer)
        self.adaptive = nn.ModuleList(self.adaptive)
        # output conv
        self.smooth = []
        for i in range(len(chns)):
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
    def __init__(self, in_channels, out_channels, kernel_size=3, num_filters=[32]):
        super(MFEBlock, self).__init__()
        layers = []
        layers.append(GhostBottleneck(in_channels,32,32))
        layers.append(torch.nn.LeakyReLU(inplace=False, negative_slope=0.1))
        layers.append(GhostBottleneck(32,out_channels,out_channels))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class DAFlowNet(nn.Module):
    def __init__(self, num_pyramid, fpn_dim=256,head_nums = 1):
        super(DAFlowNet, self).__init__()
        self.Self_MFEs = []

        self.Cross_MFEs = []
        self.Refine_MFEs = []
        self.k = head_nums
        self.out_ch = fpn_dim
        for i in range(num_pyramid):
            # self-MFE for model img 2k:flow 1k:att_map
            Self_MFE_layer = MFEBlock(in_channels=2*fpn_dim, out_channels=self.k*3,kernel_size=7)
            # cross-MFE for cloth img
            Cross_MFE_layer = MFEBlock(in_channels=2*fpn_dim, out_channels=self.k*3)
            # refine-MFE for cloth and model imgs
            Refine_MFE_layer = MFEBlock(in_channels=2*fpn_dim, out_channels=self.k*6)
            self.Self_MFEs.append(Self_MFE_layer)
            self.Cross_MFEs.append(Cross_MFE_layer)
            self.Refine_MFEs.append(Refine_MFE_layer)

        self.Self_MFEs = nn.ModuleList(self.Self_MFEs)
        self.Cross_MFEs = nn.ModuleList(self.Cross_MFEs)
        self.Refine_MFEs = nn.ModuleList(self.Refine_MFEs)

        self.lights_decoder = torch.nn.Sequential(torch.nn.Conv2d(64, out_channels=32, kernel_size=1, stride=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1,padding=1))
        self.lights_encoder = torch.nn.Sequential(torch.nn.Conv2d(3, out_channels=32, kernel_size=3, stride=1,padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1))


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
            input_feat =  torch.cat([att_source_feat,feat_ref],1)
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
            input_feat = torch.cat([att_source_feat,att_reference_feat],1)
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



class NPDNet(nn.Module):
    def __init__(self, person_img = 3, clothes_img = 3, head_nums = 6, cfg=None):
        super(NPDNet, self).__init__()

        num_filters = [64,128,256,256,256]

        # num_filters = [32,64,128,128,128]
        # num_filters = [32,64,64,64,64]
        self.source_features = FeatureEncoder(clothes_img, num_filters)
        self.reference_features = FeatureEncoder(person_img, num_filters)
        self.source_FPN = RefinePyramid(num_filters)
        self.reference_FPN = RefinePyramid(num_filters)
        self.dafnet = DAFlowNet(len(num_filters),head_nums = head_nums)

    def forward(self, clothes_img, person_img, use_light_en_de=True, return_all=False, warp_feature=True):
        reference_feats = self.reference_FPN(self.reference_features(person_img))
        source_feats = self.source_FPN(self.source_features(clothes_img))
        result = self.dafnet(clothes_img, person_img, source_feats, reference_feats, use_light_en_de=use_light_en_de, return_all=return_all, warp_feature=warp_feature)
        return result

from torchviz import make_dot
import netron
if __name__ == '__main__':

    # path = 'F:\学习\服创\pruning model\DAFlow\model.pt'
    # netron.start(path)
    net = NPDNet().cuda()
    print(net)
    # net.load_state_dict(torch.load(path))
    # pruned_model = pruner(net, op.output_dir, args.ratio)
    # for k,v in net.state_dict().items():
    #     print(k,v.shape)
    # path_new = r'/home/xxfs/document/ai/fuchuang/DAFlow/checkpoint/dis'
    # torch.save(net.state_dict(), os.path.join(path_new, 'nf_dis0.25.pt'))
    from nni.compression.pytorch.utils import count_flops_params

    with torch.no_grad():
        source_image = torch.ones(8, 3, 128, 128).cuda()
        ref_image = torch.ones(8, 3, 128, 128).cuda()
        # out = net(source_image,ref_image)
        # g = make_dot(out)
        # g.view()
        flops, params, _ = count_flops_params(net, (source_image, ref_image), verbose=False)
        print(f"\nPruned Model after Weight Replacing:\nFLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M")
        # flops, params, _ = count_flops_params(net,(source_image, ref_image), verbose=False)
        # print(f"\nPruned Model after Weight Replacing:\nFLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M")
    # print(out.shape)Model after Weight Replacing:\nFLOPs {flops / 1e6:.2f}M, Params {params / 1e6:.2f}M")
    # print(out.shape)