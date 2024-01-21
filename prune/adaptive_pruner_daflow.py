import os

import torch
from torch import nn
import numpy as np
from models.refactor_sdanet import prun_SDAFNet_Tryon

def get_prun_ifo(model,new_model):
    cfg = []
    cfg_mask = []

    flag = 0
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if isinstance(m0,nn.Conv2d):
            out_channels = m0.weight.data.shape[0]
            weight_copy = m0.weight.data.abs().clone().cpu().numpy()
            L1_norm = np.sum(weight_copy,axis=(1,2,3))
            arg_max = np.argsort(L1_norm)
            nums_of_keep_channels = m1.weight.data.shape[0]
            cfg.append(nums_of_keep_channels)
            arg_max_rev = arg_max[::-1][:nums_of_keep_channels]
            mask = torch.zeros(out_channels)
            mask[arg_max_rev.tolist()] = 1
            cfg_mask.append(mask)
    return cfg,cfg_mask

def get_prun_FPN_ifo(model,new_model):
    cfg = []
    cfg_mask = []
    layer_id = 0
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if isinstance(m0,nn.Conv2d):
            in_channels = m0.weight.data.shape[1]
            weight_copy = m0.weight.data.abs().clone().cpu().numpy()
            L1_norm = np.sum(weight_copy,axis=(0,2,3))
            arg_max = np.argsort(L1_norm)
            nums_of_keep_channels = m1.weight.data.shape[1]
            cfg.append(nums_of_keep_channels)
            arg_max_rev = arg_max[::-1][:nums_of_keep_channels]
            mask = torch.zeros(in_channels)
            mask[arg_max_rev.tolist()] = 1
            cfg_mask.append(mask)
            layer_id+=1
            if layer_id == 5 :break
    return cfg,cfg_mask

def update_pruned_features_ifo(model, new_model, cfg, cfg_mask,cate):
    layer_id_in_cfg = 0
    if cate=='s':
        start_mask = torch.ones(3)
    else:
        start_mask = torch.ones(6)
    end_mask = cfg_mask[layer_id_in_cfg]
    batch_index = -1
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print("con2d input shape:{:d},output shape:{:d}".format(idx0.size, idx1.size))

            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            if m1.bias:
                m1.bias.data = m0.bias.data[idx1.tolist()]

            layer_id_in_cfg += 1
            if len(cfg) == layer_id_in_cfg:break
            start_mask = end_mask
            end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0,nn.BatchNorm2d):
            if batch_index == -1:
                m1.load_state_dict(m0.state_dict())
            else:
                mask=cfg_mask[batch_index]
                idx = np.squeeze(np.argwhere(np.asarray(mask.cpu())))
                m1.weight.data = m0.weight.data[idx].clone()
                m1.running_var.data = m0.running_var.data[idx].clone()
                m1.running_mean.data = m0.running_mean.data[idx].clone()
                m1.bias.data = m0.bias.data[idx].clone()
            batch_index += 1

def update_pruned_source_FPN_ifo(model,new_model,cfg,cfg_mask):
    layer_id_in_cfg = 0
    flag = 0
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if flag:
            m1.load_state_dict(m0.state_dict())
        elif isinstance(m0, nn.Conv2d):
            stark_mask = cfg_mask[layer_id_in_cfg]
            idx0 = np.squeeze(np.argwhere(np.asarray(stark_mask.cpu().numpy())))
            print('reference_FPN con2d input shape :{:d}'.format(idx0.size))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            m1.weight.data = w1.clone()
            layer_id_in_cfg += 1
            if layer_id_in_cfg == len(cfg):
                flag = 1
                continue
def update_pruned_reference_FPN_ifo(model,new_model,cfg,cfg_mask):
    layer_id_in_cfg = 0
    flag = 0
    for [m0,m1] in zip(model.modules(),new_model.modules()):
        if flag:
            m1.load_state_dict(m0.state_dict())
        elif isinstance(m0,nn.Conv2d):
            stark_mask = cfg_mask[layer_id_in_cfg]
            idx0 = np.squeeze(np.argwhere(np.asarray(stark_mask.cpu().numpy())))
            print('reference_FPN con2d input shape :{:d}'.format(idx0.size))
            w1 = m0.weight.data[:,idx0.tolist(),:,:].clone()
            m1.weight.data = w1.clone()
            m1.bias.data = m0.bias.data
            layer_id_in_cfg += 1
            if layer_id_in_cfg == len(cfg):
                flag = 1
                continue


def update_pruned_danet(model, new_model,cfg, cfg_mask):
    layer_id_in_cfg = 0

    start_mask = torch.ones(512)
    end_mask = cfg_mask[layer_id_in_cfg]
    print('len_mask',len(cfg_mask))
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if isinstance(m0, nn.Conv2d):
            if layer_id_in_cfg%4 == 0:
                start_mask = torch.ones(512)
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print("con2d input shape:{:d},output shape:{:d}".format(idx0.size, idx1.size))

            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()

            m1.bias.data = m0.bias.data[idx1.tolist()]
            layer_id_in_cfg+=1
            # print('layid:',layer_id_in_cfg)

            if 60 == layer_id_in_cfg: break

            start_mask = end_mask
            end_mask = cfg_mask[layer_id_in_cfg]

def prunner(model,new_model,output_dir,threshold):

    sf_conv_cfg,sf_conv_cfg_mask = get_prun_ifo(model.source_features,new_model.source_features)
    update_pruned_features_ifo(model.source_features, new_model.source_features, sf_conv_cfg, sf_conv_cfg_mask,'s')
    #prun reference_features
    rf_conv_cfg,rf_conv_cfg_mask = get_prun_ifo(model.reference_features,new_model.reference_features)
    update_pruned_features_ifo(model.reference_features,new_model.reference_features,rf_conv_cfg,rf_conv_cfg_mask,'f')
    # update_pruned_ifo(model.reference_features,new_model.reference_features)

    # #prun reference_FPN
    reference_FPN_cfg,reference_FPN_cfg_mask = get_prun_FPN_ifo(model.reference_FPN,new_model.reference_FPN)
    update_pruned_reference_FPN_ifo(model.reference_FPN, new_model.reference_FPN, reference_FPN_cfg, reference_FPN_cfg_mask)
    # update_pruned_ifo(model.reference_FPN,new_model.reference_FPN)
    #
    # #prun sourse_FPN
    source_FPN_cfg, source_FPN_cfg_mask = get_prun_FPN_ifo(model.source_FPN,new_model.source_FPN)
    update_pruned_source_FPN_ifo(model.source_FPN, new_model.source_FPN, source_FPN_cfg,source_FPN_cfg_mask)
    # update_pruned_ifo(model.source_FPN,new_model.source_FPN)

    # # dafnet_cfg,dafnet_cfg_mask = get_prun_ifo(model.reference_features,ratio)
    # dafnet_cfg,dafnet_cfg_mask = get_prun_ifo(model.dafnet,new_model.dafnet)
    # update_pruned_danet(model.dafnet, new_model.dafnet,dafnet_cfg,dafnet_cfg_mask)



    save_path = os.path.join(output_dir,f"ad_pruned_model_ts={threshold}.pt")
    torch.save(new_model.state_dict(),save_path)
    print(f'\nsave pruned model to:{save_path}\n')

    return new_model

