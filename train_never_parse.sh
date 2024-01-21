#!/bin/bash

name="VITON"
batch_size=2
lr=0.00002
ratio=0.25
workers=4
load_height=256
load_width=192
resume=false
iter=30001
shuffle=false
mode="train"
# dataset_dir="./data"
dataset_imgpath="data/VITON/VITON_train"
dataset_list="data/VITON/train_unpairs.txt"
checkpoint_dir="checkpoint/0.8init_pruned_model.pt"
save_dir="./results/"
display_freq=200
save_freq=1000
multi_flows=6
alpha=0.3
temperture=1

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --name)
      name=$2
      shift
      shift
      ;;
    -b|--batch_size)
      batch_size=$2
      shift
      shift
      ;;
    --lr)
      lr=$2
      shift
      shift
      ;;
    --ratio)
      ratio=$2
      shift
      shift
      ;;
    -j|--workers)
      workers=$2
      shift
      shift
      ;;
    --load_height)
      load_height=$2
      shift
      shift
      ;;
    --load_width)
      load_width=$2
      shift
      shift
      ;;
    --resume)
      resume=$2
      shift
      shift
      ;;
    --iter)
      iter=$2
      shift
      shift
      ;;
    --shuffle)
      shuffle=$2
      shift
      shift
      ;;
    --mode)
      mode=$2
      shift
      shift
      ;;
    --dataset_imgpath)
      dataset_imgpath=$2
      shift
      shift
      ;;
    --dataset_list)
      dataset_list=$2
      shift
      shift
      ;;
    --checkpoint_dir)
      checkpoint_dir=$2
      shift
      shift
      ;;
    --save_dir)
      save_dir=$2
      shift
      shift
      ;;
    --display_freq)
      display_freq=$2
      shift
      shift
      ;;
    --save_freq)
      save_freq=$2
      shift
      shift
      ;;
    --multi_flows)
      multi_flows=$2
      shift
      shift
      ;;
    --alpha)
      alpha=$2
      shift
      shift
      ;;
    --temperture)
      temperture=$2
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      shift
      ;;
  esac
done

# 执行程序或其他操作，使用上述参数
python never_parse_ds_train_SDANet_viton.py --name "$name" -b "$batch_size" --lr "$lr" --ratio $ratio \
  -j "$workers" --load_height "$load_height" --load_width "$load_width" \
  --resume "$resume" --iter "$iter" --shuffle "$shuffle" \
  --mode "$mode" --dataset_imgpath "$dataset_imgpath" --dataset_list "$dataset_list" \
  --checkpoint_dir "$checkpoint_dir" --save_dir "$save_dir" \
  --display_freq "$display_freq" --save_freq "$save_freq" \
  --multi_flows "$multi_flows" --alpha "$alpha" --temperture "$temperture"