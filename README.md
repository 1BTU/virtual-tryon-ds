# 基于知识蒸馏的轻量化虚拟试衣研究与实现



![输入图片说明](https://foruda.gitee.com/images/1726367703561225980/37e2049f_10916458.png "屏幕截图")

现有大部分虚拟试衣技术通常依赖人体解析，并且模型参数量较大，限制了其在实际应用中的效率和便利性。针对这些问题，本文提出了一种无解析器的轻量化虚拟试衣方法，为了降低对人体解析的依赖，采用"教师-导师-学生"知识蒸馏框架，其中SD-AFN充当教师网络。通过在线蒸馏技术替换传统的静态蒸馏方法，实现了对轻量化学生网络的同步训练，在保证试衣效果的同时减少了计算资源的消耗。

## 准备工作

- 数据集:[Train_data](https://pan.baidu.com/s/12To-zP4A0H7pGNOFRlDyaA?pwd=nbew ) [Test_data](https://pan.baidu.com/s/13Bom6LHcVoBAo3b5msBUKA?pwd=0pox), put the folders into `data/VITON`
- 权重: [ckpt_viton.pt](https://pan.baidu.com/s/1x99f-sR62gGNGu7EHpcCzg?pwd=je5k),由于实验需要调整模型的大小，不同的模型需要不同的代码进行调整网络层，因此我将每个模型所需要的代码放入相关的checkpoint文件夹当中，在使用模型时需要将对应文件夹中的**never_parse_ds_studentmodel.py**文件拷贝到**models / never_parse_ds_studentmodel.py**中。
- python 3.8.0
- pytorch1.9
- torchvision 0.10.0

##  推理

在推理时，推荐使用**pf_b_2l2e-05m_35_freeze.pt**或者**pf_b_2l2e-05m_35.pt**

```
python never_parse_ds_test_SDAFNet_viton.py
```

### 批量推理优化

推理入口支持 `--device auto|cpu|cuda`，默认自动选择设备，并在评估模式和
`torch.no_grad()` 下运行。批量任务不再逐张弹窗，仅加载人物与目标服装，
无需 `vis_pose`、`img_agnostic` 或配对服装图片；训练数据加载器保持原样。
原有图像缩放、归一化与模型输入顺序保持兼容，模型输入固定为 256×192。

```bash
python never_parse_ds_test_SDAFNet_viton.py \
  --device auto --batch_size 4 --workers 4 \
  --dataset_imgpath data/VITON/VITON_test \
  --dataset_list data/VITON/test_any.txt \
  --checkpoint_dir checkpoint/pf_b_2l2e-05m_35/pf_b_2l2e-05m_35_data.pt
```

数据根目录应包含 `test_img/<人物文件名>` 和 `clothes/<服装文件名>`，
配对列表每行是两个以空格分隔的文件名。CPU 或 Windows 调试时可使用
`--device cpu --workers 0`。增大 batch_size 会增加内存/显存占用；同一批次的
源图片应有一致宽高比。输出目录为 `results/TEST_PAIR/vis_viton_out`，
`--add_compare` 可保存“人物、服装、试衣结果”的横向对比图。

权重文件不会随仓库自动下载。此入口接收可信来源的完整 `nn.Module` 检查点，
不直接接收 `state_dict`；仍需按前文说明使用与权重匹配的模型定义。
此轮改动不包含重新训练，真实模型的速度、显存和图像指标需使用权重与数据集实测。

安装匹配的 PyTorch/torchvision 后，可运行不需要真实权重或 VITON 数据的回归测试：

```bash
python -m unittest discover -s tests -v
```

## Train

在训练时，需要下载[原始权重](https://pan.baidu.com/s/1x3WN_tB2yTdY4VcbeZ_IYQ?pwd=cpna )到**checkpoint/**路径中来指导学生网络训练。

```
python never_parse_ds_train_SDANet_viton.py -b 2 --lr 0.00002 --ratio 0.25 
```



## 结果

- VITON dataset

实验结果表明，该算法在无人体解析的条件下能够保持SDAFN模型精度，并将其参数量和计算量，分别降低了61.46%和75.92%。此外，在CPU上的推理速度提升了20%。

| **Model**      | **Flops** | **params** | **FID(↓)** | **SSIM(↑)** | **PSNR(↑)** | **IS(↑)** | **time** |
| -------------- | --------- | ---------- | ---------- | ----------- | ----------- | --------- | -------- |
| SD-AFN         | 24879.95M | 37.63M     | 7.729      | 0.839       | 23.487      | 2.755     | 1.51     |
| HybridVTON     | 10887.92M | 15.84M     | 10         | 0.825       | 22.324      | 2.715     | 1.221    |
| HybridVTON0.25 | 9588.73M  | 9.06M      | 10.221     | 0.823       | 22.217      | 2.724     | 1.193    |
| HybridVTON0.5  | 9710.63M  | 10.34M     | 9.558      | 0.842       | 22.881      | 2.738     | 1.204    |

---

## **感谢**

我们的代码参考了[DAFlow](https://github.com/OFA-Sys/DAFlow?tab=readme-ov-file)中的多级联网络和[PF-AFN](https://github.com/geyuying/PF-AFN)知识蒸馏技术。感谢他们的精彩工作。
## License
此代码的使用仅限于非商业研究和教育目的。
