import torch
device = "cpu"
model = torch.load("/mnt/bb/yc/virtual-tryon-ds/checkpoint/pf_b_2l2e-05m_35/pf_b_2l2e-05m_35_data.pt")
source_image = torch.ones(8, 3, 128, 128).to(device)
ref_image = torch.ones(8, 3, 128, 128).to(device)
# model(source_image,ref_image)
print(model)
# print(1)