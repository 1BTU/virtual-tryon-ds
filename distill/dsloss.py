import torch.nn as nn
import torch
kd_loss_fuction = nn.MSELoss()
criterionL2 = nn.MSELoss('sum')
criterionL1 = nn.L1Loss('sum')
def KDLoss(teacher_fpn,student_fpn,temperature=1):
    soft_loss = nn.functional.kl_div(torch.log_softmax(student_fpn/temperature,dim=0),
                                     torch.softmax(teacher_fpn/temperature,dim=0),reduction='mean')
    # soft_loss = kd_loss_fuction(teacher_fpn,student_fpn)
    return soft_loss
def PFKDLoss(teacher_fpn,student_fpn):
    return criterionL2(teacher_fpn,student_fpn)
