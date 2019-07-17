import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
class PointNetfeat(nn.Module):
    def __init__(self, code_ntfs, num_points=2500, global_feat=True, trans=False):
        super(PointNetfeat, self).__init__()
        self.code_ntfs = code_ntfs
        #self.stn = STN3d(self, num_points=num_points)
        self.convs = [  torch.nn.Conv1d(3, 64, 1),
                        torch.nn.Conv1d(64, 128, 1),
                        torch.nn.Conv1d(128, 128, 1),
                        torch.nn.Conv1d(128, 256, 1),
                        torch.nn.Conv1d(256, self.code_ntfs, 1)]
        self.convs = nn.ModuleList(self.convs)
        self.bns = [
                        torch.nn.BatchNorm1d(64),
                        torch.nn.BatchNorm1d(128),
                        torch.nn.BatchNorm1d(128),
                        torch.nn.BatchNorm1d(256),
                        torch.nn.BatchNorm1d(self.code_ntfs)]
        self.bns = nn.ModuleList(self.bns)
        self.dropout = torch.nn.Dropout()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = self.bns[i](x)
            x = self.relu(x)
            x = self.dropout(x)
        x,_ = torch.max(x, 2)
        x = x.view(-1, self.code_ntfs)
        return x
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
