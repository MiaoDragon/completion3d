import torch
from torch.autograd import Variable
import torch.nn as nn
from common import PointNetfeat
from dist_chamfer import chamferDist as chamfer
import numpy as np

class PointNetFCAE(nn.Module):
    """ PointNet Encoder, MLP Decoder"""
    def __init__(self, code_ntfs, num_points=2048, output_channels=3):
        super(PointNetFCAE, self).__init__()
        self.code_ntfs = code_ntfs
        self.num_points = num_points
        self.output_channels = output_channels
        self.encoder = PointNetfeat(code_ntfs, num_points, global_feat=True, trans=False)
        self.decoder = nn.Sequential(
            nn.Linear(self.code_ntfs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(256, output_channels*num_points)
        )

    def forward(self, x):
        code = self.encoder(x)
        x = self.decoder(code)
        x = x.view(-1, self.output_channels, self.num_points)
        #x = x.transpose(2, 1).contiguous()

        return x
