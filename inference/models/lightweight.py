import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from inference.models.grasp_model import GraspModel, MFF, MFA

class Lightweight(GraspModel):
    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super(Lightweight, self).__init__()
        self.conv1a = nn.Conv2d(3, channel_size, kernel_size=3, padding=1, stride=1)
        self.conv1a_bn = nn.BatchNorm2d(channel_size)
        self.conv2a = nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1, stride=2)
        self.conv2a_bn = nn.BatchNorm2d(channel_size)
        self.conv3a = nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1, stride=2)
        self.conv3a_bn = nn.BatchNorm2d(channel_size)
        self.conv4a = nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1, stride=2)
        self.conv4a_bn = nn.BatchNorm2d(channel_size)

        self.mff1 = MFF(channel_size, channel_size)
        self.mff2 = MFF(channel_size, channel_size)
        self.mff3 = MFF(channel_size, channel_size)
        self.mff4 = MFF(channel_size, channel_size)

        self.conv1b = nn.Conv2d(1, channel_size, kernel_size=3, padding=1, stride=1)
        self.conv1b_bn = nn.BatchNorm2d(channel_size)
        self.conv2b = nn.Conv2d(channel_size*2, channel_size, kernel_size=3, padding=1, stride=2)
        self.conv2b_bn = nn.BatchNorm2d(channel_size)
        self.conv3b = nn.Conv2d(channel_size*2, channel_size, kernel_size=3, padding=1, stride=2)
        self.conv3b_bn = nn.BatchNorm2d(channel_size)
        self.conv4b = nn.Conv2d(channel_size*2, channel_size, kernel_size=3, padding=1, stride=2)
        self.conv4b_bn = nn.BatchNorm2d(channel_size)

        self.mfa1 = MFA(channel_size*2, channel_size*2, 28)
        self.mfa2 = MFA(channel_size*2, channel_size*2, 56)
        self.mfa3 = MFA(channel_size*2, channel_size*2, 112)

        self.conv1_transpose = nn.ConvTranspose2d(channel_size*2, channel_size*2, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv1_transpose_bn = nn.BatchNorm2d(channel_size*2)
        self.conv2_transpose = nn.ConvTranspose2d(channel_size*2, channel_size*2, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv2_transpose_bn = nn.BatchNorm2d(channel_size*2)
        self.conv3_transpose = nn.ConvTranspose2d(channel_size*2, channel_size*2, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.conv3_transpose_bn = nn.BatchNorm2d(channel_size*2)
        self.conv4_transpose = nn.ConvTranspose2d(channel_size*2, channel_size*2, kernel_size=3, padding=1, stride=1)
        self.conv4_transpose_bn = nn.BatchNorm2d(channel_size*2)

        self.pos_output = nn.Conv2d(in_channels=channel_size*2, out_channels=1, kernel_size=1)
        self.cos_output = nn.Conv2d(in_channels=channel_size*2, out_channels=1, kernel_size=1)
        self.sin_output = nn.Conv2d(in_channels=channel_size*2, out_channels=1, kernel_size=1)
        self.width_output = nn.Conv2d(in_channels=channel_size*2, out_channels=1, kernel_size=1)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = x[:, :3, :, :] # Split the input tensor into RGB and Depth
        d = x[:, 3, :, :]
        d = d.unsqueeze(1)

        rgb = self.conv1a_bn(self.conv1a(rgb))
        rgb = F.relu(rgb)
        d = self.conv1b_bn(self.conv1b(d))
        d = F.relu(d)
        d = self.mff1(rgb, d)
        rgb = self.conv2a_bn(self.conv2a(rgb))
        rgb = F.relu(rgb)
        d = self.conv2b_bn(self.conv2b(d))
        d = F.relu(d)
        d = self.mff2(rgb, d)
        rgb = self.conv3a_bn(self.conv3a(rgb))
        rgb = F.relu(rgb)
        d_new = self.conv3b_bn(self.conv3b(d))
        d_new = F.relu(d_new)
        d_new = self.mff3(rgb, d_new)
        rgb = self.conv4a_bn(self.conv4a(rgb))
        rgb = F.relu(rgb)
        d_new1 = self.conv4b_bn(self.conv4b(d_new))
        d_new1 = F.relu(d_new1)
        d_1 = self.mff4(rgb, d_new1)
        d_new1 = torch.cat([rgb, d_new1], 1)
        d_new1 = d_new1 + d_1
        d_new1 = self.mfa1(d_new1)
        d_new1 = self.conv1_transpose_bn(self.conv1_transpose(d_new1))
        d_new1 = F.relu(d_new1)
        d_new1 = self.mfa2(d_new1 + d_new)
        d_new1 = self.conv2_transpose_bn(self.conv2_transpose(d_new1))
        d_new1 = F.relu(d_new1)
        d_new1 = self.mfa3(d_new1 + d)
        d_new1 = self.conv3_transpose_bn(self.conv3_transpose(d_new1))
        d_new1 = F.relu(d_new1)
        d_new1 = self.conv4_transpose_bn(self.conv4_transpose(d_new1))

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(d_new1))
            cos_output = self.cos_output(self.dropout_cos(d_new1))
            sin_output = self.sin_output(self.dropout_sin(d_new1))
            width_output = self.width_output(self.dropout_wid(d_new1))
        else:
            pos_output = self.pos_output(d_new1)
            cos_output = self.cos_output(d_new1)
            sin_output = self.sin_output(d_new1)
            width_output = self.width_output(d_new1)

        return pos_output, cos_output, sin_output, width_output
