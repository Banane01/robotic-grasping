from inference.models.grasp_model import GraspModel, Fire

import torch
import torch.nn as nn

class SqueezeNet(GraspModel):
    def __init__(self, input_channels=4, output_channels=1, channel_size=64, dropout=False, prob=0.0) -> None:
        super(SqueezeNet, self).__init__()

        self.squeeze = nn.Sequential(
            nn.Conv2d(input_channels, channel_size, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(32, 16, 32, 32),
            Fire(64, 16, 32, 32),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 32, 64, 64),
            Fire(128, 32, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 48, 96, 96),
            Fire(192, 48, 96, 96),
            Fire(192, 64, 128, 128),
            Fire(256, 64, 128, 128),
            nn.AdaptiveAvgPool2d((224, 224)),
        )
        self.pos_output = nn.Conv2d(256, 1, kernel_size=1)
        self.cos_output = nn.Conv2d(256, 1, kernel_size=1)
        self.sin_output = nn.Conv2d(256, 1, kernel_size=1)
        self.width_output = nn.Conv2d(256, 1, kernel_size=1)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.pos_output or self.cos_output or self.sin_output or self.width_output:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.kaiming_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze(x)

        # Output
        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)
            
        return pos_output, cos_output, sin_output , width_output

