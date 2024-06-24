import torch.nn as nn
import torch.nn.functional as F
import torch


class GraspModel(nn.Module):
    """
    An abstract model for grasp network in a common format.
    """

    def __init__(self):
        super(GraspModel, self).__init__()

    def forward(self, x_in):
        raise NotImplementedError()

    def compute_loss(self, xc, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

    def predict(self, xc):
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)
        return {
            'pos': pos_pred,
            'cos': cos_pred,
            'sin': sin_pred,
            'width': width_pred
        }


class ResidualBlock(nn.Module):
    """
    A residual block with dropout option
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in
    

class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )
    
class MFE(nn.Module):
    def __init__(self, in_channels, kernel_size = 1, kernel_size_dilated = 3):
        super(MFE, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size_dilated, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.bn3 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x1 = self.bn1(x_in)
        x2 = self.conv1(x_in)
        x2 = self.bn2(x2)
        x3 = self.conv3(x_in)
        x3 = self.bn3(x3)
        return (x1 + x2 + x3)

class SEModule(nn.Module):
    def __init__(self, channels, reduction=2):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

class MFF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, kernel_size_dilated=3):
        super(MFF, self).__init__()
        self.MFE1 = MFE(in_channels, kernel_size, kernel_size_dilated)
        self.MFE2 = MFE(in_channels, kernel_size, kernel_size_dilated)
        self.SE1 = SEModule(in_channels)
        self.SE2 = SEModule(in_channels)

    def forward(self, x_rgb: torch.Tensor, x_depth: torch.Tensor):
        x1 = self.MFE1(x_rgb)
        x2 = self.MFE2(x_depth)
        return torch.cat([self.SE1(x1), self.SE2(x2)], 1)

class MFA(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size):
        super(MFA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, int(out_channels/4), kernel_size=1)
        self.bn1 = nn.BatchNorm2d(int(out_channels/4))

        self.max_pool = nn.AdaptiveMaxPool2d(pool_size)
        self.conv2 = nn.Conv2d(in_channels, int(out_channels/4), kernel_size=1)
        self.bn2 = nn.BatchNorm2d(int(out_channels/4))

        self.conv3 = nn.Conv2d(in_channels, int(out_channels/4), kernel_size=1)
        self.bn3 = nn.BatchNorm2d(int(out_channels/4))
        self.conv3_1 = nn.Conv2d(int(out_channels/4), int(out_channels/4), kernel_size=(1,3), padding=(0,1))
        self.bn3_1 = nn.BatchNorm2d(int(out_channels/4))
        self.conv3_2 = nn.Conv2d(int(out_channels/4), int(out_channels/4), kernel_size=(3,1), padding=(1,0))
        self.bn3_2 = nn.BatchNorm2d(int(out_channels/4))

        self.conv4 = nn.Conv2d(in_channels, int(out_channels/4), kernel_size=1)
        self.bn4 = nn.BatchNorm2d(int(out_channels/4))
        self.conv4_1 = nn.Conv2d(int(out_channels/4), int(out_channels/4), kernel_size=(1,7), padding=(0,3))
        self.bn4_1 = nn.BatchNorm2d(int(out_channels/4))
        self.conv4_2 = nn.Conv2d(int(out_channels/4), int(out_channels/4), kernel_size=(7,1), padding=(3,0))
        self.bn4_2 = nn.BatchNorm2d(int(out_channels/4))

    def forward(self, x_in):
        x1 = self.conv1(x_in)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.max_pool(x_in)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)

        x3 = self.conv3(x_in)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x3 = self.conv3_1(x3)
        x3 = self.bn3_1(x3)
        x3 = F.relu(x3)
        x3 = self.conv3_2(x3)
        x3 = self.bn3_2(x3)
        x3 = F.relu(x3)

        x4 = self.conv4(x_in)
        x4 = self.bn4(x4)
        x4 = F.relu(x4)
        x4 = self.conv4_1(x4)
        x4 = self.bn4_1(x4)
        x4 = F.relu(x4)
        x4 = self.conv4_2(x4)
        x4 = self.bn4_2(x4)
        x4 = F.relu(x4)

        return torch.cat([x1, x2, x3, x4], 1)