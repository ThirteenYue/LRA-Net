from typing import Dict
import torch
import torch.nn as nn

from torch.nn import functional as F

class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) +
                            torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels,kernel_size=3, padding=1, bias=False, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret

# 扩大感受野、获得更多feature map
# # without bn version
class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

# 通道注意力
class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        # self.gamma = torch.nn.parameter(torch.tensor(torch.zeros(1)))
        self.gamma = torch.tensor(0)
        self.softmax =torch.nn.Softmax(-1)   # 对每一行进行softmax

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × C × C
        """
        m_batchsize, C, height, width = x.size()
        # A -> (N,C,HW)
        proj_query = x.view(m_batchsize, C, -1)
        # A -> (N,HW,C)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        # 矩阵乘积，通道注意图：X -> (N,C,C)
        energy = torch.bmm(proj_query, proj_key)
        # 这里实现了softmax用最后一维的最大值减去了原始数据，获得了一个不是太大的值
        # 沿着最后一维的C选择最大值，keepdim保证输出和输入形状一致，除了指定的dim维度大小为1
        # expand_as表示以复制的形式扩展到energy的尺寸
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        # A -> (N,C,HW)
        proj_value = x.view(m_batchsize, C, -1)
        # XA -> （N,C,HW）
        out = torch.bmm(attention, proj_value)
        # output -> (N,C,H,W)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


# 空间金字塔池化注意力
class _PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim,reduction_dim,setting):
        super(_PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:      # 对应不同的池化操作,单个bin,多个bin
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)
        self.conv1=nn.Conv2d(reduction_dim*4+in_dim, 1, kernel_size=1)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.upsample(f(x), x_size[2:], mode='bilinear'))
        out = torch.cat(out,1)
        out = self.conv1(out)
        out=x*out
        return out


# 回路残差结构
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu =  nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels * 7, out_channels * 7, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * 7)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv11 = nn.Conv2d(out_channels * 7, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):


        # 一层残差开始
        residual = x # 保留初始
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out =out+ residual # 结束

        # 第二层
        residual1 = out  # 保留第一次
        out1 = self.conv1(residual1)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)
        out1 =out1+ residual1  # 结束

        # 通道合并
        # 上拼接
        cat1 = torch.cat([residual, residual1], dim=1)
        cat2 = torch.cat([cat1, out1], dim=1)
        # 下拼接
        cat3=torch.cat([out1, residual1], dim=1)
        cat4 = torch.cat([cat3, residual], dim=1)
        cat_out = torch.cat([cat2, cat4, out1], dim=1)

        # 结束
        out2 = self.conv2(cat_out)
        out2 = self.bn2(out2)
        out2 = self.relu2(out2)
        out2 = self.dropout(out2)

        # 1*1卷积
        out3 = self.conv11(out2)

        return out3


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None,):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(

            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            # 加入ASPP


            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),


            # ASPP(out_channels, out_channels),
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2),
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class LRANet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_CoordConv = CoordConv(in_channels, base_c)
        self.in_conv = DoubleConv(base_c, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.out_conv = OutConv(base_c, num_classes)
        self.res1 = ResidualBlock(base_c, base_c)
        self.res2 = ResidualBlock(base_c * 2, base_c * 2)
        self.res3 = ResidualBlock(base_c * 4, base_c * 4)
        self.res4 = ResidualBlock(base_c * 16 // factor, base_c * 16 // factor)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)

        self.sa1 = _PyramidPoolingModule(base_c * 2, 1, (1, 3, 5, 7))
        self.ca1 = CAM_Module(base_c * 2)
        self.sa2 = _PyramidPoolingModule(base_c * 4, 1, (1, 3, 5, 7))
        self.ca2 = CAM_Module(base_c * 4)
        self.sa3 = _PyramidPoolingModule(base_c * 8, 1, (1, 3, 5, 7))
        self.ca3 = CAM_Module(base_c * 8)
        self.sa = _PyramidPoolingModule(base_c * 16 // factor, 1, (2, 3, 6, 12))
        self.ca = CAM_Module(base_c * 16 // factor)
        self.aspp = ASPP(base_c * 16 // factor, base_c * 16 // factor)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        xx = self.in_CoordConv(x)
        x1 = self.in_conv(xx)
        x2 = self.down1(x1)
        x2 = self.ca1(x2)
        x2 = self.sa1(x2)
        x3 = self.down2(x2)
        x3 = self.ca2(x3)
        x3 = self.sa2(x3)
        x4 = self.down3(x3)
        x4 = self.ca3(x4)
        x4 = self.sa3(x4)
        x5 = self.down4(x4)
        x51 = self.ca(x5)
        x52 = self.sa(x51)
        x53 = self.aspp(x52)

        x5 = x53

        # 加入最后一层残差块
        x4 = self.res4(x4)
        x3 = self.res3(x3)
        x2 = self.res2(x2)
        x1 = self.res1(x1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        # return {"out": logits}
        return logits

