import math
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import Linear


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class net(nn.Module):

    def __init__(self,
                 lstm_in=24,
                 lstm_out=12,
                 conv_in=16,
                 conv_out=8,
                 dropout=0.1):
        super(net, self).__init__()
        self.lstm_in = lstm_in  # 节点表示向量的输入特征数
        self.lstm_out = lstm_out  # 节点表示向量的输出特征数
        self.conv_in = conv_in
        self.conv_out = conv_out
        self.dropout = dropout  # dropout参数
        self.Conv1 = nn.Conv2d(in_channels=self.conv_in,
                               out_channels=32,
                               kernel_size=10,
                               stride=5)
        self.bn1 = nn.BatchNorm2d(1)
        self.Conv2 = nn.Conv2d(in_channels=32,
                               out_channels=16,
                               kernel_size=5,
                               stride=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.Conv3 = nn.Conv2d(in_channels=16,
                               out_channels=self.conv_out,
                               kernel_size=4,
                               stride=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.sat1 = ChannelAttention(in_planes=self.lstm_in)
        self.spat1 = SpatialAttention()
        self.spat2 = SpatialAttention()
        self.spat3 = SpatialAttention()
        self.chat1 = ChannelAttention(in_planes=32)
        self.chat2 = ChannelAttention(in_planes=16)
        self.chat3 = ChannelAttention(in_planes=8)
        self.lstm = nn.LSTM(input_size=self.lstm_in,
                            hidden_size=self.lstm_out,
                            num_layers=2,
                            dropout=0.1)
        self.linear1 = nn.Linear(20, 20 // 2)
        self.dropout = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(20 // 2, 1)

    def forward(self, x, y):
        x = x.permute(0, 2, 1, 3)
        a, b, c, d = x.shape
        x = x.reshape(a, b, -1)
        x = x.unsqueeze(1)
        x = self.bn1(x)
        x = self.Conv1(x)
        x = torch.relu(x)
        x = self.chat1(x) * x
        x = self.spat1(x) * x
        x = self.bn2(x)
        x = self.Conv2(x)
        x = torch.relu(x)
        x = self.chat2(x) * x
        x = self.spat2(x) * x
        x = self.bn3(x)
        x = self.Conv3(x)
        x = torch.relu(x)
        x = self.chat3(x) * x
        x = self.spat3(x) * x
        x = x.squeeze(2)
        self.ccc = x
        y = y.permute(0, 2, 1)
        y = y.unsqueeze(3)
        y = self.sat1(y) * y
        y = y.squeeze(3)
        y = y.permute(0, 2, 1)
        y, _ = self.lstm(y)
        y = y[:, -1, :]
        x = nn.AdaptiveMaxPool1d(1)(x)
        x = x.squeeze()
        output = torch.cat((x, y), dim=1)
        self.tsne = output
        output = self.linear1(output)
        output = self.dropout(output)
        output = torch.relu(output)
        output = self.linear2(output)
        output = torch.sigmoid(output)
        output = output.squeeze()
        return output

    def getx(self):
        return self.ccc

    def gettsne(self):
        return self.tsne
