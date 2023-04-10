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

        # y = self.avg_pool(x)
        # # y = y.squeeze()
        # print(y.shape)
        # y = self.fc1(y)
        # x = self.relu1(y)
        # print(y.shape)
        # y = self.fc2(y)
        # print(y.shape)
        # print(x.shape)
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
        # print(avg_out.shape)
        x = torch.cat([avg_out, max_out], dim=1)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
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
        # print(x.shape)
        x = x.permute(0, 2, 1, 3)
        # print(x.shape)
        a, b, c, d = x.shape
        x = x.reshape(a, b, -1)
        x = x.unsqueeze(1)
        # print(x.shape)
        x = self.bn1(x)
        x = self.Conv1(x)
        x = torch.relu(x)
        # print(x.shape)
        x = self.chat1(x) * x
        x = self.spat1(x) * x
        # print(x.shape)
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
        return self.tsnenepool(x)
        # # y = y.squeeze()
        # print(y.shape)
        # y = self.fc1(y)
        # x = self.relu1(y)
        # print(y.shape)
        # y = self.fc2(y)
        # print(y.shape)
        # print(x.shape)
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
        # print(avg_out.shape)
        x = torch.cat([avg_out, max_out], dim=1)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
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
        self.dropout = dropout  # dropout参数azz
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
        # self.bn4 = nn.BatchNorm1d(self.lstm_in)
        # self.rsbu2 = RSBU_CW(in_channels=48, out_channels=48, kernel_size=3)
        # self.drop = nn.Dropout2d(p=dropout)
        # self.global_pool1 = nn.AdaptiveMaxPool2d((1, 1))
        # self.global_pool2 = nn.AdaptiveMaxPool2d((1, 1))
        # self.at = nn.Sequential(nn.Linear(self.lstm_in, self.lstm_in // 3),
        #                         nn.ReLU(inplace=True),
        #                         nn.Linear(self.lstm_in // 3, self.lstm_in),
        #                         nn.Sigmoid())
        # self.sat1 = SelfAttention(num_attention_heads=1,
        #                           input_size=self.lstm_in,
        #                           hidden_size=self.lstm_in,
        #                           hidden_dropout_prob=0.1)
        self.sat1 = ChannelAttention(in_planes=self.lstm_in)
        self.spat1 = SpatialAttention()
        self.spat2 = SpatialAttention()
        self.spat3 = SpatialAttention()
        self.chat1 = ChannelAttention(in_planes=32)
        self.chat2 = ChannelAttention(in_planes=16)
        self.chat3 = ChannelAttention(in_planes=8)
        # self.mlp_1 = Linear(110, 110 // 3)
        # self.mlp_2 = Linear(110 // 3, 110)
        self.lstm = nn.LSTM(input_size=self.lstm_in,
                            hidden_size=self.lstm_out,
                            num_layers=2,
                            dropout=0.1)
        self.linear1 = nn.Linear(20, 20 // 2)
        self.dropout = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(20 // 2, 1)

    def forward(self, x, y):
        # print(x.shape)
        x = x.permute(0, 2, 1, 3)
        # print(x.shape)
        a, b, c, d = x.shape
        x = x.reshape(a, b, -1)
        x = x.unsqueeze(1)
        # print(x.shape)
        x = self.bn1(x)
        x = self.Conv1(x)
        x = torch.relu(x)
        # print(x.shape)
        x = self.chat1(x) * x
        x = self.spat1(x) * x
        # print(x.shape)
        x = self.bn2(x)
        x = self.Conv2(x)
        x = torch.relu(x)
        x = self.chat2(x) * x
        x = self.spat2(x) * x
        # print(x.shape)
        x = self.bn3(x)
        x = self.Conv3(x)
        x = torch.relu(x)
        x = self.chat3(x) * x
        # print(x.shape)
        x = self.spat3(x) * x
        # x = self.bn3(x)
        x = x.squeeze(2)
        self.ccc = x
        # print(x.shape)
        # print(x[0, 1, :])
        # print(x[190, 1, :])
        # x = self.sat2(x)
        # print(x.shape)

        # y = y.permute(0, 2, 1)
        # at = nn.AdaptiveMaxPool1d(1)(y)
        # at = at.squeeze(-1)
        # at = self.at(at)
        # at = at.unsqueeze(-1)
        # at = at.expand_as(y)
        # y = torch.mul(y, at)
        y = y.permute(0, 2, 1)
        y = y.unsqueeze(3)
        # print(y.shape)
        y = self.sat1(y) * y
        # print(y.shape)
        y = y.squeeze(3)
        y = y.permute(0, 2, 1)
        y, _ = self.lstm(y)
        y = y[:, -1, :]
        # y = torch.relu(y)
        # print(y.shape)
        # x = x.flatten(1)
        x = nn.AdaptiveMaxPool1d(1)(x)
        x = x.squeeze()
        # print(x.shape)
        # x = x.view(a, b, -1)
        # print(x.shape)
        output = torch.cat((x, y), dim=1)
        self.tsne = output
        # print(output.shape)
        output = self.linear1(output)
        # output = torch.sigmoid(output)
        output = self.dropout(output)
        output = torch.relu(output)
        output = self.linear2(output)
        output = torch.sigmoid(output)
        # output = torch.relu(output)
        output = output.squeeze()
        return output

    def getx(self):
        return self.ccc

    def gettsne(self):
        return self.tsne


# x = torch.randn(119, 5, 80, 80)
# y = torch.randn(119, 5, 24)
# test = net(24, 12, 1, 8, 0.1)
# z = test(x, y)
# x = torch.randn(615, 12, 48, 48)
# test = RSBU_CW(12, 12, 3)
# y = test(x)
# x = torch.randn(615, 48, 2, 2)
# test = RSBU_CW(48, 48, 3)
# y = test(x)