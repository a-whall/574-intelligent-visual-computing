import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, num_classes):

        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, stride=1, padding=0, bias=False)
        self.norm1 = nn.LayerNorm(normalized_shape=[8, 106, 106])
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, stride=2, padding=0, groups=8, bias=False)
        self.norm2 = nn.LayerNorm(normalized_shape=[8, 24, 24])
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pconv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=1, padding=0, bias=True)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=0, groups=16, bias=False)
        self.norm3 = nn.LayerNorm(normalized_shape=[16, 6, 6])
        self.relu3 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pconv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)

        self.fc_conv = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=3, padding=0, bias=True)

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='leaky_relu')

        nn.init.xavier_normal_(self.pconv1.weight)
        nn.init.xavier_normal_(self.pconv2.weight)
        nn.init.xavier_normal_(self.fc_conv.weight)

        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)

        nn.init.zeros_(self.norm1.bias)
        nn.init.zeros_(self.norm2.bias)
        nn.init.zeros_(self.norm3.bias)

        nn.init.zeros_(self.pconv1.bias)
        nn.init.zeros_(self.pconv2.bias)

        nn.init.zeros_(self.fc_conv)

    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.pconv1(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.pconv2(x)

        x = self.fc_conv(x)

        return x