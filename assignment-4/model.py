import torch
from torch.nn import Module, Sequential as Seq, Linear as Lin, Dropout, PReLU, Tanh
from torch.nn.utils import weight_norm

class Decoder(Module):
    def __init__(self, args, dropout_prob=0.1,):
        super(Decoder, self).__init__()
        self.dropout = Dropout(p=dropout_prob)
        self.fc1 = Seq(weight_norm(Lin(  3, 512)), PReLU(), self.dropout)
        self.fc2 = Seq(weight_norm(Lin(512, 512)), PReLU(), self.dropout)
        self.fc3 = Seq(weight_norm(Lin(512, 512)), PReLU(), self.dropout)
        self.fc4 = Seq(weight_norm(Lin(512, 509)), PReLU(), self.dropout)
        self.fc5 = Seq(weight_norm(Lin(512, 512)), PReLU(), self.dropout)
        self.fc6 = Seq(weight_norm(Lin(512, 512)), PReLU(), self.dropout)
        self.fc7 = Seq(weight_norm(Lin(512, 512)), PReLU(), self.dropout)
        self.fc8 = Seq(Lin(512, 1), Tanh())

    def forward(self, x):
        x_in = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = torch.cat([x, x_in], dim=1)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        return x
