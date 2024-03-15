import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class InnerProduct(nn.Module):
    def __init__(self, in_feature=128, out_feature=10575):
        super(InnerProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)


    def forward(self, x, label):
        # label not used
        output = F.linear(x, self.weight)
        return output


if __name__ == '__main__':
    pass
