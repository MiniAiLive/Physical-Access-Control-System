import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class CosineMarginProduct(nn.Module):
    def __init__(self, in_feature=128, out_feature=10575, s=30.0, m=0.35):
        super(CosineMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)


    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = self.s * (cosine - one_hot * self.m)
        return output


if __name__ == '__main__':
    pass
