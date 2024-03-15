import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class SphereMarginProduct(nn.Module):
    def __init__(self, in_feature, out_feature, m=4, base=1000.0, gamma=0.0001, power=2, lambda_min=5.0):
        super().__init__()
        assert m in [1, 2, 3, 4], 'margin should be 1, 2, 3 or 4'
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.m = m
        self.base = base
        self.gamma = gamma
        self.power = power
        self.lambda_min = lambda_min
        self.iter = 0
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        # duplication formula
        self.margin_formula = [
            lambda x : x ** 0,
            lambda x : x ** 1,
            lambda x : 2 * x ** 2 - 1,
            lambda x : 4 * x ** 3 - 3 * x,
            lambda x : 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x : 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, x, label):
        self.iter += 1
        self.cur_lambda = max(self.lambda_min, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        cos_theta = F.linear(F.normalize(x), F.normalize(self.weight))
        cos_theta = cos_theta(-1, 1)

        cos_m_theta = None #self.margin_formula(self.m)(cos_theta) -- error
        theta = cos_theta.data.acos()
        k = ((self.m * theta) / math.pi).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        phi_theta_ = (self.cur_lambda * cos_theta + phi_theta) / (1 + self.cur_lambda)
        norm_of_feature = torch.norm(x, 2, 1)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        output = one_hot * phi_theta_ + (1 - one_hot) * cos_theta
        output *= norm_of_feature.view(-1, 1)

        return output


class SphereLoss(nn.Module):
    def __init__(self, m, scale, num_class, use_gpu):
        super(SphereLoss, self).__init__()
        self.m = m
        self.scale = scale
        self.num_class = num_class
        self.use_gpu = use_gpu
        self.loss = nn.CrossEntropyLoss()

    def theta_to_psi(self, theta_yi_i):
        k = torch.floor(theta_yi_i * self.m / pi)
        sign = torch.full(k.shape, -1)
        if self.use_gpu:
            sign = sign.cuda()
        co = torch.pow(sign, k)
        cos_m_theta_yi_i = torch.cos(self.m * theta_yi_i)
        return co * cos_m_theta_yi_i - 2 * k

    def forward(self, y_hat, y):
        y = torch.unsqueeze(y, 0)
        label = torch.reshape(y, (y.shape[1], 1))
        one_hot = torch.zeros(y.shape[1], self.num_class)
        if self.use_gpu:
            one_hot = one_hot.cuda()
        one_hot = one_hot.scatter_(1, label, 1)
        mask = one_hot.to(torch.bool)
        #theta(yi, i)
        cos_theta_yi_i = torch.masked_select(y_hat, mask)
        theta_yi_i = torch.acos(cos_theta_yi_i)
        psi_yi_i = self.theta_to_psi(theta_yi_i)

        fc = y_hat * 1.0
        index = torch.Tensor(range(y_hat.shape[0]))
        fc[index.long(), y.long()] = psi_yi_i[index.long()]
        fc = fc * self.scale

        y = y.squeeze(0)
        loss = self.loss(fc, y)

        return loss


if __name__ == '__main__':
    pass
