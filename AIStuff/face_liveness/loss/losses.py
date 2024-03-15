import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def linear_combination(x, y, epsilon):
    """ define the linear combination function """
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    """ reduce the loss value """
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class DefaultFocalLoss(nn.Module):
    """ define the default focal loss """
    def __init__(self, gamma=2, eps=1e-7):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, target):
        """ calculate the loss value with input and target value """
        logp = self.ce(x, target)
        p = torch.exp(0 - logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class DepthFocalLoss(nn.Module):
    """ define the depth focal loss """
    def __init__(self, gamma=1, eps=1e-7):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.MSELoss(reduction='none')

    def forward(self, x, target):
        """ calculate the loss value with input and target value """
        loss = self.ce(x, target)
        loss = (loss) ** self.gamma
        return loss.mean()


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations
            for each minibatch. However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, device, class_num, alpha=None, gamma=2, size_average=True):
        super().__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1)) # 0.25 *
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(torch.tensor(alpha))
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.device = device

    def forward(self, inputs, targets):
        """ calculate the loss value with input and target value """
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
        # print('========batch size=============>', N, C)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        # print("=======targets======>: ", targets)
        class_mask.scatter_(1, ids.data, 1.)
        # print('======class mask====>: ', class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(self.device)
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        # print('-----bacth_loss------: ', batch_loss.size())

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def accuracy(pred, target, topk=1):
    """ calculate the accuracy """
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, 1, True, True)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


# https://arxiv.org/pdf/1511.05042.pdf
class TalyorCrossEntroyLoss(nn.Module):
    """ define the Talyor cross entropy loss """

    def forward(self, logits, labels):
        """ calculate the loss value with input and target value """
        #batch_size, num_classes =  logits.size()
        # labels = labels.view(-1,1)
        # logits = logits.view(-1,num_classes)

        talyor_exp = 1 + logits + logits**2
        loss = talyor_exp.gather(dim=1, index=labels.view(-1,1)).view(-1) /talyor_exp.sum(dim=1)
        loss = loss.mean()

        return loss


class SmoothCrossEntropy(nn.Module):
    """ define the smooth cross entropy """
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        """ calculate the loss value with input and target value """
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)
