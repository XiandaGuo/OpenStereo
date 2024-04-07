import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss:  # nn.BCELoss()
    def __init__(self):
        self.reduction = 'mean'

    def __call__(self, inputs, targets):
        """
        no c dim, because each item means a prob
        :param inputs: [bz, d1, d2, ...], after softmax or sigmod
        :param targets: [bz, d1, d2, ...], same shape with inputs, between 0 and 1, such as 0, 0.7, 1
        :return:
        t = torch.rand(size=[4, 3])
        a = torch.randn(size=[4, 3])
        b = F.sigmoid(a)
        nn.BCELoss()(b, t) == BCELoss()(b, t)
        """
        bce_loss = -(targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        if self.reduction == 'mean':
            bce_loss = torch.mean(bce_loss)
        else:
            bce_loss = torch.sum(bce_loss)
        return bce_loss


class BCEWithLogitsLoss:  # nn.BCEWithLogitsLoss()
    def __call__(self, inputs, targets):
        """
        t = torch.rand(size=[4, 3])
        a = torch.randn(size=[4, 3])
        b = F.sigmoid(a)
        nn.BCEWithLogitsLoss()(a, t) == BCEWithLogitsLoss()(a, t) == BCELoss()(b, t)
        """
        inputs = F.sigmoid(inputs)
        return BCELoss()(inputs, targets)


class CrossEntropyLoss:  # nn.CrossEntropyLoss()
    def __call__(self, inputs, targets):
        """
        :param inputs: [bz, c, d1, d2, ...], before softmax
        :param targets: [bz, d1, d2, ...], value in the range [0, c), or [bz, c, d1, d2, ...], between 0 and 1, such as 0.7
        :return:
        """
        inputs = F.softmax(inputs, dim=1)
        inputs = torch.log(inputs)

        if targets.shape != inputs.shape:
            return nn.NLLLoss()(inputs, targets)

        loss = - torch.sum(targets * inputs, dim=1)
        loss = torch.mean(loss)
        return loss


class KLDivLoss:
    def __init__(self):
        self.reduction = 'mean'

    def __call__(self, inputs, targets):
        loss_pointwise = targets * (torch.log(targets) - inputs)

        if self.reduction == "mean":
            loss = loss_pointwise.mean()
        elif self.reduction == "batchmean":
            loss = loss_pointwise.sum() / inputs.shape[0]
        else:
            loss = loss_pointwise

        return loss


if __name__ == '__main__':
    pred = torch.randn(3, 5, 4, 4, requires_grad=True)
    # target = torch.empty(3, 4, 4, dtype=torch.long).random_(5)
    target = torch.randn(3, 5, 4, 4).softmax(dim=1)
    # print(nn.CrossEntropyLoss()(pred, target))
    # print(CrossEntropyLoss()(pred, target))
    print(nn.KLDivLoss()(F.log_softmax(pred, dim=1), target))
    print(KLDivLoss()(F.log_softmax(pred, dim=1), target))