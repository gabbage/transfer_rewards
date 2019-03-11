import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ignite.metrics import Accuracy
from torch.autograd import Function, Variable
from torch.nn import Module


class SimpleMarginRankingLoss(object):
    def __init__(self, margin=0., reduction='mean', str_params=None):
        super(SimpleMarginRankingLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

        # Overwrite the preset parameters with a parameter string of form "{margin:s}_{reduction:s}"
        if str_params:
            splits = str_params.split("_")

            if len(splits) >= 1:
                self.margin = splits[0]

                if self.margin != 'max':
                    self.margin = float(self.margin)

            if len(splits) == 2:
                self.reduction = splits[1]

    def __call__(self, input, target, margin=None):
        assert len(input.size()) == 2, "For MarginRankingLoss the prediction Tensor must have two dimensions!"
        dim = input.size(1)
        assert dim > 1 and dim % 2 == 0, "For MarginRankingLoss the prediction must have a second dimension greater " \
                                         "than 1 and even!"
        input1, input2 = input.split((dim // 2, dim // 2), dim=1)

        if self.margin == 'max':
            assert margin is not None, "If maximum margin shall be used the margin has to be set on each loss " \
                                       "computation via the margin argument!"

            sample_losses = []

            for x1, x2, t, m in zip(input1.split(1, dim=0), input2.split(1, dim=0), target, margin):
                sample_losses.append(F.margin_ranking_loss(x1.view(1), x2.view(1), t.view(1), margin=m.item(),
                                                           reduction='none'))

            return torch.mean(torch.cat(tuple(sample_losses)))
        else:
            return F.margin_ranking_loss(input1, input2, target.view(-1, 1), margin=self.margin, reduction=self.reduction)


class WARP(Function):
    @staticmethod
    def forward(ctx, inputs, targets, max_num_trials=None):
        batch_size = targets.size(0)
        nb_labels = targets.size(1)

        if max_num_trials is None:
            max_num_trials = nb_labels - 1

        assert max_num_trials >= 1

        # positive_indices = torch.zeros_like(inputs)
        negative_indices = torch.zeros_like(inputs)
        L = torch.zeros(batch_size)

        all_labels_idx = np.arange(nb_labels)

        # J = torch.nonzero(targets)

        for i in range(batch_size):
            msk = np.ones(nb_labels, dtype=bool)
            j = torch.argmax(targets[i, :]).item()
            msk[j] = False

            # initialize the sample_score_margin
            sample_score_margin = -1.0
            num_trials = 0

            neg_labels_idx = all_labels_idx[msk]

            while sample_score_margin <= 0 and num_trials < max_num_trials:
                # randomly sample a negative label
                neg_idx = np.random.choice(neg_labels_idx, 1).item()
                msk[neg_idx] = False
                neg_labels_idx = all_labels_idx[msk]

                num_trials += 1
                # calculate the score margin
                sample_score_margin = 1.0 + inputs[i, neg_idx].item() - inputs[i, j].item()

            if sample_score_margin < 0:
                # checks if no violating examples have been found
                continue
            else:
                loss_weight = np.log(np.floor((float(nb_labels) - 1) / float(num_trials)))
                L[i] = loss_weight
                negative_indices[i, neg_idx] = 1

        loss = L * (1 - torch.sum(targets * inputs, dim=1) + torch.sum(negative_indices * inputs, dim=1))

        ctx.save_for_backward(inputs, targets)
        ctx.L = L
        ctx.negative_indices = negative_indices

        return torch.sum(loss, dim=0, keepdim=True)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        inputs, targets = ctx.saved_tensors
        L = Variable(torch.unsqueeze(ctx.L, 1), requires_grad=False)
        negative_indices = Variable(ctx.negative_indices, requires_grad=False)
        grad_input = grad_output * L * (negative_indices - targets)

        return grad_input, None, None


class WARPLoss(Module):
    def __init__(self, max_num_trials=None):
        super(WARPLoss, self).__init__()
        self.max_num_trials = max_num_trials

    def forward(self, inputs, target):
        return WARP.apply(inputs, target, self.max_num_trials)


class PairwiseMSE(Module):
    def __init__(self):
        super(PairwiseMSE, self).__init__()

    def forward(self, inputs, target):
        assert len(inputs.size()) == 2 and inputs.size(1) == 2, "Dimension mismatch for PairwiseMSE!"
        targets = torch.index_select(torch.FloatTensor([[1, 0], [0, 1]]), 0, torch.clamp(1 - target, 0, 1).long().cpu())

        if inputs.is_cuda:
            targets = targets.cuda()

        return torch.mean(torch.sum(F.mse_loss(inputs, targets, reduction='none'), dim=1))


class SaferAccuracy(Accuracy):
    def update(self, output):
        y_pred, y = self._check_shape(output)
        self._check_type((y_pred, y))

        if self._type == "binary":
            indices = torch.round(y_pred).type(y.type())
        elif self._type == "multiclass":
            indices = torch.max(y_pred, dim=1)[1]

        if y.dtype != torch.long:
            y = (1 - torch.clamp(y, 0, 1)).long()

        correct = torch.eq(indices, y).view(-1)
        self._num_correct += torch.sum(correct).item()
        self._num_examples += correct.shape[0]


if __name__ == '__main__':
    # w = WARPLoss()
    # a = torch.tensor([1.], requires_grad=True)
    # t = torch.FloatTensor([[0., 0., 0., 1., 0.], [0., 0., 1., 0., 0.]])
    # y_pred = torch.FloatTensor([[0.49, 0.35, 0.02, 0.40, 0.22], [0.59, 0.56, 0.54, 0.5, 0.4]])
    # loss = w(a * y_pred, t)
    # print(loss)
    # loss.backward()
    # exit(0)

    a = torch.FloatTensor([0.])
    x = np.linspace(-2., 2., 101, dtype=np.float32)
    b = torch.from_numpy(x)
    margins = [0, 1, 2, 3]
    t = torch.FloatTensor([1.]).expand_as(b).view(-1, 1)

    plt.figure()
    plt.title("Margin Ranking Loss ($x_1=0, x_2 \in [-2, 2]$)")
    acc = SaferAccuracy()

    for m in margins:
        smr = SimpleMarginRankingLoss(str_params='{}_none'.format(m))
        i = torch.cat([a.expand_as(b).view(-1, 1), b.view(-1, 1)], dim=1)
        r = smr(i, t)
        print("m={}, {}".format(m, r[50]))
        acc.update((i, t))

        plt.plot(x, r.cpu().numpy(), label="margin={}".format(m))

    print(acc.compute())

    plt.ylabel("Loss")
    plt.xlabel("Value of $x_2$")
    plt.legend()
    plt.grid()
    plt.savefig("latex/mid-term/pics/margin_ranking_loss.pdf")
    plt.show()
