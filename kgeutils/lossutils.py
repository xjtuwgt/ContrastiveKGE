from torch import nn
from torch import Tensor as T
import torch.nn.functional as F
import torch

get_scalar = lambda x: x.detach().item()

class HingeLoss(nn.Module):
    def __init__(self, margin):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self, score: T, label):
        loss = self.margin - label * score
        loss[loss < 0] = 0
        return loss

class LogisticLoss(nn.Module):
    def __init__(self):
        super(LogisticLoss, self).__init__()

    def forward(self, score: T, label):
        loss = -label * score
        loss = F.softplus(loss)
        return loss

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, score: T, label):
        loss = -(label * torch.log(F.sigmoid(score)) + (1 - label) * torch.log(1 - F.sigmoid(score)))
        return loss

class LogsigmoidLoss(nn.Module):
    def __init__(self):
        super(LogsigmoidLoss, self).__init__()

    def forward(self, score: T, label):
        loss =  -F.logsigmoid(label * score)
        return loss

class LossGenerator(nn.Module):
    def __init__(self, pairwise: bool, margin: float = None, loss_genre='Logsigmoid', neg_adversarial_sampling: bool = False, adversarial_temperature: float=1.0):
        super(LossGenerator, self).__init__()

        if loss_genre == 'Hinge':
            self.neg_label = -1
            assert margin is not None
            self.loss_criterion = HingeLoss(margin)
        elif loss_genre == 'Logistic':
            self.neg_label = -1
            self.loss_criterion = LogisticLoss()
        elif loss_genre == 'Logsigmoid':
            self.neg_label = -1
            self.loss_criterion = LogsigmoidLoss()
        elif loss_genre == 'BCE':
            self.neg_label = 0
            self.loss_criterion = BCELoss()
        else:
            raise ValueError('loss genre %s is not support' % loss_genre)

        if self.pairwise and loss_genre not in ['Logistic', 'Hinge']:
            raise ValueError('{} loss cannot be applied to pairwise loss function'.format(loss_genre))

    def _get_pos_loss(self, pos_score):
        return self.loss_criterion(pos_score, 1)

    def _get_neg_loss(self, neg_score):
        return self.loss_criterion(neg_score, self.neg_label)

    def get_total_loss(self, pos_score, neg_score, edge_weight=None):
        log = {}
        if edge_weight is None:
            edge_weight = 1
        if self.pairwise:
            pos_score = pos_score.unsqueeze(-1)
            loss = torch.mean(self.loss_criterion((pos_score - neg_score), 1) * edge_weight)
            log['loss'] = get_scalar(loss)
            return loss, log

        pos_loss = self._get_pos_loss(pos_score) * edge_weight
        neg_loss = self._get_neg_loss(neg_score) * edge_weight
        # MARK - would average twice make loss function lose precision?
        # do mean over neg_sample
        if self.neg_adversarial_sampling:
            neg_loss = torch.sum(torch.softmax(neg_score * self.adversarial_temperature, dim=-1).detach() * neg_loss, dim=-1)
        else:
            neg_loss = torch.mean(neg_loss, dim=-1)
        # do mean over chunk
        neg_loss = torch.mean(neg_loss)
        pos_loss = torch.mean(pos_loss)
        loss = (neg_loss + pos_loss) / 2
        log['pos_loss'] = get_scalar(pos_loss)
        log['neg_loss'] = get_scalar(neg_loss)
        log['loss'] = get_scalar(loss)
        return loss, log

        return