import torch
import torch.nn as nn


class CategoryLoss(nn.Module):
    def __init__(self, category_num, gpu_ids):
        super(CategoryLoss, self).__init__()
        emb = nn.Embedding(category_num, category_num, device=gpu_ids[0] if gpu_ids else None)
        emb.weight.data = torch.eye(category_num)
        self.emb = emb
        self.loss = nn.BCEWithLogitsLoss()
        self.gpu_ids = gpu_ids

    def forward(self, category_logits, labels):
        target = self.emb(labels.cuda())
        return self.loss(category_logits.cuda(), target)


class BinaryLoss(nn.Module):
    def __init__(self, real):
        super(BinaryLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.real = real

    def forward(self, logits):
        if self.real:
            labels = torch.ones(logits.shape[0], 1)
        else:
            labels = torch.zeros(logits.shape[0], 1)
        if logits.is_cuda:
            labels = labels.cuda()
        return self.bce(logits.cuda(), labels)
