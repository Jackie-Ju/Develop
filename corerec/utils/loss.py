import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class CELoss(nn.Module):
    def __init__(self, config):
        super(CELoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        return self.loss(pred, target)

class MSELoss(nn.Module):
    def __init__(self, config):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        return self.loss(pred, target)

class BPRLoss(nn.Module):
    def __init__(self, config):
        super(BPRLoss, self).__init__()
        self.gamma = config.loss.gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()

        return loss

class EmbLoss(nn.Module):
    """
        EmbLoss, regularization on embeddings
    """
    def __init__(self, config):
        super(EmbLoss, self).__init__()
        self.norm = config.loss.norm

    def forward(self, embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(input=torch.norm(embedding, p=self.norm), exponent=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss

class GCNLoss(nn.Module):
    def __init__(self, config):
        super(GCNLoss, self).__init__()
        self.BPRLoss = BPRLoss(config=config)
        self.EmbLoss = EmbLoss(config=config)
        self.require_pow = config.loss.require_pow
        self.reg_weight = config.loss.reg_weight

    def forward(self, pos_score, neg_score, ego_embds):
        bpr_loss = self.BPRLoss(pos_score, neg_score)
        reg_loss = self.EmbLoss(ego_embds, require_pow=self.require_pow)

        loss = bpr_loss + self.reg_weight * reg_loss
        return loss

class CEKLLoss(nn.Module):
    def __init__(self, config):
        super(CEKLLoss, self).__init__()
        self.update = 0
        self.anneal_cap = config.loss.anneal_cap
        self.total_anneal_steps = config.loss.total_anneal_steps

    def forward(self, z, mu, logvar, rating_matrix):
        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1. * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        # KL loss
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)) * anneal

        # CE loss
        ce_loss = -(F.log_softmax(z, 1) * rating_matrix).sum(1).mean()
        return ce_loss + kl_loss