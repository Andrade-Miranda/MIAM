"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function
from packaging import version
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=1e-1, contrast_mode='all',
                 base_temperature=1e-1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask #denominator
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) #log(exp(zi.zp/t))-log(sum_A(i)(exp(zi*za/t)))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) #sum_P(i)[log(exp(zi.zp/t))-log(sum_A(i)(exp(zi*za/t)))]/(1/P(i))

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos# extra termino para cambiar la loss
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.opt.nce_T=1000
        
    def forward(self, feat_q, feat_k):
        
  #       num_patches = feat_q.shape[1]
  #       dim = feat_q.shape[2]
  #       feat_k = feat_k.detach()

  #       # pos logit
  #       l_pos = torch.bmm(
  #           feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
  #       l_pos = l_pos.view(num_patches, 1)

  # # neg logit

  # # Should the negatives from the other samples of a minibatch be utilized?
  # # In CUT and FastCUT, we found that it's best to only include negatives
  # # from the same image. Therefore, we set
  # # --nce_includes_all_negatives_from_minibatch as False
  # # However, for single-image translation, the minibatch consists of
  # # crops from the "same" high-resolution image.
  # # Therefore, we will include the negatives from the entire minibatch.
  #       batch_dim_for_bmm = self.opt.batchSize

  #     # reshape features to batch size
  #       feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
  #       feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
  #       npatches = feat_q.size(1)
  #       l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        
  #       # diagonal entries are similarity between same features, and hence meaningless.
  #       # just fill the diagonal with very small number, which is exp(-10) and almost zero
  #       diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
  #       l_neg_curbatch.masked_fill_(diagonal, -10.0)
  #       l_neg = l_neg_curbatch.view(-1, npatches)

  #       out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

  #       loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
  #                                                       device=feat_q.device))        

        
        # move sample location to last dimension (B,C,S)
        feat_q=torch.moveaxis(feat_q,(1,2),(-1,-2))
        feat_k=torch.moveaxis(feat_k,(1,2),(-1,-2))
        feat_k = feat_k.detach()

        # pos logit
        l_pos = (feat_k * feat_q).sum(dim=1)[:, :, None]
        
        # reshape features to batch size
        l_neg= torch.bmm(feat_q.transpose(1, 2),feat_k)
        npatches = feat_q.size(-1)
        # The diagonal entries are not negatives. Remove them.
        identity_matrix = torch.eye(npatches,device=feat_q.device,dtype=self.mask_dtype)[None, :, :]
        l_neg.masked_fill_(identity_matrix, -float('inf'))

        # calculate logits: (B)x(S)x(S+1)
        logits = torch.cat((l_pos, l_neg), dim=2) / self.opt.nce_T

        # return PatchNCE loss
        predictions = logits.flatten(0, 1)

        loss = self.cross_entropy_loss(predictions, torch.zeros(predictions.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        
        
        return loss
        
        
class FocalLossBin(nn.Module):
    """Focal loss function for binary segmentation."""

    def __init__(self, alpha=1, gamma=2, num_classes=2, reduction="sum"):
        super(FocalLossBin, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets=targets[:, 0, ...].long()
        inputs = torch.sigmoid(inputs)
        targets = F.one_hot(targets, num_classes=self.num_classes).float()
        targets = torch.moveaxis(targets, (0, 1, 2, 3, 4), (0, 2, 3, 4, 1))
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = (inputs[-1] * targets[-1]) + ((1 - inputs[-1]) * (1 - targets[-1]))
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets[-1] + (1 - self.alpha) * (1 - targets[-1])
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
