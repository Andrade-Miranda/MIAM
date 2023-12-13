"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function
from packaging import version
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnUNet.nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnUNet.nnunet.training.network_training.nnUNet_variants.loss_function.nnUNetTrainerV2_focalLoss import \
    FocalLoss
from torch import nn

import warnings
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.losses.focal_loss import FocalLoss
from monai.networks import one_hot
from monai.losses.dice import DiceLoss,GeneralizedDiceLoss
from monai.utils import LossReduction, Weight




##### TO CHECK ###############################################################################################
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
        
###########PICAI LOSSS##########################################################""""        
class FocalLossBin(nn.Module):
    """Focal loss function for binary segmentation."""

    def __init__(self, alpha=1, gamma=2, num_classes=2, reduction="sum"):
        super(FocalLossBin, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        targets = F.one_hot(targets[:, 0, ...].long(), num_classes=self.num_classes).float()
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

#################################################



##replace FocalLoss by fixed implemetation (and set smooth=0 in that one?)
class FL_and_CE_loss(nn.Module):
    def __init__(self, fl_kwargs=None, ce_kwargs=None, alpha=0.5,apply_nonlin=nn.Softmax(dim=1), aggregate="sum"):
        super(FL_and_CE_loss, self).__init__()
        if fl_kwargs is None:
            fl_kwargs = {}
        if ce_kwargs is None:
            ce_kwargs = {}

        self.aggregate = aggregate
        self.fl = FocalLossBin(alpha=fl_kwargs['alpha'])
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.alpha = alpha

    def forward(self, net_output, target):
        fl_loss = self.fl(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = self.alpha*fl_loss + (1-self.alpha)*ce_loss
        else:
            raise NotImplementedError("nah son")
        return result
    

class DiceFocalLoss(_Loss):
    """
    Compute both Dice loss and Focal Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Focal Loss is shown in ``monai.losses.FocalLoss``.

    ``gamma``, ``focal_weight`` and ``lambda_focal`` are only used for the focal loss.
    ``include_background`` and ``reduction`` are used for both losses
    and other parameters are only used for dice loss.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        gamma: float = 2.0,
        focal_weight: Optional[Union[Sequence[float], float, int, torch.Tensor]] = None,
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0,
    ) -> None:
        """
        Args:
            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `FocalLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `FocalLoss`.
            other_act: callable function to execute other activation layers, Defaults to ``None``.
                for example: `other_act = torch.tanh`. only used by the `DiceLoss`, not for `FocalLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            gamma: value of the exponent gamma in the definition of the Focal loss.
            focal_weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes).
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_focal: the trade-off weight value for focal loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=False,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.focal = FocalLoss(
            include_background=include_background,
            to_onehot_y=False,
            gamma=gamma,
            weight=focal_weight,
            reduction=reduction,
        )
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_focal < 0.0:
            raise ValueError("lambda_focal should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.to_onehot_y = to_onehot_y

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD]. The input should be the original logits
                due to the restriction of ``monai.losses.FocalLoss``.
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )
        if self.to_onehot_y:
            n_pred_ch = input.shape[1]
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)
        dice_loss = self.dice(input, target)
        focal_loss = self.focal(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss
        return total_loss #,dice_loss,focal_loss
    


class GeneralizedDiceFocalLoss(torch.nn.modules.loss._Loss):
    """Compute both Generalized Dice Loss and Focal Loss, and return their weighted average. The details of Generalized Dice Loss
    and Focal Loss are available at ``monai.losses.GeneralizedDiceLoss`` and ``monai.losses.FocalLoss``.

    Args:
        include_background (bool, optional): if False channel index 0 (background category) is excluded from the calculation.
            Defaults to True.
        to_onehot_y: whether to convert the ``target`` into the one-hot format,
            using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
        sigmoid (bool, optional): if True, apply a sigmoid function to the prediction. Defaults to False.
        softmax (bool, optional): if True, apply a softmax function to the prediction. Defaults to False.
        other_act (Optional[Callable], optional): callable function to execute other activation layers,
            Defaults to ``None``. for example: `other_act = torch.tanh`.
            only used by the `GeneralizedDiceLoss`, not for the `FocalLoss`.
        w_type (Union[Weight, str], optional): {``"square"``, ``"simple"``, ``"uniform"``}. Type of function to transform
            ground-truth volume to a weight factor. Defaults to ``"square"``.
        reduction (Union[LossReduction, str], optional): {``"none"``, ``"mean"``, ``"sum"``}. Specified the reduction to
            apply to the output. Defaults to ``"mean"``.
            - ``"none"``: no reduction will be applied.
            - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
            - ``"sum"``: the output will be summed.
        smooth_nr (float, optional): a small constant added to the numerator to avoid zero. Defaults to 1e-5.
        smooth_dr (float, optional): a small constant added to the denominator to avoid nan. Defaults to 1e-5.
        batch (bool, optional): whether to sum the intersection and union areas over the batch dimension before the dividing.
            Defaults to False, i.e., the areas are computed for each item in the batch.
        gamma (float, optional): value of the exponent gamma in the definition of the Focal loss. Defaults to 2.0.
        focal_weight (Optional[Union[Sequence[float], float, int, torch.Tensor]], optional): weights to apply to
            the voxels of each class. If None no weights are applied. The input can be a single value
            (same weight for all classes), a sequence of values (the length of the sequence hould be the same as
            the number of classes). Defaults to None.
        lambda_gdl (float, optional): the trade-off weight value for Generalized Dice Loss. The value should be
            no less than 0.0. Defaults to 1.0.
        lambda_focal (float, optional): the trade-off weight value for Focal Loss. The value should be no less
            than 0.0. Defaults to 1.0.

    Raises:
        ValueError: if either `lambda_gdl` or `lambda_focal` is less than 0.
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        w_type: Union[Weight, str] = Weight.SQUARE,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        gamma: float = 2.0,
        focal_weight: Optional[Union[Sequence[float], float, int, torch.Tensor]] = None,
        lambda_gdl: float = 1.0,
        lambda_focal: float = 1.0,
    ) -> None:
        super().__init__()
        self.generalized_dice = GeneralizedDiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            w_type=w_type,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.focal = FocalLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            gamma=gamma,
            weight=focal_weight,
            reduction=reduction,
        )
        if lambda_gdl < 0.0:
            raise ValueError("lambda_gdl should be no less than 0.0.")
        if lambda_focal < 0.0:
            raise ValueError("lambda_focal should be no less than 0.0.")
        self.lambda_gdl = lambda_gdl
        self.lambda_focal = lambda_focal

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): the shape should be BNH[WD]. The input should be the original logits
                due to the restriction of ``monai.losses.FocalLoss``.
            target (torch.Tensor): the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When the input and target tensors have different numbers of dimensions, or the target
                channel isn't either one-hot encoded or categorical with the same shape of the input.

        Returns:
            torch.Tensor: value of the loss.
        """
        if input.dim() != target.dim():
            raise ValueError(
                f"Input - {input.shape} - and target - {target.shape} - must have the same number of dimensions."
            )

        gdl_loss = self.generalized_dice(input, target)
        focal_loss = self.focal(input, target)
        total_loss: torch.Tensor = self.lambda_gdl * gdl_loss + self.lambda_focal * focal_loss
        return total_loss #,gdl_loss,focal_loss