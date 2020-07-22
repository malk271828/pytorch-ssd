import torch.nn as nn
import torch.nn.functional as F
import torch
from colorama import *
init()

from ..utils import box_utils


class MultiboxLoss(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device,
                 reduction:str = "sum",
                 binary:bool = False,
                 verbose:int = 0):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)
        self.reduction = reduction
        self.device = device
        self.binary = binary
        self.verbose = verbose

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        Return:
            Tuple of regression loss and classification loss.

        """
        if self.verbose > 0:
            print(Fore.CYAN + "MultiboxLoss.forward [in] -------------------------" + Style.RESET_ALL)

        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            if self.neg_pos_ratio != -1:
                mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)
        if self.neg_pos_ratio != -1:
            confidence = confidence[mask, :]
        if self.binary:
            one_hot_label = torch.eye(num_classes)[labels].to(self.device)
            classification_loss = F.binary_cross_entropy_with_logits(confidence, one_hot_label, reduction=self.reduction)
        else:
            if self.neg_pos_ratio != -1:
                classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], reduction=self.reduction)
            else:
                # https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py
                # compute cross entropy by-hand
                one_hot_label = torch.eye(num_classes)[labels].to(self.device)
                log_pr_confidence = F.log_softmax(confidence)
                classification_loss = - one_hot_label * log_pr_confidence

        if self.neg_pos_ratio == -1:
            pos_mask = torch.ones(predicted_locations.shape[:2], dtype=torch.uint8)
        else:
            pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction=self.reduction)
        num_pos = gt_locations.size(0)

        if self.verbose > 0:
            print("num_pos:{0}".format(num_pos))
            if self.neg_pos_ratio == -1:
                print("log_pr_confidence shape:{0} batch mean:{1} range:[{2}, {3}]".format(log_pr_confidence.shape, log_pr_confidence.sum()/num_pos, torch.min(log_pr_confidence), torch.max(log_pr_confidence)))
            print("smooth_l1_loss shape:{0} batch mean:{1} range:[{2}, {3}]".format(smooth_l1_loss.shape, smooth_l1_loss.sum()/num_pos, torch.min(smooth_l1_loss), torch.max(smooth_l1_loss)))
            print("classification_loss shape:{0} batch mean:{1} range:[{2}, {3}]".format(classification_loss.shape, classification_loss.sum()/num_pos, torch.min(classification_loss), torch.max(classification_loss)))
            print(Fore.CYAN + "MultiboxLoss.forward [out] -------------------------" + Style.RESET_ALL)

        return smooth_l1_loss/num_pos, classification_loss/num_pos
