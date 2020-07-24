import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from colorama import *
init()

from ..utils import box_utils


class MultiboxLoss(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device,
                 reduction:str = "sum",
                 num_classes:int = None,
                 class_reduction:bool = True,
                 verbose:int = 0):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.

        Parameters
        ----------
        class_reduction: bool, Optional, default=True
            If enabled, 
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
        self.num_classes = num_classes
        self.class_reduction = class_reduction
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

        if self.num_classes == None:
            self.num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            if self.neg_pos_ratio != -1:
                mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)
        if self.neg_pos_ratio != -1:
            confidence = confidence[mask, :]

        if self.num_classes == 2:
            one_hot_label = torch.eye(self.num_classes, dtype=torch.long)[labels].to(self.device)
            classification_loss = F.binary_cross_entropy_with_logits(confidence, one_hot_label, reduction=self.reduction)
        else:
            if self.neg_pos_ratio != -1:
                # smooth_l1_loss shape:torch.Size([407, 4]) batch mean:3.170289993286133 range:[3.635753387243312e-08, 7.234347343444824]
                # classification_loss shape:torch.Size([1628]) batch mean:1.1413660049438477 range:[1.510108232498169, 12.064205169677734]
                classification_loss = F.cross_entropy(confidence.reshape(-1, self.num_classes), labels[mask], reduction=self.reduction)
            else:
                # Note: classification loss results are slightly difference between class reduction is enabled or not.
                # class_reduction is enabled:
                # classification_loss shape:torch.Size([209568]) batch mean:3.3828086853027344 range:[0.015668869018554688, 12.064205169677734]
                # class_reduction is enabled:
                # classification_loss shape:torch.Size([24, 8732, 21]) batch mean:3.3609931468963623 range:[0.010865924879908562, 10.473838806152344]
                if self.class_reduction:
                    classification_loss = F.cross_entropy(confidence.reshape(-1, self.num_classes), labels.flatten(), reduction=self.reduction)
                else:
                    # naive computation of cross entropy is quite slow:
                    #     ce = - log_prob * prob
                    # In order to speed up, pytorch build-in function should be used as much as possible.
                    # https://stackoverflow.com/a/55643379
                    one_hot_label = torch.eye(self.num_classes, dtype=torch.long)[labels].to(self.device)
                    confidence_log_prob = torch.nn.functional.log_softmax(confidence)
                    classification_loss = torch.gather(-confidence_log_prob, 2, one_hot_label)

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
            if not self.class_reduction:
                norm_factor = num_pos * self.num_classes
            print("smooth_l1_loss shape:{0} batch mean:{1} range:[{2}, {3}]".format(smooth_l1_loss.shape, smooth_l1_loss.sum()/num_pos, torch.min(smooth_l1_loss), torch.max(smooth_l1_loss)))
            print("classification_loss shape:{0} batch mean:{1} range:[{2}, {3}]".format(classification_loss.shape, classification_loss.sum()/norm_factor, torch.min(classification_loss), torch.max(classification_loss)))
            print(Fore.CYAN + "MultiboxLoss.forward [out] -------------------------" + Style.RESET_ALL)

        return smooth_l1_loss/num_pos, classification_loss/num_pos
