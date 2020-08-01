from typing import Union
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from colorama import *
init()

from ..utils import box_utils

class MultiboxLoss(nn.Module):
    def __init__(self, priors,
                 iou_threshold,
                 neg_pos_ratio: Union[int, None],
                 center_variance,
                 size_variance,
                 device,
                 reduction:str = "sum",
                 num_classes:int = None,
                 class_reduction:bool = True,
                 verbose:int = 0):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.

        Parameters
        ----------
        neg_pos_ratio: Union[int, None], required
            If None, use focal loss[1] for class-imbalanced problem.
            If integer, use OHEM(Online Hard Exsample Mining) in stead of above method.
        class_reduction: bool, Optional, default=True
            If enabled, dimension of classes on output tensor will be reduced.

        References
        ----------
        [1] Liu, Wei, et al. "Ssd: Single shot multibox detector." European conference on computer vision. Springer, Cham, 2016.
        [2] A. Shrivastava, A. Gupta, and R. Girshick. Training regionbased object detectors with online hard example mining. In CVPR, 2016.
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
        self.num_pos = None
        self.verbose = verbose

        if self.neg_pos_ratio is None:
            self.gamma = 2.0
            self.alpha = 0.75

    def showTensor(self, v, name):
        if len(v.shape) < 2:
            v = v / self.num_pos
        print(name + " shape:{0} range:[{1}, {2}]".format(v.shape, torch.min(v), torch.max(v)))

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
            if self.neg_pos_ratio is not None:
                mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)
        if self.neg_pos_ratio is not None:
            confidence = confidence[mask, :]

        if self.num_classes == 2:
            # binary-class classfication
            one_hot_label = torch.eye(self.num_classes, dtype=torch.long)[labels].to(self.device)
            classification_loss = F.binary_cross_entropy_with_logits(confidence, one_hot_label, reduction=self.reduction)
        else:
            # multi-class classification
            if self.neg_pos_ratio is not None:
                # Online hard example mining
                if self.verbose > 0:
                    print("Online Hard Example Mining (ratio: {0})".format(self.neg_pos_ratio))
                # smooth_l1_loss shape:torch.Size([407, 4]) batch mean:3.170289993286133 range:[3.635753387243312e-08, 7.234347343444824]
                # classification_loss shape:torch.Size([1628]) batch mean:1.1413660049438477 range:[1.510108232498169, 12.064205169677734]
                classification_loss = F.cross_entropy(confidence.reshape(-1, self.num_classes), labels[mask], reduction=self.reduction)

                pos_mask = labels > 0
                predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
                gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
                smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction=self.reduction)
            else:
                # Focal loss
                if self.verbose > 0:
                    print("Focal Loss")

                predicted_locations = predicted_locations.reshape(-1, 4)
                gt_locations = gt_locations.reshape(-1, 4)
                smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction="none")

                # Note: classification loss results are slightly difference between class reduction is enabled or not.
                # class_reduction is enabled:
                # classification_loss shape:torch.Size([209568]) batch mean:3.3828086853027344 range:[0.015668869018554688, 12.064205169677734]
                # class_reduction is disabled:
                # classification_loss shape:torch.Size([24, 8732, 21]) batch mean:3.3609931468963623 range:[0.010865924879908562, 10.473838806152344]
                if self.class_reduction:
                    # https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289
                    ce_loss = F.cross_entropy(confidence.reshape(-1, self.num_classes), labels.flatten(), reduction="none")
                    pt = torch.exp( - ce_loss)
                    focal_term = self.alpha*(1 - pt)**self.gamma
                    classification_loss = focal_term * ce_loss
                    smooth_l1_loss = focal_term.unsqueeze(1).expand(-1, 4) * smooth_l1_loss
                else:
                    # naive computation of cross entropy is quite slow:
                    #     ce = - log_prob * prob
                    # In order to speed up, pytorch build-in function should be used as much as possible.
                    # https://stackoverflow.com/a/55643379
                    one_hot_label = torch.eye(self.num_classes, dtype=torch.long)[labels].to(self.device)
                    confidence_log_prob = torch.nn.functional.log_softmax(confidence)
                    classification_loss = torch.gather(-confidence_log_prob, 2, one_hot_label)

        if self.num_pos is None:
            self.num_pos = gt_locations.size(0)
        if self.verbose > 0:
            print("num_pos:{0}".format(self.num_pos))
            self.showTensor(smooth_l1_loss, "smooth_l1_loss")
            self.showTensor(classification_loss, "classification_loss")
            print(Fore.CYAN + "MultiboxLoss.forward [out] -------------------------" + Style.RESET_ALL)

        return smooth_l1_loss/self.num_pos, classification_loss/self.num_pos
