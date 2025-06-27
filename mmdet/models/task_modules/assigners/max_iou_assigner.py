# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Optional, Union

import torch
from mmengine.structures import InstanceData

from mmdet.registry import TASK_UTILS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

count = 0
gt_weight_list = torch.tensor([])
anchor_in_per_gt_list = torch.tensor([])
gt_areas_list = torch.tensor([])
sum_anchor_in_per_gt_list = torch.tensor([])
count_gt = 0
count_pos = 0


@TASK_UTILS.register_module()
class MaxIoUAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each prior.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    If ``alpha`` is not None, it means that the dynamic cost
    ATSSAssigner is adopted, which is currently only used in the DDOD.

    Args:
        topkpos (int): number of priors selected in each level
        alpha (float, optional): param of cost rate for each proposal only
            in DDOD. Defaults to None.
        iou_calculator (:obj:`ConfigDict` or dict): Config dict for iou
            calculator. Defaults to ``dict(type='BboxOverlaps2D')``
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes. Defaults to -1.
    """

    def __init__(self,
                 pos_iou_thr: float,
                 neg_iou_thr: Union[float, tuple],
                 min_pos_iou: float = .0,
                 gt_max_assign_all: bool = True,
                 ignore_iof_thr: float = -1,
                 ignore_wrt_candidates: bool = True,
                 match_low_quality: bool = True,
                 gpu_assign_thr: float = -1,
                 alpha: Optional[float] = None,
                 iou_calculator: dict = dict(type='BboxSiM2D')) -> None:
        self.alpha = alpha
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py
    def assign(
            self,
            pred_instances: InstanceData,
            gt_instances: InstanceData,
            gt_instances_ignore: Optional[InstanceData] = None
    ) -> AssignResult:
        """Assign gt to priors.

        The assignment is done in following steps

        1. compute iou between all prior (prior of all pyramid levels) and gt
        2. compute center distance between all prior and gt
        3. on each pyramid level, for each gt, select k prior whose center
           are closest to the gt center, so we total select k*l prior as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt

        If ``alpha`` is not None, and ``cls_scores`` and `bbox_preds`
        are not None, the overlaps calculation in the first step
        will also include dynamic cost, which is currently only used in
        the DDOD.

        Args:
            pred_instances (:obj:`InstaceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors, points, or bboxes predicted by the model,
                shape(n, 4).
            num_level_priors (List): Number of bboxes in each level
            gt_instances (:obj:`InstaceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            gt_instances_ignore (:obj:`InstaceData`, optional): Instances
                to be ignored during training. It includes ``bboxes``
                attribute data that is ignored during training and testing.
                Defaults to None.

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        gt_bboxes = gt_instances.bboxes
        priors = pred_instances.priors
        gt_labels = gt_instances.labels
        # pred_bboxes = pred_instances.bboxes
        if gt_instances_ignore is not None:
            gt_bboxes_ignore = gt_instances_ignore.bboxes
        else:
            gt_bboxes_ignore = None

        INF = 100000000
        priors = priors[:, :4]
        num_gt, num_priors = gt_bboxes.size(0), priors.size(0)

        message = 'Invalid alpha parameter because cls_scores or ' \
                  'bbox_preds are None. If you want to use the ' \
                  'cost-based ATSSAssigner,  please set cls_scores, ' \
                  'bbox_preds and self.alpha at the same time. '

        # compute iou between all bbox and gt
        if self.alpha is None:
            # ATSSAssigner
            # shape: [num_priors, num_gts]
            overlaps = self.iou_calculator(priors, gt_bboxes)
            if ('scores' in pred_instances or 'bboxes' in pred_instances):
                warnings.warn(message)
        else:
            # Dynamic cost ATSSAssigner in DDOD
            assert ('scores' in pred_instances
                    and 'bboxes' in pred_instances), message
            cls_scores = pred_instances.scores
            bbox_preds = pred_instances.bboxes

            # compute cls cost for bbox and GT
            cls_cost = torch.sigmoid(cls_scores[:, gt_labels])

            # compute iou between all bbox and gt
            overlaps = self.iou_calculator(bbox_preds, gt_bboxes)

            # make sure that we are in element-wise multiplication
            assert cls_cost.shape == overlaps.shape

            # overlaps is actually a cost matrix
            overlaps = cls_cost ** (1 - self.alpha) * overlaps ** self.alpha

        # assign 0 by default
        # 首先默认所有样本均为负样本 0
        assigned_gt_inds = overlaps.new_full((num_priors,),
                                             0,
                                             dtype=torch.long)
        # assigned_gt_inds: [num_priors]
        if num_gt == 0 or num_priors == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_priors,))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            assigned_labels = overlaps.new_full((num_priors,),
                                                -1,
                                                dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0
        num_level_priors = [overlaps.size(0)]
        for level, priors_per_level in enumerate(num_level_priors):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + priors_per_level
            distances_per_level = overlaps[start_idx:end_idx, :]
            selectable_k = priors_per_level
            _, topkpos_idxs_per_level = distances_per_level.topk(
                selectable_k, dim=0, largest=True)
            candidate_idxs.append(topkpos_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_priors
        priors_cx = (priors[:, 0] + priors[:, 2]) / 2.0
        priors_cy = (priors[:, 1] + priors[:, 3]) / 2.0
        ep_priors_cx = priors_cx.view(1, -1).expand(
            num_gt, num_priors).contiguous().view(-1)
        ep_priors_cy = priors_cy.view(1, -1).expand(
            num_gt, num_priors).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # prior center and gt side
        l_ = ep_priors_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_priors_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_priors_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_priors_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01

        sum_in_gts = is_in_gts.sum(1)
        assigned_gt_inds[sum_in_gts > 0] = -1

        sum_anchor_in_per_gt = is_in_gts.sum(0)
        global sum_anchor_in_per_gt_list
        sum_anchor_in_per_gt_list = torch.concat([sum_anchor_in_per_gt_list, sum_anchor_in_per_gt.cpu()])
        anchor_in_per_gt = sum_anchor_in_per_gt / sum_anchor_in_per_gt_list.float().mean(0)
        # anchor_in_per_gt = sum_anchor_in_per_gt / sum_anchor_in_per_gt.float().mean(0)
        ones = torch.ones([num_gt]).cuda()
        anchor_in_per_gt = torch.max(anchor_in_per_gt.int(), ones)

        overlaps_thr_per_gt = candidate_overlaps[0, :]
        # for i in range(num_gt):
        #     overlaps_thr_per_gt[i] = candidate_overlaps[int(anchor_in_per_gt[i]) - 1][i]
        row_indices = anchor_in_per_gt[:num_gt].long() - 1
        col_indices = torch.arange(num_gt, device=overlaps_thr_per_gt.device)
        overlaps_thr_per_gt[:num_gt] = candidate_overlaps[row_indices, col_indices]

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        assigned_labels = assigned_gt_inds.new_full((num_priors,), -1)
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]

        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
