# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from maskrcnn_benchmark.modeling import registry

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms, boxlist_nms_index
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder

@registry.ROI_BOX_POSTPROCESS.register("PostProcessor")
class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        if len(x) == 3:
            class_logits, box_regression, features = x
            attr_logits = None
        elif len(x) == 4:
            attr_logits, class_logits, box_regression, features = x
        class_prob = F.softmax(class_logits, -1)

        if attr_logits is not None:
            _, attr_word = torch.max(attr_logits, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)
        features = features.split(boxes_per_image, dim=0)
        if attr_logits is not None:
            attr_word = attr_word.split(boxes_per_image, dim=0)
        results = []
        if attr_logits is not None:
            for attr_w, prob, boxes_per_img, image_shape, feature in zip(
                attr_word, class_prob, proposals, image_shapes, features
            ):
                boxlist = self.prepare_boxlist(boxes_per_img, prob, feature, image_shape, attr=attr_w)
                boxlist = boxlist.clip_to_image(remove_empty=False)
                boxlist = self.filter_results(boxlist, num_classes)
                results.append(boxlist)
        else:
            for prob, boxes_per_img, image_shape, feature in zip(
                class_prob, proposals, image_shapes, features
            ):
                boxlist = self.prepare_boxlist(boxes_per_img, prob, feature, image_shape)
                boxlist = boxlist.clip_to_image(remove_empty=False)
                boxlist = self.filter_results(boxlist, num_classes)
                results.append(boxlist)
        
        return results

    def prepare_boxlist(self, boxes, scores, features, image_shape, attr=None):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        boxlist.add_field("features", features)
        if attr is not None:
            boxlist.add_field("attr", attr)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)
        features = boxlist.get_field("features").reshape(scores.size(0), -1)

        if boxlist.has_field('attr'):
            attrs = boxlist.get_field("attr")
        else:
            attrs = None

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            scores_j = scores[inds, j]
            features_j = features[inds]

            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class.add_field("features", features_j)
            if attrs is not None:
                attrs_j = attrs[inds]
                boxlist_for_class.add_field("attrs", attrs_j)
            
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result

@registry.ROI_BOX_POSTPROCESS.register("ExactionPostProcessor")
class ExactionPostProcessor(PostProcessor):

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)
        features = boxlist.get_field("features").reshape(scores.size(0), -1)

        if boxlist.has_field('attr'):
            attrs = boxlist.get_field("attr")
        else:
            attrs = None

        device = scores.device
        result = []

        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        _conf = np.zeros((boxes.shape[0], num_classes))
        for j in range(1, num_classes):
            boxes_j = boxes[:, j * 4 : (j + 1) * 4]
            scores_j = scores[:, j]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            keep = boxlist_nms_index(
                boxlist_for_class, self.nms
            )
            _conf[keep.cpu(), j] = scores_j[keep].cpu()

        max_conf = np.max(max_conf, axis=1)
        max_cls = np.argmax(max_conf, axis=1)

        boxes = boxes.cpu()
        scores = scores.cpu()
        features = features.cpu()

        keep_boxes = np.where(max_conf >= self.score_thresh)[0]
        if len(keep_boxes) < 10:
            keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
        elif len(keep_boxes) > self.detections_per_img:
            keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

        keep_labels = max_cls[keep_boxes]
        keep_boxes = np.zeros((keep_labels.shape[0], 4), dtype=np.float32)
        selected_boxes = boxes[keep_boxes]
        for i in range(keep_labels.shape[0]):
            l = keep_labels[i]
            keep_boxes[i] = selected_boxes[i, l * 4 : (l + 1) * 4]
        final_boxlist = BoxList(keep_boxes, boxlist.size, mode="xyxy")
        final_boxlist.add_field("features", features[keep_boxes])
        final_boxlist.add_field("labels", keep_labels)
        if attrs is not None:
            final_boxlist.add_field("attrs", attrs[keep_boxes])

        return final_boxlist


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    func = registry.ROI_BOX_POSTPROCESS[
        cfg.MODEL.ROI_HEADS.POSTPROCESS_TYPE
    ]

    postprocessor = func(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg
    )
    return postprocessor
