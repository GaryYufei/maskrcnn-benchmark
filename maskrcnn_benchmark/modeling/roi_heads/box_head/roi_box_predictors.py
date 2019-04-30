# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
from torch import nn
import h5py
import torch

@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None
        self.cfg = cfg

        num_inputs = cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_DIM if cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_INIT else in_channels
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(in_channels, num_bbox_reg_classes * 4)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

        if not cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_INIT:
            nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
            nn.init.constant_(self.cls_score.bias, 0)
        else:
            self.tanh = nn.Tanh()
            self.embedding_fc = nn.Linear(in_channels, cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_DIM)
            nn.init.normal_(self.embedding_fc.weight, mean=0, std=0.01)
            nn.init.constant_(self.embedding_fc.bias, 0)
            with h5py.File(cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_WEIGHT, 'r') as fin:
                self.cls_score.weight.data.copy_(torch.FloatTensor(fin['cls']['W']))
                self.cls_score.bias.data.copy_(torch.FloatTensor(fin['cls']['b']))
                self.cls_score.weight.requires_grad = False
                self.cls_score.bias.requires_grad = False

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)

        bbox_pred = self.bbox_pred(x)

        if self.cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_INIT:
            cls_vec = self.tanh(self.embedding_fc(x))
        else:
            cls_vec = x

        cls_logit = self.cls_score(cls_vec)
        return cls_logit, bbox_pred

@registry.ROI_BOX_PREDICTOR.register("FastRCNNAttrPredictor")
class FastRCNNAttrPredictor(FastRCNNPredictor):
    def __init__(self, cfg, in_channels):
        super(FastRCNNAttrPredictor, self).__init__(cfg, in_channels)
        num_classes = cfg.MODEL.ROI_BOX_HEAD.ATTR_NUM_CLASSES
        num_inputs = cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_DIM if cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_INIT else in_channels
        self.attr_score = nn.Linear(num_inputs, num_classes)

        if not cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_INIT:
            nn.init.normal_(self.attr_score.weight, std=0.01)
            nn.init.constant_(self.attr_score.bias, 0)
        else:
            with h5py.File(cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_WEIGHT, 'r') as fin:
                self.attr_score.weight.data.copy_(torch.FloatTensor(fin['attr']['W']))
                self.attr_score.bias.data.copy_(torch.FloatTensor(fin['attr']['b']))
                self.attr_score.weight.requires_grad = False
                self.attr_score.bias.requires_grad = False
            

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)

        bbox_pred = self.bbox_pred(x)

        if self.cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_INIT:
            _vec = self.tanh(self.embedding_fc(x))
        else:
            _vec = x

        cls_logit = self.cls_score(_vec)
        attr_scores = self.attr_score(_vec)

        return attr_scores, cls_logit, bbox_pred


@registry.ROI_BOX_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        self.cfg = cfg
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_DIM if cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_INIT else in_channels


        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        self.cls_score = nn.Linear(representation_size, num_classes)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

        if not cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_INIT:
            nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
            nn.init.constant_(self.cls_score.bias, 0)
        else:

            with h5py.File(cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_WEIGHT, 'r') as fin:
                self.cls_score.weight.data.copy_(torch.FloatTensor(fin['cls']['W']))
                self.cls_score.bias.data.copy_(torch.FloatTensor(fin['cls']['b']))
                self.cls_score.weight.requires_grad = False
                self.cls_score.bias.requires_grad = False


    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)

        bbox_deltas = self.bbox_pred(x)

        if self.cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_INIT:
            [cls_vec, _] = torch.split(x, [512, 1536], dim=-1)
        else:
            cls_vec = x

        scores = self.cls_score(cls_vec)
        
        return scores, bbox_deltas

@registry.ROI_BOX_PREDICTOR.register("FPNAttrPredictor")
class FPNAttrPredictor(FPNPredictor):
    def __init__(self, cfg, in_channels):
        super(FPNAttrPredictor, self).__init__(cfg, in_channels)
        num_classes = cfg.MODEL.ROI_BOX_HEAD.ATTR_NUM_CLASSES
        representation_size = in_channels

        if not cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_INIT:
            self.attr_score = nn.Linear(representation_size, num_classes)
            nn.init.normal_(self.attr_score.weight, std=0.01)
            nn.init.constant_(self.attr_score.bias, 0)
        else:
            self.attr_score = nn.Linear(cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_DIM, num_classes)
            with h5py.File(cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_WEIGHT, 'r') as fin:
                self.attr_score.weight.data.copy_(torch.FloatTensor(fin['attr']['W']))
                self.attr_score.bias.data.copy_(torch.FloatTensor(fin['attr']['b']))
                self.attr_score.weight.requires_grad = False
                self.attr_score.bias.requires_grad = False
            

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)

        bbox_deltas = self.bbox_pred(x)
        if self.cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_INIT:
            [cls_vec, attr_vec, _] = torch.split(x, [512, 512, 1024], dim=-1)
        else:
            cls_vec, attr_vec = x, x

        scores = self.cls_score(cls_vec)
        attr_scores = self.attr_score(attr_vec)

        return attr_scores, scores, bbox_deltas


def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)
