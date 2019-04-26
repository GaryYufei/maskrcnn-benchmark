# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
from torch import nn
import h5py


@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None
        self.cfg = cfg

        num_inputs = in_channels
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

        if not cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_INIT:
            self.cls_score = nn.Linear(num_inputs, num_classes)
            nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
            nn.init.constant_(self.cls_score.bias, 0)
        else:
            self.embedding_fc = nn.Linear(num_inputs, 512)
            nn.init.normal_(self.cls_fc.weight, mean=0, std=0.01)
            nn.init.constant_(self.cls_fc.bias, 0)

            self.cls_score = nn.Linear(512, num_classes)
            with h5py.File(config.MODEL.ROI_BOX_HEAD.EMBEDDING_WEIGHT, 'r') as fin:
                self.cls_score.weight.data.copy_(torch.FloatTensor(fin['cls']['W']))
                self.cls_score.bias.data.copy_(torch.FloatTensor(fin['cls']['b']))

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        bbox_pred = self.bbox_pred(x)
        if self.cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_INIT:
            x = self.embedding_fc(x)
        cls_logit = self.cls_score(x)
        return cls_logit, bbox_pred

@registry.ROI_BOX_PREDICTOR.register("FastRCNNAttrPredictor")
class FastRCNNAttrPredictor(FastRCNNPredictor):
    def __init__(self, cfg, in_channels):
        super(FastRCNNAttrPredictor, self).__init__(cfg, in_channels)
        num_classes = cfg.MODEL.ROI_BOX_HEAD.ATTR_NUM_CLASSES
        num_inputs = in_channels

        if not cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_INIT:
            self.attr_score = nn.Linear(num_inputs, num_classes)
            nn.init.normal_(self.attr_score.weight, std=0.01)
            nn.init.constant_(self.attr_score.bias, 0)
        else:
            self.attr_score = nn.Linear(512, num_classes)
            with h5py.File(config.MODEL.ROI_BOX_HEAD.EMBEDDING_WEIGHT, 'r') as fin:
                self.cls_score.weight.data.copy_(torch.FloatTensor(fin['attr']['W']))
                self.cls_score.bias.data.copy_(torch.FloatTensor(fin['attr']['b']))
            

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        bbox_pred = self.bbox_pred(x)

        if self.cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_INIT:
            x = self.embedding_fc(x)

        cls_logit = self.cls_score(x)
        attr_scores = self.attr_score(x)

        return attr_scores, cls_logit, bbox_pred


@registry.ROI_BOX_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        self.cfg = cfg
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

        if not cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_INIT:
            self.cls_score = nn.Linear(representation_size, num_classes)
            nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
            nn.init.constant_(self.cls_score.bias, 0)
        else:
            self.embedding_fc = nn.Linear(num_inputs, 512)
            nn.init.normal_(self.cls_fc.weight, mean=0, std=0.01)
            nn.init.constant_(self.cls_fc.bias, 0)

            self.cls_score = nn.Linear(512, num_classes)
            with h5py.File(config.MODEL.ROI_BOX_HEAD.EMBEDDING_WEIGHT, 'r') as fin:
                self.cls_score.weight.data.copy_(torch.FloatTensor(fin['cls']['W']))
                self.cls_score.bias.data.copy_(torch.FloatTensor(fin['cls']['b']))

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        bbox_deltas = self.bbox_pred(x)

        if self.cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_INIT:
            x = self.embedding_fc(x)
        scores = self.cls_score(x)
        
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
            self.attr_score = nn.Linear(512, num_classes)
            with h5py.File(config.MODEL.ROI_BOX_HEAD.EMBEDDING_WEIGHT, 'r') as fin:
                self.cls_score.weight.data.copy_(torch.FloatTensor(fin['attr']['W']))
                self.cls_score.bias.data.copy_(torch.FloatTensor(fin['attr']['b']))
            

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)

        bbox_deltas = self.bbox_pred(x)

        if self.cfg.MODEL.ROI_BOX_HEAD.EMBEDDING_INIT:
            x = self.embedding_fc(x)
            
        scores = self.cls_score(x)
        attr_scores = self.attr_score(x)

        return attr_scores, scores, bbox_deltas


def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)
