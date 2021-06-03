from typing import List
import hydra

import torch
from torch import nn
from torchvision import ops
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import pytorch_lightning as pl


class PlateDetector(nn.Module):
    def __init__(self, backbone, num_classes: int):
        super().__init__()

        self.detector = backbone
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features

        box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.detector.roi_heads.box_predictor = box_predictor

        mask_predictor = MaskRCNNPredictor(256, 256, num_classes)
        self.detector.roi_heads.mask_predictor = mask_predictor

    def freeze_backbone(self):
        for param in self.detector.parameters():
            if isinstance(param, nn.Module):
                param.eval()
            param.requires_grad_(False)

        for module in (self.detector.backbone.fpn, self.detector.rpn, self.detector.roi_heads):
            for param in module.parameters():
                if isinstance(param, nn.Module):
                    param.train()
                param.requires_grad_(True)

    def forward(self, images: torch.tensor, plates_info: List[dict] = None):
        return self.detector(images, plates_info)


class PlateTrainDetector(pl.LightningModule):
    def __init__(self, *, model, optimizer_config,
                 target_metric: str,
                 scheduler_config=None) -> None:
        super().__init__()
        self.model = model
        self._optimizer_config = optimizer_config
        self._scheduler_config = scheduler_config
        self._target_metric = target_metric

    def training_step(self, batch, batch_idx):
        images, plates_info = batch
        losses = self.model(images, plates_info)

        for key in losses:
            self.log(f"{key}", losses[key])

        total_loss = sum(losses.values())
        self.log("total loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, plates_info = batch
        predicted_info = self.model(images)

        miou = ops.box_iou(predicted_info["boxes"], plates_info["boxes"])

        self.log(self._target_metric, miou)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self._optimizer_config, self.model.parameters())

        opt_settings = {"optimizer": optimizer}
        if self._scheduler_config is not None:
            opt_settings["lr_scheduler"] = {"scheduler": hydra.utils.instantiate(
                self._scheduler_config, optimizer)}
            if isinstance(opt_settings["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau):
                opt_settings["lr_scheduler"]["monitor"] = "total loss"

        return opt_settings
