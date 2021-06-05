from typing import List

import hydra
import torch
from torch import nn
from torchvision import ops
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import pytorch_lightning as pl

from utils import denormalize_tensor_to_image, draw_bbox


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
                 nms_iou_threshold: float = 0.5,
                 scheduler_config=None) -> None:
        super().__init__()
        self.model = model
        self._optimizer_config = optimizer_config
        self._scheduler_config = scheduler_config
        self._target_metric = target_metric
        self._nms_iou_threshold = nms_iou_threshold

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

        max_iou = 0
        min_iou = 0
        index_min = 0
        index_max = 0
        min_indices = []
        max_indices = []

        mean_iou = 0

        for i, (pred_info, true_info) in enumerate(zip(predicted_info, plates_info)):
            bbox_true = true_info["boxes"]
            bbox_pred = torch.round(pred_info["boxes"])

            bbox_indices = ops.nms(bbox_pred,
                                   pred_info["scores"], self._nms_iou_threshold).view(-1)

            high_score_indices = torch.nonzero(pred_info["scores"][bbox_indices] > 0.5).view(-1)
            bbox_indices = bbox_indices[high_score_indices].view(-1)

            if len(bbox_indices) > 0:
                bbox_pred = bbox_pred[bbox_indices]
                iou_avalues, _ = ops.box_iou(bbox_true, bbox_pred).max(dim=1)
                iou = iou_avalues.mean().item()

                if iou > max_iou:
                    max_iou = iou
                    index_max = i
                    min_indices = bbox_indices

                if iou < min_iou:
                    min_iou = iou
                    index_min = i
                    max_indices = bbox_indices

                mean_iou += iou

        for index, bbox_indices in zip((index_max, index_min), (max_indices, min_indices)):
            image = (denormalize_tensor_to_image(
                images[index].cpu()) * 255).permute(1, 2, 0).to(torch.uint8)
            fig = draw_bbox(image, plates_info[index]["boxes"].cpu(),
                            torch.round(predicted_info[index]["boxes"][bbox_indices]).cpu())

            tensorboard_logger = self.logger.experiment
            tensorboard_logger.add_figure(
                "Valid/Pred", fig, global_step=self.global_step, close=True)

        self.log(self._target_metric, mean_iou / len(batch))

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self._optimizer_config, self.model.parameters())

        opt_settings = {"optimizer": optimizer}
        if self._scheduler_config is not None:
            opt_settings["lr_scheduler"] = {"scheduler": hydra.utils.instantiate(
                self._scheduler_config, optimizer)}
            if isinstance(opt_settings["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau):
                opt_settings["lr_scheduler"]["monitor"] = "total loss"

        return opt_settings
