from typing import List

import torch
import hydra
import pytorch_lightning as pl
from torch import nn
from torchvision import models
from torch.nn import functional as F


class FeatureExtractor(nn.Module):

    def __init__(self, input_size=(64, 320), output_len=20):
        super().__init__()

        h, w = input_size
        resnet = getattr(models, 'resnet18')(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

        self.pool = nn.AvgPool2d(kernel_size=(h // 32, 1))
        self.proj = nn.Conv2d(w // 32, output_len, kernel_size=1)

        self.num_output_features = self.cnn[-1][-1].bn2.num_features

    def apply_projection(self, x):
        """Use convolution to increase width of a features.
        Accepts tensor of features (shaped B x C x H x W).
        Returns new tensor of features (shaped B x C x H x W').
        """
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def forward(self, x):
        # Apply conv layers
        features = self.cnn(x)

        # Pool to make height == 1
        features = self.pool(features)

        # Apply projection to increase width
        features = self.apply_projection(features)

        return features


class SequencePredictor(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=False):
        super().__init__()

        self.num_classes = num_classes
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          bidirectional=bidirectional, batch_first=False)

        fc_in = hidden_size if not bidirectional else 2 * hidden_size
        self.fc = nn.Linear(in_features=fc_in,
                            out_features=num_classes)

    def _prepare_features_(self, x):
        """Change dimensions of x to fit RNN expected input.
        Accepts tensor x shaped (B x (C=1) x H x W).
        Returns new tensor shaped (W x B x H).
        """
        x = x.squeeze(1)
        x = x.permute(2, 0, 1)
        return x

    def forward(self, x):
        x = self._prepare_features_(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x


class CRNN(nn.Module):

    def __init__(
        self,
        alphabet: List[str],
        cnn_input_size=(64, 320),
        cnn_output_len: int = 20,
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 2,
        rnn_dropout: float = 0.3,
        rnn_bidirectional: bool = False
    ):
        super().__init__()
        self.alphabet = alphabet

        self.features_extractor = FeatureExtractor(
            input_size=cnn_input_size,
            output_len=cnn_output_len
        )

        self.sequence_predictor = SequencePredictor(
            input_size=self.features_extractor.num_output_features,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            num_classes=(len(alphabet) + 1),
            dropout=rnn_dropout,
            bidirectional=rnn_bidirectional
        )

    def forward(self, x):
        features = self.features_extractor(x)
        sequence = self.sequence_predictor(features)
        return sequence


class OCRDetectorTrainer(pl.LightningModule):
    def __init__(self, *, model, optimizer_config,
                 target_metric: str,
                 scheduler_config=None) -> None:
        super().__init__()
        self.model = model
        self._optimizer_config = optimizer_config
        self._scheduler_config = scheduler_config
        self._target_metric = target_metric

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        seq_gt = batch["seq"]
        seq_lens_gt = batch["seq_len"]
        predicted_logits = self.model(images)

        log_probs = F.log_softmax(predicted_logits, dim=-1)

        seq_lens_pred = torch.ones(
            predicted_logits.shape[1], dtype=torch.long, device=log_probs.device) * predicted_logits.shape[0]

        loss = F.ctc_loss(
            log_probs=log_probs,  # (T, N, C)
            targets=seq_gt,  # N, S or sum(target_lengths)
            input_lengths=seq_lens_pred,  # N
            target_lengths=seq_lens_gt  # N
        )

        return loss

    def validation_step(self, batch, batch_idx):
        self.log("valid", 1)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self._optimizer_config, self.model.parameters())

        opt_settings = {"optimizer": optimizer}
        if self._scheduler_config is not None:
            opt_settings["lr_scheduler"] = {"scheduler": hydra.utils.instantiate(
                self._scheduler_config, optimizer)}
            if isinstance(opt_settings["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau):
                opt_settings["lr_scheduler"]["monitor"] = "total loss"

        return opt_settings
