import os
import pathlib
import warnings

import hydra
import torch
from hydra.core.config_store import ConfigStore
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from dataset import PlatesDetectionDataModule
from model import PlateTrainDetector, PlateDetector
from config_data import TrainConfig
from transforms import get_train_transform

cs = ConfigStore()
cs.store("train_detector", node=TrainConfig)


def get_model():
    backbone = models.detection.maskrcnn_resnet50_fpn(
        pretrained=True,
        pretrained_backbone=True,
        progress=True,
    )

    model = PlateDetector(backbone, 2)
    model.freeze_backbone()

    return model


@hydra.main(config_name="train_detector")
def main(train_config: TrainConfig):
    os.chdir(hydra.utils.get_original_cwd())

    pl.seed_everything(train_config.seed)

    model = get_model()
    train_tranaforms = get_train_transform()

    data = PlatesDetectionDataModule(data_dir=train_config.data_dir,
                                     train_size=train_config.train_size,
                                     seed=train_config.seed,
                                     max_size=train_config.max_image_size,
                                     num_workers=train_config.num_workers,
                                     batch_size=train_config.detector.batch_size,
                                     train_transforms=train_tranaforms,
                                     val_transforms=train_tranaforms)

    target_metric_name = "mIoU"

    exp_dir = pathlib.Path(train_config.exp_dir)

    checkpoint_dir = exp_dir / "checkpoint"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    train_module = PlateTrainDetector(model=model,
                                      optimizer_config=train_config.optimizer,
                                      scheduler_config=train_config.scheduler,
                                      target_metric=target_metric_name)

    checkpoint_callback = callbacks.ModelCheckpoint(monitor=target_metric_name + '_step',
                                                    dirpath=checkpoint_dir,
                                                    filename=f"{{step}}-{{{target_metric_name}:.4f}}",
                                                    verbose=True,
                                                    save_last=True,
                                                    save_top_k=2,
                                                    mode="max",
                                                    save_weights_only=False)

    lr_monitor = callbacks.LearningRateMonitor(logging_interval='step')

    log_dir = exp_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    logger = TensorBoardLogger(str(log_dir))

    gpus = -1 if torch.cuda.is_available() else None

    if gpus is None:
        warnings.warn("GPU is not available. Try train on CPU. It may will bew very slow")

    trainer = pl.Trainer(amp_backend="native",
                         gpus=gpus,
                         logger=logger,
                         auto_select_gpus=True,
                         benchmark=True,
                         check_val_every_n_epoch=train_config.check_val_every_n_epoch,
                         flush_logs_every_n_steps=train_config.flush_logs_every_n_steps,
                         default_root_dir=str(exp_dir),
                         deterministic=False,
                         fast_dev_run=train_config.fast_dev_run,
                         progress_bar_refresh_rate=10,
                         precision=train_config.precision,
                         max_epochs=train_config.max_epochs,
                         val_check_interval=train_config.val_check_interval,
                         callbacks=[checkpoint_callback, lr_monitor]
                         )

    trainer.fit(train_module, datamodule=data)


if __name__ == "__main__":
    main()
