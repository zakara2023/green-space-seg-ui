"""
aegis_bolt.py

The lightning AeGIS wrapper module.
"""
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import AdamW
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall
)
from torchmetrics.segmentation import GeneralizedDiceScore
from lightning import LightningModule

from model.aegis import AeGIS
from model.scheduler import CosineStepDecay


def param_groups_lrd(model, blocks: int, base_lr: float=1e-4, weight_decay=0.05, layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_groups = {}

    num_layers = blocks + 1

    scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n.endswith('.bias'):
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_groups:
            this_scale = scales[layer_id]
            param_groups[group_name] = {
                "lr": this_scale * base_lr,
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "group_name": group_name,
                "params": [],
                "param_names": []
            }

        param_groups[group_name]["param_names"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


def get_layer_id(name: str, num_layers: int) -> int:
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    is_cls = 'cls' in name
    is_pos = 'pos_encoder' in name
    is_emb = 'src_embed' in name

    if is_cls or is_pos or is_emb:
        return 0
    elif 'encoder.encoder.encoder' in name:
        return int(name.split('.')[3]) + 1

    return num_layers


class AeGISBolt(LightningModule):
    """The lightning AeGIS wrapper module."""

    def __init__(self,
                 aegis: AeGIS,
                 warmup: int,
                 epochs: int,
                 lr: float=1e-5,
                 min_lr: float=1e-6,
                 weight_decay: float=0.05,
                 layer_decay: float=0.65,
                 accumulate_grad_batches: int=4):
        super().__init__()
        self.aegis = aegis
        self.warmup = warmup
        self.epochs = epochs
        self.min_lr = min_lr
        self.lr = lr
        self.layer_decay = layer_decay
        self.weight_decay = weight_decay
        self.accumulate_grad_batches = accumulate_grad_batches
        self.automatic_optimization = False
        self.acc = BinaryAccuracy()
        self.iou = BinaryJaccardIndex()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.dice = GeneralizedDiceScore(num_classes=1, include_background=False)

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        return self.aegis(x, meta)

    def _step(self, batch: torch.Tensor, _: int) -> torch.Tensor:
        """Perform forward pass and calculate loss."""
        x, meta, y = batch
        y_hat = self(x, meta).squeeze(1)
        acc = self.acc(y_hat, y)
        iou = self.iou(y_hat, y)
        loss = binary_cross_entropy_with_logits(y_hat, y)
        return loss, acc, iou

    def _step_eval(self, batch: torch.Tensor, _: int) -> torch.Tensor:
        """Perform forward pass and calculate loss."""
        x, meta, y = batch
        y_hat = self(x, meta).squeeze(1)
        acc = self.acc(y_hat, y)
        iou = self.iou(y_hat, y)
        loss = binary_cross_entropy_with_logits(y_hat, y)
        dice = None#self.dice(y_hat, y)
        precision = self.precision(y_hat, y)
        recall = self.recall(y_hat, y)
        return loss, acc, iou, dice, precision, recall

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step."""
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        if batch_idx % self.accumulate_grad_batches == 0:
            total_batches = self.trainer.num_training_batches
            scheduler.step((batch_idx / total_batches) + self.current_epoch)

        loss, acc, iou = self._step(batch, batch_idx)

        self.log(f"train_acc", acc, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log(f"train_iou", iou, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log(f"train_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        loss /= self.accumulate_grad_batches
        self.manual_backward(loss)

        # accumulation
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            optimizer.step()
            optimizer.zero_grad()

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        loss, acc, iou, dice, precision, recall = self._step_eval(batch, batch_idx)
        self.log(f"val_acc", acc, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log(f"val_iou", iou, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log(f"val_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        #self.log(f"val_dice", dice, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.log(f"val_precision", precision, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.log(f"val_recall", recall, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Test step."""
        loss, acc, iou, dice, precision, recall = self._step_eval(batch, batch_idx)
        self.log(f"test_acc", acc, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log(f"test_iou", iou, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log(f"test_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        #self.log(f"test_dice", dice, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.log(f"test_precision", precision, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        self.log(f"test_recall", recall, on_epoch=True, on_step=False, prog_bar=False, logger=True)
        return loss

    def configure_optimizers(self) -> tuple:
        groups = param_groups_lrd(
            self.aegis,
            len(self.aegis.encoder.encoder.encoder),
            self.lr,
            self.weight_decay,
            self.layer_decay
        )

        optimizer = AdamW(groups, lr=self.lr)

        scheduler = CosineStepDecay(optimizer,
                                   base_lr=self.lr,
                                   min_lr=self.min_lr,
                                   warmup=self.warmup,
                                   epochs=self.epochs)

        return [optimizer], [scheduler]
