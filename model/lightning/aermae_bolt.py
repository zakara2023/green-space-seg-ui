"""
aermae_bolt.py

The lightning AerMae wrapper module.
"""
import torch
from torch.optim import AdamW
from lightning import LightningModule

from model.aermae import AerMAE, mae_loss
from model.scheduler import CosineStepDecay
from model.transforms import patch_images, normalize_patches


class AerMAEBolt(LightningModule):
    """The lightning AerMae wrapper module."""

    def __init__(self,
                 mae: AerMAE,
                 warmup: int,
                 epochs: int,
                 lr: float=1e-5,
                 min_lr: float=0.0,
                 weight_decay: float=0.05,
                 unmasked_weight: float=0.,
                 norm_tgt: bool=True,
                 accumulate_grad_batches: int=4):
        super().__init__()
        self.mae = mae
        self.warmup = warmup
        self.epochs = epochs
        self.min_lr = min_lr
        self.lr = lr
        self.weight_decay = weight_decay
        self.norm_tgt = norm_tgt
        self.unmasked_weight = unmasked_weight
        self.accumulate_grad_batches = accumulate_grad_batches
        self.automatic_optimization = False

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        return self.mae(x, meta)

    def _step(self, batch: torch.Tensor, _: int) -> torch.Tensor:
        """Perform forward pass and calculate loss."""
        x, meta = batch
        y = patch_images(x, self.mae.patch_size)
        if self.norm_tgt == True:
            y = normalize_patches(y)

        y_hat, mask = self(x, meta)

        loss = mae_loss(y_true=y,
                        y_pred=y_hat,
                        mask=mask,
                        weight=self.unmasked_weight)

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step."""
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        if batch_idx % self.accumulate_grad_batches == 0:
            total_batches = self.trainer.num_training_batches
            scheduler.step((batch_idx / total_batches) + self.current_epoch)

        loss = self._step(batch, batch_idx)

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
        loss = self._step(batch, batch_idx)
        self.log(f"val_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Test step."""
        loss = self._step(batch, batch_idx)
        self.log(f"test_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self) -> tuple:
        optimizer = AdamW(self.mae.parameters(),
                        lr=self.lr,
                        weight_decay=self.weight_decay,
                        betas=(0.9, 0.95))

        scheduler = CosineStepDecay(optimizer,
                                   base_lr=self.lr,
                                   min_lr=self.min_lr,
                                   warmup=self.warmup,
                                   epochs=self.epochs)

        return [optimizer], [scheduler]
