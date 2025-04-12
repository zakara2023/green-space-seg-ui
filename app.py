import os
import torch
import numpy as np
import gradio as gr
from PIL import Image
import pytorch_lightning as pl
from torchvision import transforms
import json
import sys
sys.path.insert(1, '../')
import model.transforms as T
from model.aermae import AerMAE
from model.aegis import AeGISUpernet, AeGISFormer
from model.lightning.aermae_bolt import AerMAEBolt
from model.lightning.aegis_bolt import AeGISBolt
from matplotlib import pyplot as plt
lr = 0.005
epochs = 10
accum = 4
mae = AerMAE(img_size=(224, 224),
             patch_size=8,
             enc_meta_dim=128,
             enc_dim=768,
             dec_meta_dim=128,
             dec_dim=512,
             enc_layers=6,
             dec_layers=4,
             enc_heads=16,
             dec_heads=16,
             ff_mul=4,
             mask_pct=0.75)
l_mae = AerMAEBolt(mae, lr=lr, warmup=5, epochs=epochs, accumulate_grad_batches=accum, norm_tgt=False, unmasked_weight=0.05)
aegis = AeGISUpernet(768, l_mae.mae.encoder, inindex=[1, 3, 4, 5])
l_aeg = AeGISBolt(aegis, lr=lr, warmup=5, epochs=epochs, accumulate_grad_batches=accum)
l_aeg = AeGISBolt(aegis, lr=lr, warmup=5, epochs=epochs, accumulate_grad_batches=accum)
l_aeg.load_state_dict(torch.load('checkpoint/aegis-exp13-epoch=15-val_loss_epoch=0.23.ckpt', weights_only=True)['state_dict'])
