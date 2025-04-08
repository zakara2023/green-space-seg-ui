import gradio as gr
import torch
import numpy as np
import os
from torchvision import transforms
from PIL import Image

from aegis import AeGISUpernet
from aermae import AerMAE
from lightning.aermae_bolt import AerMAEBolt
from lightning.aegis_bolt import AeGISBolt

# Load model checkpoint
print(os.getcwd())
ckpt = torch.load("checkpoint/aegis.ckpt", map_location="cpu")

# Rebuild model
mae = AerMAE(img_size=(224, 224), patch_size=8, enc_meta_dim=128, enc_dim=768, dec_meta_dim=128, dec_dim=512,
             enc_layers=6, dec_layers=4, enc_heads=16, dec_heads=16, ff_mul=4, mask_pct=0.75)

l_mae = AerMAEBolt(mae, lr=1e-4, warmup=5, epochs=20, accumulate_grad_batches=1)
aegis = AeGISUpernet(768, l_mae.mae.encoder, inindex=[1, 3, 4, 5])
l_aeg = AeGISBolt(aegis, lr=1e-4, warmup=5, epochs=20, accumulate_grad_batches=1)

l_aeg.load_state_dict(ckpt["state_dict"], strict=False)
l_aeg.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Inference function
def segment_image(image, lat, lon, month, year):
    img_tensor = transform(image).unsqueeze(0)
    meta_tensor = torch.tensor([[lat, lon, month, year]], dtype=torch.float32)

    with torch.no_grad():
        output = l_aeg.model(img_tensor, meta_tensor)
        mask = torch.sigmoid(output).squeeze().numpy()
        return (mask > 0.5).astype(np.uint8) * 255

# Gradio interface
demo = gr.Interface(
    fn=segment_image,
    inputs=[
        gr.Image(type="pil"),
        gr.Number(label="Latitude"),
        gr.Number(label="Longitude"),
        gr.Number(label="Month"),
        gr.Number(label="Year")
    ],
    outputs=gr.Image(label="Segmentation Mask"),
    title="Green Space Segmentation with AeGIS",
    description="Upload an aerial image and provide location/time to get predicted green space."
)

if __name__ == "__main__":
    demo.launch()
