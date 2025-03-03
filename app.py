import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
from diffusers import FluxPipeline
import torch
from io import BytesIO
import base64

class InferlessPythonModel:
    def initialize(self):
        model_id = "black-forest-labs/FLUX.1-schnell"
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")

    def infer(self, inputs):
        prompt = inputs["prompt"]
        height = inputs.get("height", 512)
        width = inputs.get("width", 512)
        guidance_scale = inputs.get("guidance_scale", 7.5)
        num_inference_steps = inputs.get("num_inference_steps", 4)
        max_sequence_length = inputs.get("max_sequence_length", 256)

        image = self.pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
        ).images[0]

        buff = BytesIO()
        image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode()

        return {"generated_image_base64": img_str}

    def finalize(self):
        self.pipe = None
