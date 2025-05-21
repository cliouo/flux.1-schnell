import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
from diffusers import FluxPipeline
import torch
from io import BytesIO
import base64

class InferlessPythonModel:
    def initialize(self):
        model_id = "black-forest-labs/FLUX.1-dev"
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
        self.pipe.enable_model_cpu_offload()

    def infer(self, inputs, seed = None):
        prompt = inputs["prompt"]
        height = inputs.get("height", 1024)
        width = inputs.get("width", 1024)
        guidance_scale = inputs.get("guidance_scale", 3.5)
        num_inference_steps = inputs.get("num_inference_steps", 50)
        max_sequence_length = inputs.get("max_sequence_length", 512)

        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)

        image = self.pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=generator,
        ).images[0]

        buff = BytesIO()
        image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode()

        return {"generated_image_base64": img_str}

    def finalize(self):
        self.pipe = None
