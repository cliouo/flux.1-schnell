import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
from diffusers import AutoPipelineForText2Image # 修改: 导入 AutoPipelineForText2Image
import torch
from io import BytesIO
import base64
from peft import PeftModel, PeftConfig # 新增: 导入 PeftModel, PeftConfig

class InferlessPythonModel:
    def initialize(self):
        model_id = "black-forest-labs/FLUX.1-dev"
        # snapshot_download 步骤保留
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        # 修改: 使用 AutoPipelineForText2Image 并更改 torch_dtype
        self.pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
        # 新增: 加载 LoRA 权重
        self.pipe.load_lora_weights('Heartsync/Flux-NSFW-uncensored', weight_name='lora.safetensors', adapter_name="uncensored")
        # 移除: self.pipe.enable_model_cpu_offload()

    def infer(self, inputs, seed = None):
        prompt = inputs["prompt"]
        height = inputs.get("height", 1024)
        width = inputs.get("width", 1024)
        # 修改: guidance_scale 默认值
        guidance_scale = inputs.get("guidance_scale", 7.0)
        # 修改: num_inference_steps 默认值
        num_inference_steps = inputs.get("num_inference_steps", 28)
        # 新增: 获取 negative_prompt
        negative_prompt = inputs.get("negative_prompt", "text, watermark, signature, cartoon, anime, illustration, painting, drawing, low quality, blurry")
        # 移除: max_sequence_length

        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)

        image = self.pipe(
            prompt,
            negative_prompt=negative_prompt, # 新增: negative_prompt 参数
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            # 移除: max_sequence_length 参数
            generator=generator,
        ).images[0]

        buff = BytesIO()
        image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode()

        return {"generated_image_base64": img_str}

    def finalize(self):
        self.pipe = None
