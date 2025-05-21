INPUT_SCHEMA = {
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["A cat holding a sign that says hello world"]
    },
    "negative_prompt": {
        'datatype': 'STRING',
        'required': False,
        'shape': [1],
        'example': ["text, watermark, signature, cartoon, anime, illustration, painting, drawing, low quality, blurry"]
    },
    "height": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [512]
    },
    "width": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [512]
    },
    "num_inference_steps": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [28]
    },
    "guidance_scale": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [7.0]
    }
}
