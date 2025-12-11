import gradio as gr
import numpy as np
import random
import spaces
import torch
from diffusers import ZImagePipeline

import os
import requests
import tempfile
import shutil
from urllib.parse import urlparse

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load the model pipeline
pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=dtype).to(device)

torch.cuda.empty_cache()

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

# pipe.flux_pipe_call_that_returns_an_iterable_of_images = flux_pipe_call_that_returns_an_iterable_of_images.__get__(pipe)

# flymy-ai/qwen-image-realism-lora

def load_lora_auto(pipe, lora_input):
    lora_input = lora_input.strip()
    if not lora_input:
        return

    # If it's just an ID like "author/model"
    if "/" in lora_input and not lora_input.startswith("http"):
        pipe.load_lora_weights(lora_input)
        return

    if lora_input.startswith("http"):
        url = lora_input

        # Repo page (no blob/resolve)
        if "huggingface.co" in url and "/blob/" not in url and "/resolve/" not in url:
            repo_id = urlparse(url).path.strip("/")
            pipe.load_lora_weights(repo_id)
            return

        # Blob link → convert to resolve link
        if "/blob/" in url:
            url = url.replace("/blob/", "/resolve/")

        # Download direct file
        tmp_dir = tempfile.mkdtemp()
        local_path = os.path.join(tmp_dir, os.path.basename(urlparse(url).path))

        try:
            print(f"Downloading LoRA from {url}...")
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved LoRA to {local_path}")
            pipe.load_lora_weights(local_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

@spaces.GPU()
def infer(prompt, seed=42, randomize_seed=False, width=1024, height=1024, guidance_scale=4, num_inference_steps=28, lora_id=None, lora_scale=0.95, progress=gr.Progress(track_tqdm=True)):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)

    
    if lora_id and lora_id.strip() != "":
        pipe.unload_lora_weights()
        load_lora_auto(pipe, lora_id)
    
    try:
        image = pipe(
        prompt=prompt,
        negative_prompt="",
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        generator=generator,
        guidance_scale=guidance_scale  # Use a fixed default for distilled guidance
    ).images[0]
        print("Image Generation Completed for: ", prompt, lora_id)
        return image, seed
    finally:
        # Unload LoRA weights if they were loaded
        if lora_id:
            pipe.unload_lora_weights()
    
examples = [
    "a tiny astronaut hatching from an egg on the moon",
    "a cat holding a sign that says hello world",
    "an anime illustration of a wiener schnitzel",
]
    
css = """
#col-container {
   margin: 0 auto;
   max-width: 960px;
}
.generate-btn {
   background: linear-gradient(90deg, #4B79A1 0%, #283E51 100%) !important;
   border: none !important;
   color: white !important;
}
.generate-btn:hover {
   transform: translateY(-2px);
   box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}
"""

with gr.Blocks(css=css) as app:
    gr.HTML("<center><h1>Qwen Image with LoRA support</h1></center>")
    with gr.Column(elem_id="col-container"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text_prompt = gr.Textbox(label="Prompt", placeholder="Enter a prompt here", lines=3, elem_id="prompt-text-input")
                with gr.Row():
                    custom_lora = gr.Textbox(label="Custom LoRA (optional)", info="URL or the path to the LoRA weights", placeholder="kudzueye/boreal-qwen-image")
                with gr.Row():
                    with gr.Accordion("Advanced Settings", open=False):
                        lora_scale = gr.Slider(
                            label="LoRA Scale",
                            minimum=0,
                            maximum=2,
                            step=0.01,
                            value=1,
                        )
                        with gr.Row():
                            width = gr.Slider(label="Width", value=1024, minimum=64, maximum=2048, step=16)
                            height = gr.Slider(label="Height", value=1024, minimum=64, maximum=2048, step=16)
                        seed = gr.Slider(label="Seed", value=-1, minimum=-1, maximum=4294967296, step=1)
                        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                        with gr.Row():
                            steps = gr.Slider(label="Inference steps steps", value=9, minimum=1, maximum=20, step=1)
                            cfg = gr.Slider(label="Guidance Scale", value=0, minimum=0, maximum=20, step=0.5)
                        # method = gr.Radio(label="Sampling method", value="DPM++ 2M Karras", choices=["DPM++ 2M Karras", "DPM++ SDE Karras", "Euler", "Euler a", "Heun", "DDIM"])

                with gr.Row():
                    # text_button = gr.Button("Run", variant='primary', elem_id="gen-button")
                    text_button = gr.Button("✨ Generate Image", variant='primary', elem_classes=["generate-btn"])
            with gr.Column():
                with gr.Row():
                    image_output = gr.Image(type="pil", label="Image Output", elem_id="gallery")
        
        # gr.Markdown(article_text)
        with gr.Column():
            gr.Examples(
                examples = examples,
                inputs = [text_prompt],
            )
    gr.on(
        triggers=[text_button.click, text_prompt.submit],
        fn = infer,
        inputs=[text_prompt, seed, randomize_seed, width, height, cfg, steps, custom_lora, lora_scale], 
        outputs=[image_output, seed]
    )
        
        # text_button.click(query, inputs=[custom_lora, text_prompt, steps, cfg, randomize_seed, seed, width, height], outputs=[image_output,seed_output, seed])
        # text_button.click(infer, inputs=[text_prompt, seed, randomize_seed, width, height, cfg, steps, custom_lora, lora_scale], outputs=[image_output,seed_output, seed])

app.launch(share=True)