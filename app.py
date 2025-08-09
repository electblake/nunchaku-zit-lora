import gradio as gr
import numpy as np
import random
import spaces
import torch
from diffusers import  QwenImagePipeline

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"



# Load the model pipeline
pipe = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=dtype).to(device)

torch.cuda.empty_cache()

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

# pipe.flux_pipe_call_that_returns_an_iterable_of_images = flux_pipe_call_that_returns_an_iterable_of_images.__get__(pipe)

@spaces.GPU()
def infer(prompt, seed=42, randomize_seed=False, width=1024, height=1024, guidance_scale=4, num_inference_steps=28, lora_id=None, lora_scale=0.95, progress=gr.Progress(track_tqdm=True)):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)

    
    if lora_id and lora_id.strip() != "":
        pipe.unload_lora_weights()
        pipe.load_lora_weights(lora_id.strip())

    apply_cache_on_pipe(
       pipe,
   )
    
    try:
        image = pipe(
        prompt=prompt,
        negative_prompt="",
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        generator=generator,
        true_cfg_scale=guidance_scale,
        guidance_scale=1.0  # Use a fixed default for distilled guidance
    ).images[0]
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
                    custom_lora = gr.Textbox(label="Custom LoRA (optional)", info="LoRA Hugging Face path", placeholder="flymy-ai/qwen-image-realism-lora")
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
                            steps = gr.Slider(label="Inference steps steps", value=28, minimum=1, maximum=100, step=1)
                            cfg = gr.Slider(label="Guidance Scale", value=4, minimum=1, maximum=20, step=0.5)
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