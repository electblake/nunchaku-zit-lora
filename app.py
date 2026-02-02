"""
Consolidated Gradio app for Z-Image-Turbo LoRA nodes.
This replaces graph_primitives.py and test_nodes.py.
"""
from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime
from typing import ClassVar, Generic, TypeVar, TypedDict, cast
from urllib.parse import urlparse

import gradio as gr
import requests
import torch
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
from nunchaku import NunchakuZImageTransformer2DModel
from nunchaku.utils import get_precision, is_turing
from PIL import Image, PngImagePlugin

BASE_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
CACHE_DIR = "loras"
PROMPT_TRIGGER = "c0l0ringb00k"
MAX_SEED = 2_147_483_647

SlotSpec = tuple[str, str]
SlotSpecs = tuple[SlotSpec, ...]
TOutput = TypeVar("TOutput", bound=tuple[object, ...])


class Slot:
    """
    Single input or output slot with a string type.
    Slots are ordered and accessed by index.
    """

    def __init__(self, name: str, slot_type: str, index: int):
        self.name = name  # for debugging only
        self.type = slot_type  # STRING, IMAGE, INT, FLOAT, etc.
        self.index = index  # ordinal position

    def __repr__(self) -> str:
        return f"Slot[{self.index}]({self.name}: {self.type})"


class Node(Generic[TOutput]):
    """
    Base node class following ComfyUI pattern.
    Nodes have ordered input/output slots and execute work.
    All returns are tuples in slot order.
    """

    INPUT_TYPES: ClassVar[SlotSpecs] = ()
    OUTPUT_TYPES: ClassVar[SlotSpecs] = ()

    def __init__(self):
        self.input_slots = tuple(
            Slot(name, slot_type, i)
            for i, (name, slot_type) in enumerate(self.INPUT_TYPES)
        )
        self.output_slots = tuple(
            Slot(name, slot_type, i)
            for i, (name, slot_type) in enumerate(self.OUTPUT_TYPES)
        )

    def execute(self, *inputs: object) -> TOutput:
        """Override in subclasses. Must return tuple matching OUTPUT_TYPES length."""
        raise NotImplementedError("Subclass must implement execute()")

    def __call__(self, *inputs: object) -> TOutput:
        if len(inputs) != len(self.input_slots):
            raise ValueError(
                f"Expected {len(self.input_slots)} inputs, got {len(inputs)}"
            )
        return self.execute(*inputs)

    @classmethod
    def get_input_types(cls) -> tuple[str, ...]:
        return tuple(slot_type for _, slot_type in cls.INPUT_TYPES)

    @classmethod
    def get_output_types(cls) -> tuple[str, ...]:
        return tuple(slot_type for _, slot_type in cls.OUTPUT_TYPES)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in={len(self.input_slots)}, out={len(self.output_slots)})"


def _normalize_lora_url(url: str) -> str:
    if "/blob/" in url:
        return url.replace("/blob/", "/resolve/")
    return url


def _download_file(url: str, dst_path: str) -> None:
    print(f"Downloading LoRA from {url}...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dst_path, "wb") as handle:
        for chunk in resp.iter_content(chunk_size=8192):
            handle.write(chunk)
    print(f"Saved LoRA to {dst_path}")


def _ensure_lora_cached(lora_url: str, cache_dir: str) -> str:
    url = _normalize_lora_url(lora_url)
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(urlparse(url).path)
    if not filename:
        raise ValueError(f"Could not derive filename from LoRA URL: {lora_url}")
    local_path = os.path.join(cache_dir, filename)
    if not os.path.exists(local_path):
        _download_file(url, local_path)
    else:
        print(f"Using cached LoRA: {local_path}")
    return local_path


def _load_lora_adapter(pipe: ZImagePipeline, lora_url: str, adapter_name: str) -> str:
    local_path = _ensure_lora_cached(lora_url, CACHE_DIR)
    if getattr(pipe, "_loaded_lora_path", None) == local_path:
        print("LoRA already loaded, skipping")
        return local_path

    print(f"Loading LoRA into {type(pipe.transformer).__name__}")
    pipe.transformer.load_lora_adapter(local_path, adapter_name=adapter_name)
    pipe.transformer.enable_lora()
    pipe._loaded_lora_path = local_path
    pipe._loaded_lora_adapter = adapter_name
    print(f"✓ LoRA adapter '{adapter_name}' loaded and enabled")
    return local_path


def _extract_image(result: object) -> Image.Image:
    images = getattr(result, "images", None)
    if images is None:
        images = result[0] if isinstance(result, tuple) else result
    if isinstance(images, (list, tuple)):
        image = images[0]
    else:
        image = images
    return cast(Image.Image, image)


class LoRAGeneratorNode(Node[tuple[Image.Image, int]]):
    """
    Generate images using Z-Image-Turbo with a specific LoRA.
    Inputs: (prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps)
    Outputs: (image, seed)
    """

    INPUT_TYPES = (
        ("prompt", "STRING"),
        ("seed", "INT"),
        ("randomize_seed", "BOOLEAN"),
        ("width", "INT"),
        ("height", "INT"),
        ("guidance_scale", "FLOAT"),
        ("num_inference_steps", "INT"),
    )

    OUTPUT_TYPES = (
        ("image", "IMAGE"),
        ("seed", "INT"),
    )

    def __init__(self, pipe: ZImagePipeline, name: str, lora_url: str, adapter_name: str):
        super().__init__()
        self.pipe = pipe
        self.name = name
        self.lora_url = lora_url
        self.adapter_name = adapter_name

    def execute(self, *inputs: object) -> tuple[Image.Image, int]:
        if len(inputs) != len(self.input_slots):
            raise ValueError(
                f"Expected {len(self.input_slots)} inputs, got {len(inputs)}"
            )
        (
            prompt,
            seed,
            randomize_seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
        ) = inputs
        prompt = cast(str, prompt)
        seed = cast(int, seed)
        randomize_seed = cast(bool, randomize_seed)
        width = cast(int, width)
        height = cast(int, height)
        guidance_scale = cast(float, guidance_scale)
        num_inference_steps = cast(int, num_inference_steps)
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        generator = torch.Generator().manual_seed(seed)

        if self.lora_url:
            _load_lora_adapter(self.pipe, self.lora_url, self.adapter_name)

        prompt = f"{PROMPT_TRIGGER} {prompt}"
        exectn = self.pipe(
            prompt=prompt,
            negative_prompt="",
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
        )
        image = _extract_image(exectn)
        print(f"{self.name} Image Generated: {prompt}")
        return (image, seed)


print("Loading Nunchaku 4-bit Z-Image-Turbo...")
precision = get_precision()
rank = 128
dtype = torch.float16 if is_turing() else torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

print(
    f"Loading quantized transformer: nunchaku-tech/nunchaku-z-image-turbo/"
    f"svdq-{precision}_r{rank}-z-image-turbo.safetensors"
)
transformer = NunchakuZImageTransformer2DModel.from_pretrained(
    f"nunchaku-tech/nunchaku-z-image-turbo/svdq-{precision}_r{rank}-z-image-turbo.safetensors",
    torch_dtype=dtype,
)
print(f"✓ Transformer loaded: {type(transformer).__name__}")
print(f"✓ Transformer device: {next(transformer.parameters()).device}")
print(f"✓ Transformer dtype: {next(transformer.parameters()).dtype}")

print("Loading Z-Image-Turbo pipeline components...")
pipe = ZImagePipeline.from_pretrained(
    BASE_MODEL_ID,
    transformer=transformer,
    torch_dtype=dtype,
    low_cpu_mem_usage=False,
).to(device)
print(f"✓ Pipeline loaded on {device}")
print(f"✓ Transformer type: {type(pipe.transformer).__name__}")
print("Nunchaku 4-bit model loaded!")

class LoRASpec(TypedDict):
    name: str
    url: str
    adapter: str


LORA_SPECS: list[LoRASpec] = [
    {
        "name": "Renderartist",
        "url": "https://huggingface.co/renderartist/Coloring-Book-Z-Image-Turbo-LoRA/resolve/main/Coloring_Book_Z_Image_Turbo_v1_renderartist_2000.safetensors",
        "adapter": "renderartist",
    },
    {
        "name": "Playtime AI",
        "url": "PLACEHOLDER_URL_FOR_CIVITAI_MODEL_2184429",
        "adapter": "playtime_ai",
    },
    {
        "name": "Foo Samaritan",
        "url": "PLACEHOLDER_URL_FOR_CIVITAI_MODEL_2249185",
        "adapter": "foo_samaritan",
    },
]

NODES: dict[str, LoRAGeneratorNode] = {
    spec["name"]: LoRAGeneratorNode(
        pipe=pipe,
        name=spec["name"],
        lora_url=spec["url"],
        adapter_name=spec["adapter"],
    )
    for spec in LORA_SPECS
}

EXAMPLE_PROMPTS = [
    "3/4 angle view slight turn coloring page vector lines  a small laptop with the words \"Coloring Book Z\" heading  with subheading \"for Z-Image Turbo\" on the screen written in a playful rounded font, white bezel - a man in a white suit is standing in the background holding the left and right hand of the screen  classic style, all white background, contrast black and white, perfect linework coloring page",
    "Greek mythology mermaid body, bikini top, with wavy hair 3/4 angle, classic style, white background coloring page,  black and white, thin thickness lines playful design",
    "king with beard and mustache 3/4 angle, classic style, white background coloring page,  black and white, thin thickness lines playful design",
    "warrior soldier with sword battle stance, 3/4 angle, classic style, white background coloring page,  black and white, thin thickness lines playful design",
    "A wizard casting spells with a staff in a forest. Coloring book page, black-and-white, simple design, easy to color. Plain white background.",
    "3/4 angle view slight turn  santa is giving a wrapped gift with a bow to rudolph the reindeer igloo in the background  classic style, white background coloring page, contrast black and white, perfect linework",
    "3/4 angle view slight turn coloring page vector lines  a young man on a skateboard, cute face, white hair  classic style, all white background, contrast black and white, perfect linework coloring page",
    "tv set 3/4 angle, classic style, white background coloring page, contrast black and white, thin thickness lines playful design",
    "castle 3/4 angle, classic style, white background coloring page, contrast black and white, thin thickness lines playful design",
    "3/4 angle view slight turn coloring page vector lines  a detailed starfish with a face  classic style, all white background, contrast black and white, perfect linework coloring page",
    "lion, classic style, white background coloring page, contrast black and white, thin thickness lines playful design",
    "3/4 angle view slight turn cute bear with a  pot of honey the pot has the label \"HONEY\", classic style, white background coloring page, contrast black and white, thin lines",
    "3/4 angle view slight turn coloring page vector lines  a dog standing on his hind legs, action pose on a skateboard, cute face, white hair  classic style, all white background, contrast black and white, perfect linework coloring page",
    "3/4 angle view slight turn coloring page vector lines  a chihuahua wearing a sombrero  classic style, all white background, contrast black and white, perfect linework coloring page",
    "3/4 angle view slight turn coloring page vector for children  a white monkey swings from a tree branch tropical forest  classic style, all white background, contrast black and white, perfect linework coloring page",
    "duckling 3/4 angle, classic style, white background coloring page, contrast black and white, thin thickness lines playful design",
    "werewolf 3/4 angle, classic style, white background coloring page, contrast black and white, thin thickness lines playful design",
    "island isolated 3/4 angle, classic style, white background coloring page, contrast black and white, thin thickness lines playful design",
    "shark in fornt of coral 3/4 angle, classic style, white background coloring page, contrast black and white, thin thickness lines playful design",
    "octopus in fornt of coral 3/4 angle, classic style, white background coloring page, contrast black and white, thin thickness lines playful design",
]


def load_example(example_choice: str) -> str:
    if example_choice == "Custom":
        return ""
    idx = int(example_choice.split(" ")[1]) - 1
    return EXAMPLE_PROMPTS[idx]


def generate_image(
    node_name: str,
    prompt: str,
    seed: int,
    randomize_seed: bool,
    width: int,
    height: int,
    guidance_scale: float,
    steps: int,
) -> tuple[Image.Image, str]:
    node = NODES[node_name]
    start_time = time.time()
    image, final_seed = node(
        prompt,
        seed,
        randomize_seed,
        width,
        height,
        guidance_scale,
        steps,
    )
    generation_time = time.time() - start_time

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{node_name}_{final_seed}_{generation_time:.1f}s"
    image_path = os.path.join(output_dir, f"{filename}.png")
    json_path = os.path.join(output_dir, f"{filename}.json")

    metadata = {
        "node": node_name,
        "prompt": prompt,
        "prompt_with_trigger": f"{PROMPT_TRIGGER} {prompt}",
        "seed": final_seed,
        "randomize_seed": randomize_seed,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": steps,
        "generation_time_seconds": round(generation_time, 2),
        "timestamp": timestamp,
    }

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("generation_params", json.dumps(metadata))
    image.save(image_path, "PNG", pnginfo=pnginfo)

    with open(json_path, "w") as handle:
        json.dump(metadata, handle, indent=2)

    return image, f"Seed: {final_seed}\nSaved: {filename}"


with gr.Blocks() as demo:
    gr.Markdown("# Coloring Book Generator")
    gr.Markdown("Test the generator nodes with different LoRAs")

    with gr.Row():
        with gr.Column():
            node_selector = gr.Dropdown(
                choices=list(NODES.keys()),
                value="Renderartist",
                label="Select Generator Node",
            )

            example_selector = gr.Dropdown(
                choices=["Custom"] + [f"Example {i + 1}" for i in range(len(EXAMPLE_PROMPTS))],
                value="Custom",
                label="Example Prompts",
            )

            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="a cat sitting on a chair",
                lines=3,
            )

            example_selector.change(
                fn=load_example,
                inputs=[example_selector],
                outputs=[prompt_input],
            )

            with gr.Accordion("Settings", open=True):
                seed_input = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    value=42,
                    step=1,
                )
                randomize_input = gr.Checkbox(
                    label="Randomize Seed",
                    value=False,
                )
                width_input = gr.Slider(
                    label="Width",
                    minimum=512,
                    maximum=2048,
                    value=1024,
                    step=64,
                )
                height_input = gr.Slider(
                    label="Height",
                    minimum=512,
                    maximum=2048,
                    value=1024,
                    step=64,
                )
                guidance_input = gr.Slider(
                    label="Guidance Scale",
                    minimum=0,
                    maximum=20,
                    value=0,
                    step=0.5,
                )
                steps_input = gr.Slider(
                    label="Inference Steps",
                    minimum=1,
                    maximum=50,
                    value=8,
                    step=1,
                )

            generate_btn = gr.Button("Generate Image", variant="primary")

        with gr.Column():
            image_output = gr.Image(label="Generated Image")
            seed_output = gr.Textbox(label="Used Seed", interactive=False)

    generate_btn.click(
        fn=generate_image,
        inputs=[
            node_selector,
            prompt_input,
            seed_input,
            randomize_input,
            width_input,
            height_input,
            guidance_input,
            steps_input,
        ],
        outputs=[image_output, seed_output],
    )

if __name__ == "__main__":
    demo.launch(share=False)
