# Coloring Book Nodes Skill

Quick reference for working with the Z-Image coloring book generator nodes.

## Starting the Test Server

```bash
# Activate virtual environment
.venv\Scripts\activate

# Run the test GUI
python test_nodes.py
```

The Gradio interface will start at http://127.0.0.1:7860

## Available Generator Nodes

Three generator nodes, each with different LoRA styles:

1. **RenderartistGeneratorNode** - Coloring Book Z style (working)
2. **PlaytimeAiGeneratorNode** - Coloring Book Style (placeholder URL)
3. **FooSamaritanGeneratorNode** - Coloring Drawing style (placeholder URL)

## Node Architecture

All nodes follow the ComfyUI-inspired pattern:
- **Inputs**: Ordered slots (prompt, seed, dimensions, etc.)
- **Outputs**: Ordered tuple (image, seed)
- **Execution**: Self-contained with vendored logic

## Quick Test

```python
from graph_primitives import RenderartistGeneratorNode
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline

# Load pipeline
pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo").to("cuda")

# Create node
node = RenderartistGeneratorNode(pipe)

# Generate
image, seed = node.execute(
    prompt="a cat sitting on a chair",
    seed=42,
    randomize_seed=False,
    width=1024,
    height=1024,
    guidance_scale=4,
    num_inference_steps=28
)
```

## More Information

See `references/` folder for detailed documentation on:
- Node system architecture
- Adding new LoRA models
- Troubleshooting
