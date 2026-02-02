# Design Notes

## Configuration Objects (Draft)

Future concept: Per-LoRA "use case" configurations.

**Structure:**
```
Configuration:
  - LoRA URI (HF repo, URL, file path)
  - Base model config overrides (optional)
  - LoRA parameters (weight, trigger, etc.)
  - Supported prompts (text file, one per line)
```

**Example Use Case:** "Coloring Books"
- LoRA: Coloring book style adapter
- Config: Z-Image-Turbo settings optimized for this style
- Prompts: Curated list of effective prompts for coloring books

**Constraints:**
- Configs tied to specific base model
- No runtime compatibility validation
- File names/model names not trustworthy for compatibility

## Resource Loading (Draft)

Support multiple resource reference types:
- HuggingFace repo IDs (`author/model`)
- HuggingFace URLs (`https://huggingface.co/...`)
- GitHub URLs
- Local file paths
- `file://` URIs
- Binary blobs (future)
- String blobs (future)

**Principle:** Use established library that supports most of these natively.
Don't write custom loaders if a popular package handles it.

## Future: Generic Foundation Library

Long-term goal: Extract cross-cutting concerns into reusable library.

**Scope:**
- Configuration loading/persistence
- Resource reference handling
- Deployment patterns
- Node-Slot-Link primitives

**Non-Scope:**
- Application-specific logic
- Model-specific code
- Use case definitions

This enables rapid prototyping with consistent patterns across projects.

## Gradio Integration

Gradio provides:
- GUI interface (built-in)
- MCP server (built-in, enable with `app.launch(mcp_server=True)`)
- API endpoints (built-in)

Use Gradio's built-in features. Don't build custom API layer.

## Model Stack Philosophy

**Fixed Base Model:**
- Simplifies architecture
- Allows base-model-specific optimizations
- Reduces compatibility complexity
- Each project = one base model + N LoRAs

**Benefits:**
- Focus on LoRA quality and variety
- Optimize prompts and settings per LoRA
- Simpler deployment and configuration
- Clear project boundaries
