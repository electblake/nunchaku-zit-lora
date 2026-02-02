# Z-Image-LORA Project Context

## Project Overview

Gradio-based inference application for Z-Image-Turbo model with LoRA support.

## Technology Stack

- **Base Model**: Tongyi-MAI/Z-Image-Turbo
- **Framework**: HuggingFace Diffusers
- **Interface**: Gradio (with built-in MCP server)
- **Compute**: CUDA/CPU via PyTorch

## Architecture

Implements Node-Slot-Link pattern for ordered data flow.
See [docs/PATTERN-Node-Slot-Link.md](docs/PATTERN-Node-Slot-Link.md) for pattern details.
See [docs/RFC-ARCH-001-Node-Slot-Architecture.md](docs/RFC-ARCH-001-Node-Slot-Architecture.md) for specifications.

## Design Inspirations

### ComfyUI
https://github.com/comfyanonymous/ComfyUI
- Ordered slot execution pattern
- String-based type system
- Lazy evaluation model
- Node editor graph structure

### LiteGraphJS
https://github.com/jagenjo/litegraph.js
- Visual node programming paradigm
- Slot-based connections
- Graph execution model

### HuggingFace Diffusers
https://github.com/huggingface/diffusers
- Pipeline component patterns
- Model loading primitives
- LoRA weight management

## Project Files

- `app.py`: Main Gradio application
- `graph_primitives.py`: Node/Slot base classes
- `docs/`: Specifications and design patterns

## Core Principles

1. **Ordered Data Flow**: All data accessed by index, not name
2. **Single Base Model**: Project optimized for one base model
3. **Extend, Don't Create**: Use framework primitives first
4. **Lazy Loading**: Data loaded only when needed
5. **String Types**: Type compatibility via string comparison

## Development Guidelines

- Follow diffusers patterns and practices
- Use established libraries over custom code
- Prefer popular, well-supported packages
- No custom features extending packages without approval
- Keep abstractions simple and focused
