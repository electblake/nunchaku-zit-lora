Project Specification: Z-Image-LORA
Category: Implementation Specification
Status: Draft

Abstract

   Specification for Z-Image-LORA project implementing LoRA-based
   inference using Z-Image-Turbo base model with Gradio interface.

Conformance

   The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT",
   "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this
   document are to be interpreted as described in RFC 2119.

1. Technology Stack

   Base Model: Tongyi-MAI/Z-Image-Turbo (REQUIRED)
   Pipeline Framework: HuggingFace Diffusers (REQUIRED)
   Interface: Gradio with MCP server support (REQUIRED)
   Compute: CUDA when available, CPU fallback (REQUIRED)

2. Scope

2.1 In Scope

   - LoRA weight loading and inference
   - Gradio GUI for inference
   - MCP API for inference
   - Per-LoRA configuration management
   - Prompt library per configuration

2.2 Out of Scope

   - Multiple base model support
   - Runtime model compatibility verification
   - Base model selection by end users
   - Custom LoRA training

3. Configuration Objects

   Configuration objects MUST contain:
   - LoRA resource reference (REQUIRED)
   - Base model configuration overrides (OPTIONAL)
   - LoRA-specific parameters (REQUIRED)
   - Supported prompts list (OPTIONAL)

   Configurations MUST be base-model-specific.
   Configuration compatibility MUST NOT be runtime-validated.

4. Resource References

   LoRA resources MUST support HuggingFace repository IDs and URLs.
   Configuration files MUST support local file paths.
   Prompt files MUST support local file paths.

5. Design Constraints

   Single base model per project instance (REQUIRED)
   No custom features extending diffusers without approval (REQUIRED)
   Use diffusers primitives as specified (REQUIRED)
   Gradio built-in MCP support (REQUIRED)

6. Future Considerations

   - Configuration-based use cases (e.g., "coloring books")
   - Generic foundation library for cross-cutting concerns
   - Configuration persistence and sharing
