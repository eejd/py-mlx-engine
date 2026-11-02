"""
Model Kit module with automatic compatibility patches.

This module automatically applies compatibility patches for various model types
by replacing classes in their respective modules with derived, compatible versions.
"""

from .patches.gemma3n import apply_patches as _apply_patches_gemma3n
from .patches.qwen3_5 import apply_patches as _apply_patches_qwen3_5

_apply_patches_gemma3n()
_apply_patches_qwen3_5()
