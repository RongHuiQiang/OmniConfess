"""omniconfess.backbone -- Qwen2.5-Omni inference backbone.

Self-contained data loading, model I/O, prompt construction and result
serialisation for the confession pipeline. Vendored into the package so
``omniconfess`` imports cleanly with no ``sys.path`` manipulation.
"""
from .data import load_dataset, load_media_for_item, resolve_media_path
from .model import (
    load_model, build_inputs, safe_generate, decode_outputs,
    build_user_content_with_media_tokens, make_messages, prepare_for_thinker,
)
from .prompts import get_system_prompt, get_user_prompt
from .evalio import parse_pred, save_result, make_result

__all__ = [
    'load_dataset', 'load_media_for_item', 'resolve_media_path',
    'load_model', 'build_inputs', 'safe_generate', 'decode_outputs',
    'build_user_content_with_media_tokens', 'make_messages', 'prepare_for_thinker',
    'get_system_prompt', 'get_user_prompt',
    'parse_pred', 'save_result', 'make_result',
]
