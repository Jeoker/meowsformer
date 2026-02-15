"""Meowsformer DSP Engine — audio processing, prosody transformation & description."""

from src.engine.dsp_processor import (
    BREED_F0_BASELINES,
    INTENT_VA_MAP,
    SampleMatch,
    VAPoint,
    apply_prosody_transform,
    get_all_intents,
    get_best_match,
    get_breed_f0,
    map_intent_to_va,
    synthesize_meow,
)
from src.engine.description_generator import (
    PreviewDescription,
    generate_description_from_synthesis,
    generate_preview_description,
)

__all__ = [
    # DSP processor
    "BREED_F0_BASELINES",
    "INTENT_VA_MAP",
    "SampleMatch",
    "VAPoint",
    "apply_prosody_transform",
    "get_all_intents",
    "get_best_match",
    "get_breed_f0",
    "map_intent_to_va",
    "synthesize_meow",
    # Description generator
    "PreviewDescription",
    "generate_description_from_synthesis",
    "generate_preview_description",
]
