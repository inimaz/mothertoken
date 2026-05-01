"""
Shared registry vocabulary for model metadata.
"""

from __future__ import annotations

from enum import StrEnum


class ModelType(StrEnum):
    TIKTOKEN = "tiktoken"
    HUGGINGFACE = "huggingface"
    ANTHROPIC_API = "anthropic_api"
    GOOGLE_API = "google_api"


class AccessMode(StrEnum):
    LOCAL = "local"
    API = "api"


LOCAL_MODEL_TYPES = {ModelType.TIKTOKEN.value, ModelType.HUGGINGFACE.value}
API_MODEL_TYPES = {ModelType.ANTHROPIC_API.value, ModelType.GOOGLE_API.value}
MODEL_TYPE_VALUES = [member.value for member in ModelType]
