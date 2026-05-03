"""Transformers tokenizer discovery service."""

from __future__ import annotations

import ast
import copy
import re
from collections import defaultdict
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any

MODEL_TASK_MAPPING_NAMES = {
    "text-generation": ("MODEL_FOR_CAUSAL_LM_MAPPING_NAMES",),
    "text2text-generation": ("MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES",),
    "generation": ("MODEL_FOR_CAUSAL_LM_MAPPING_NAMES", "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES"),
}


class TransformersService:
    """List and look up tokenizer classes exposed by the installed Transformers package."""

    def __init__(
        self,
        *,
        transformers_module: Any | None = None,
        tokenization_auto_module: Any | None = None,
        modeling_auto_path: Path | None = None,
    ) -> None:
        self.transformers = transformers_module or import_module("transformers")
        self.tokenization_auto = tokenization_auto_module or import_module("transformers.models.auto.tokenization_auto")
        self.modeling_auto_path = modeling_auto_path

    @property
    def version(self) -> str:
        return getattr(self.transformers, "__version__", "unknown")

    def list_tokenizers(
        self,
        *,
        scope: str = "all",
        existing_entries: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Return stable tokenizer entries for the requested scope."""
        scoped_model_types = self.model_types_for_scope(scope)
        existing_by_class = {entry["class_name"]: entry for entry in existing_entries or [] if entry.get("class_name")}

        auto_model_types_by_class = self._auto_model_types_by_class()
        exported_names = set(self.exported_tokenizer_class_names())
        all_class_names = sorted(set(auto_model_types_by_class) | exported_names)

        entries = []
        for class_name in all_class_names:
            existing = copy.deepcopy(existing_by_class.get(class_name, {}))
            auto_model_types = sorted(auto_model_types_by_class.get(class_name, set()))
            generation_model_types = sorted(set(auto_model_types).intersection(scoped_model_types or set()))
            if scoped_model_types is not None and not generation_model_types:
                continue

            sources = []
            if auto_model_types:
                sources.append("auto_mapping")
            if class_name in exported_names:
                sources.append("transformers_export")

            entries.append(
                {
                    "id": existing.get("id") or self.tokenizer_id(class_name),
                    "class_name": class_name,
                    "implementation": existing.get("implementation") or self.tokenizer_implementation(class_name),
                    "source": sources,
                    "auto_model_types": auto_model_types,
                    "generation_model_types": generation_model_types,
                    "example_model_id": existing.get("example_model_id"),
                    "validation_status": existing.get("validation_status") or "unvalidated",
                    "validation_notes": existing.get("validation_notes"),
                    "requires_trust_remote_code": bool(existing.get("requires_trust_remote_code", False)),
                    "optional_dependencies": existing.get("optional_dependencies") or [],
                    "notes": existing.get("notes"),
                }
            )

        return entries

    def get_tokenizer(
        self,
        tokenizer_id_or_class_name: str,
        *,
        scope: str = "all",
        existing_entries: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Return one tokenizer entry by generated id or class name."""
        for entry in self.list_tokenizers(scope=scope, existing_entries=existing_entries):
            if tokenizer_id_or_class_name in {entry["id"], entry["class_name"]}:
                return entry
        return None

    def model_types_for_scope(self, scope: str) -> set[str] | None:
        if scope == "all":
            return None
        if scope not in MODEL_TASK_MAPPING_NAMES:
            expected = ", ".join(["all", *MODEL_TASK_MAPPING_NAMES])
            raise ValueError(f"Unknown tokenizer inventory scope {scope!r}. Expected one of: {expected}")

        model_types: set[str] = set()
        for mapping_name in MODEL_TASK_MAPPING_NAMES[scope]:
            model_types.update(self.model_types_from_modeling_auto_mapping(mapping_name))
        return model_types

    def model_types_from_modeling_auto_mapping(self, mapping_name: str) -> set[str]:
        """Return model_type keys from modeling_auto without importing torch-backed modules."""
        tree = ast.parse(self._modeling_auto_path().read_text(encoding="utf-8"))

        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == mapping_name for target in node.targets):
                continue
            if not isinstance(node.value, ast.Call) or not node.value.args:
                return set()
            pairs = ast.literal_eval(node.value.args[0])
            return {model_type for model_type, _model_class in pairs}

        raise RuntimeError(f"{mapping_name} was not found in {self._modeling_auto_path()}")

    def exported_tokenizer_class_names(self) -> list[str]:
        names = []
        for name in dir(self.transformers):
            try:
                value = getattr(self.transformers, name)
            except Exception:
                continue
            if self._is_transformers_tokenizer_export(name, value):
                names.append(name)
        return sorted(set(names))

    def tokenizer_implementation(self, class_name: str) -> str:
        try:
            tokenizer_class = getattr(self.transformers, class_name)
        except Exception:
            tokenizer_class = None

        try:
            fast_base = self.transformers.PreTrainedTokenizerFast
            if tokenizer_class is not None and issubclass(tokenizer_class, fast_base):
                return "fast"
        except Exception:
            pass

        try:
            slow_base = self.transformers.PreTrainedTokenizer
            if tokenizer_class is not None and issubclass(tokenizer_class, slow_base):
                return "slow"
        except Exception:
            pass

        if class_name.endswith("TokenizerFast"):
            return "fast"
        if class_name.endswith("Tokenizer"):
            return "standard"
        return "unknown"

    @staticmethod
    def tokenizer_id(class_name: str) -> str:
        if class_name.endswith("TokenizerFast"):
            base = f"{class_name.removesuffix('TokenizerFast')}Fast"
        else:
            base = class_name.removesuffix("Tokenizer")
        parts = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", base).replace("_", "-").lower()
        return f"hf-{parts}-tokenizer"

    def _auto_model_types_by_class(self) -> dict[str, set[str]]:
        auto_model_types_by_class: dict[str, set[str]] = defaultdict(set)
        for model_type, mapping_value in self.tokenization_auto.TOKENIZER_MAPPING_NAMES.items():
            for class_name in self._tokenizer_class_names_from_mapping_value(mapping_value):
                auto_model_types_by_class[class_name].add(model_type)
        return auto_model_types_by_class

    def _is_transformers_tokenizer_export(self, name: str, value: Any) -> bool:
        if name in {"AutoTokenizer", "PreTrainedTokenizer", "PreTrainedTokenizerFast"}:
            return False
        if not (name.endswith("Tokenizer") or name.endswith("TokenizerFast")):
            return False
        if not isinstance(value, type):
            return False
        if not getattr(value, "__module__", "").startswith(self.transformers.__name__):
            return False
        try:
            return issubclass(
                value,
                (self.transformers.PreTrainedTokenizer | self.transformers.PreTrainedTokenizerFast),
            )
        except Exception:
            return False

    def _modeling_auto_path(self) -> Path:
        if self.modeling_auto_path is not None:
            return self.modeling_auto_path
        spec = find_spec("transformers")
        if spec is None or spec.origin is None:
            raise RuntimeError("Could not locate the installed transformers package")
        return Path(spec.origin).parent / "models" / "auto" / "modeling_auto.py"

    @classmethod
    def _tokenizer_class_names_from_mapping_value(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, type):
            return [value.__name__]
        if isinstance(value, (tuple | list)):
            names = []
            for item in value:
                names.extend(cls._tokenizer_class_names_from_mapping_value(item))
            return names
        return []
