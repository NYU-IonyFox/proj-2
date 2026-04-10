"""
tests/conftest.py
Mock transformers model loading at module level so that expert_base can be imported
without a GPU or model download. Patching runs before any test file is collected.
"""
import os
from unittest.mock import MagicMock

# Ensure LOCAL_DEV=true before expert_base module-level code runs
os.environ.setdefault("LOCAL_DEV", "true")

# ---------------------------------------------------------------------------
# Persistent mock objects shared across the test session
# ---------------------------------------------------------------------------

mock_qwen_tokenizer = MagicMock(name="mock_qwen_tokenizer")
mock_qwen_model = MagicMock(name="mock_qwen_model")
mock_qwen_model.to.return_value = mock_qwen_model
mock_qwen_model.device = "cpu"

# Patch transformers classes so from_pretrained returns the mocks above.
# conftest.py module-level code executes during collection, before test_experts.py
# is imported — so expert_base's module-level from_pretrained calls see the mocks.
import transformers  # noqa: E402

_mock_tokenizer_cls = MagicMock(name="AutoTokenizer")
_mock_tokenizer_cls.from_pretrained.return_value = mock_qwen_tokenizer

_mock_model_cls = MagicMock(name="AutoModelForCausalLM")
_mock_model_cls.from_pretrained.return_value = mock_qwen_model

transformers.AutoTokenizer = _mock_tokenizer_cls
transformers.AutoModelForCausalLM = _mock_model_cls
