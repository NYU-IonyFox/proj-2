"""
tests/test_l1_l2.py
Unit tests for L1 Screening (screening.py) and L2 Multilingual Support (multilingual.py).

All NLLB translation calls are mocked — no model download occurs during testing.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

os.environ.setdefault("LOCAL_DEV", "true")


# ---------------------------------------------------------------------------
# L1: validate_input
# ---------------------------------------------------------------------------


class TestValidateInput:
    def test_raises_on_empty_string(self):
        from input_processor.screening import validate_input

        with pytest.raises(ValueError, match="empty"):
            validate_input("")

    def test_raises_on_non_string(self):
        from input_processor.screening import validate_input

        with pytest.raises(ValueError):
            validate_input(12345)  # type: ignore[arg-type]

    def test_raises_on_text_exceeding_50000_chars(self):
        from input_processor.screening import validate_input

        long_text = "a" * 50001
        with pytest.raises(ValueError, match="50000"):
            validate_input(long_text)

    def test_accepts_exactly_50000_chars(self):
        from input_processor.screening import validate_input

        text = "a" * 50000
        validate_input(text)  # should not raise

    def test_accepts_normal_text(self):
        from input_processor.screening import validate_input

        validate_input("Hello, world!")  # should not raise


# ---------------------------------------------------------------------------
# L1: detect_language
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    def test_returns_eng_latn_for_english(self):
        from input_processor.screening import detect_language

        result = detect_language("This is a simple English sentence.")
        assert result == "eng_Latn"

    def test_returns_unknown_on_langdetect_exception(self, monkeypatch):
        from langdetect import LangDetectException
        from input_processor import screening

        monkeypatch.setattr(
            "input_processor.screening.detect",
            lambda _text: (_ for _ in ()).throw(LangDetectException(0, "mocked")),
        )
        result = screening.detect_language("some text")
        assert result == "unknown"

    def test_returns_unknown_for_unmapped_language(self, monkeypatch):
        from input_processor import screening

        monkeypatch.setattr(
            "input_processor.screening.detect",
            lambda _text: "xx",  # not in _LANG_MAP
        )
        result = screening.detect_language("text")
        assert result == "unknown"

    def test_maps_zh_cn_to_zho_hans(self, monkeypatch):
        from input_processor import screening

        monkeypatch.setattr(
            "input_processor.screening.detect",
            lambda _text: "zh-cn",
        )
        assert screening.detect_language("text") == "zho_Hans"


# ---------------------------------------------------------------------------
# L2: compute_uncertainty_flag
# ---------------------------------------------------------------------------


class TestComputeUncertaintyFlag:
    def test_true_when_confidence_below_threshold(self):
        from input_processor.multilingual import compute_uncertainty_flag

        assert compute_uncertainty_flag(0.50, "fra_Latn") is True

    def test_true_when_source_lang_unknown(self):
        from input_processor.multilingual import compute_uncertainty_flag

        assert compute_uncertainty_flag(1.0, "unknown") is True

    def test_false_when_high_confidence_english(self):
        from input_processor.multilingual import compute_uncertainty_flag

        assert compute_uncertainty_flag(1.0, "eng_Latn") is False

    def test_false_when_confidence_equals_threshold(self):
        from input_processor import multilingual as ml

        # At exactly the threshold, flag should be False (condition is strictly <)
        assert ml.compute_uncertainty_flag(ml.UNCERTAINTY_THRESHOLD, "fra_Latn") is False

    def test_uses_uncertainty_threshold_constant(self, monkeypatch):
        """Monkeypatching UNCERTAINTY_THRESHOLD to 0.80 changes flag behaviour."""
        from input_processor import multilingual as ml

        original = ml.UNCERTAINTY_THRESHOLD
        try:
            ml.UNCERTAINTY_THRESHOLD = 0.80
            # confidence=0.70 → below 0.80 → True
            assert ml.compute_uncertainty_flag(0.70, "fra_Latn") is True
            # confidence=0.85 → above 0.80 → False
            assert ml.compute_uncertainty_flag(0.85, "fra_Latn") is False
        finally:
            ml.UNCERTAINTY_THRESHOLD = original


# ---------------------------------------------------------------------------
# L2: build_multilingual_bundle
# ---------------------------------------------------------------------------


class TestBuildMultilingualBundle:
    def _mock_translate(self, confidence_map: dict[str, float]):
        """Return a translate_to_english mock that uses the confidence_map."""

        def _translate(text: str, source_lang: str):
            if source_lang == "eng_Latn":
                return text, 1.0
            conf = confidence_map.get(source_lang, 0.75)
            return f"[translated]{text}", conf

        return _translate

    def test_excludes_items_with_confidence_below_threshold(self):
        from input_processor import multilingual as ml

        texts = {
            "eng_Latn": "English text",
            "fra_Latn": "Texte français",
        }

        with patch.object(ml, "translate_to_english", self._mock_translate({"fra_Latn": 0.30})):
            bundle = ml.build_multilingual_bundle(texts)

        langs = [item["source_language"] for item in bundle]
        assert "fra_Latn" not in langs
        assert "eng_Latn" in langs

    def test_includes_warning_true_for_mid_confidence_items(self):
        from input_processor import multilingual as ml

        texts = {
            "eng_Latn": "English text",
            "fra_Latn": "Texte français",
        }

        with patch.object(ml, "translate_to_english", self._mock_translate({"fra_Latn": 0.70})):
            bundle = ml.build_multilingual_bundle(texts)

        fra_items = [item for item in bundle if item["source_language"] == "fra_Latn"]
        assert len(fra_items) == 1
        assert fra_items[0]["warning"] is True

    def test_includes_warning_false_for_high_confidence_items(self):
        from input_processor import multilingual as ml

        texts = {
            "eng_Latn": "English text",
            "fra_Latn": "Texte français",
        }

        with patch.object(ml, "translate_to_english", self._mock_translate({"fra_Latn": 0.95})):
            bundle = ml.build_multilingual_bundle(texts)

        fra_items = [item for item in bundle if item["source_language"] == "fra_Latn"]
        assert len(fra_items) == 1
        assert fra_items[0]["warning"] is False

    def test_raw_text_is_unmodified_original(self):
        from input_processor import multilingual as ml

        original = "Bonjour le monde"
        texts = {"fra_Latn": original}

        with patch.object(ml, "translate_to_english", self._mock_translate({"fra_Latn": 0.85})):
            bundle = ml.build_multilingual_bundle(texts)

        assert len(bundle) == 1
        assert bundle[0]["raw_text"] == original

    def test_excludes_unknown_language(self):
        from input_processor import multilingual as ml

        texts = {"unknown": "some text", "eng_Latn": "English text"}

        with patch.object(ml, "translate_to_english", self._mock_translate({})):
            bundle = ml.build_multilingual_bundle(texts)

        langs = [item["source_language"] for item in bundle]
        assert "unknown" not in langs

    def test_empty_bundle_when_all_low_confidence(self):
        from input_processor import multilingual as ml

        texts = {"fra_Latn": "text", "spa_Latn": "text"}

        with patch.object(
            ml,
            "translate_to_english",
            self._mock_translate({"fra_Latn": 0.10, "spa_Latn": 0.20}),
        ):
            bundle = ml.build_multilingual_bundle(texts)

        assert bundle == []


# ---------------------------------------------------------------------------
# L2: assess_bundle_status
# ---------------------------------------------------------------------------


class TestAssessBundleStatus:
    def test_hold_no_english_when_english_excluded(self):
        from input_processor.multilingual import assess_bundle_status

        result = assess_bundle_status([], english_excluded=True)
        assert result["status"] == "hold_no_english"

    def test_single_language_when_bundle_is_empty(self):
        from input_processor.multilingual import assess_bundle_status

        result = assess_bundle_status([], english_excluded=False)
        assert result["status"] == "single_language"
        assert result["multilingual_jailbreak_forced_low"] is True

    def test_single_language_when_only_english_in_bundle(self):
        from input_processor.multilingual import assess_bundle_status

        bundle = [
            {
                "source_language": "eng_Latn",
                "raw_text": "text",
                "translated_text": "text",
                "translation_confidence": 1.0,
                "warning": False,
            }
        ]
        result = assess_bundle_status(bundle, english_excluded=False)
        assert result["status"] == "single_language"
        assert result["multilingual_jailbreak_forced_low"] is True

    def test_normal_when_english_and_non_english_present(self):
        from input_processor.multilingual import assess_bundle_status

        bundle = [
            {
                "source_language": "eng_Latn",
                "raw_text": "text",
                "translated_text": "text",
                "translation_confidence": 1.0,
                "warning": False,
            },
            {
                "source_language": "fra_Latn",
                "raw_text": "texte",
                "translated_text": "text",
                "translation_confidence": 0.90,
                "warning": False,
            },
        ]
        result = assess_bundle_status(bundle, english_excluded=False)
        assert result["status"] == "normal"

    def test_all_non_english_low_confidence_true_when_all_warning(self):
        from input_processor.multilingual import assess_bundle_status

        bundle = [
            {
                "source_language": "eng_Latn",
                "raw_text": "text",
                "translated_text": "text",
                "translation_confidence": 1.0,
                "warning": False,
            },
            {
                "source_language": "fra_Latn",
                "raw_text": "texte",
                "translated_text": "text",
                "translation_confidence": 0.65,
                "warning": True,
            },
            {
                "source_language": "spa_Latn",
                "raw_text": "texto",
                "translated_text": "text",
                "translation_confidence": 0.70,
                "warning": True,
            },
        ]
        result = assess_bundle_status(bundle, english_excluded=False)
        assert result["all_non_english_low_confidence"] is True

    def test_all_non_english_low_confidence_false_when_some_high_confidence(self):
        from input_processor.multilingual import assess_bundle_status

        bundle = [
            {
                "source_language": "eng_Latn",
                "raw_text": "text",
                "translated_text": "text",
                "translation_confidence": 1.0,
                "warning": False,
            },
            {
                "source_language": "fra_Latn",
                "raw_text": "texte",
                "translated_text": "text",
                "translation_confidence": 0.65,
                "warning": True,
            },
            {
                "source_language": "spa_Latn",
                "raw_text": "texto",
                "translated_text": "text",
                "translation_confidence": 0.95,
                "warning": False,
            },
        ]
        result = assess_bundle_status(bundle, english_excluded=False)
        assert result["all_non_english_low_confidence"] is False


# ---------------------------------------------------------------------------
# L1: parse_multilingual_input
# ---------------------------------------------------------------------------


class TestParseMultilingualInput:
    def test_returns_none_for_plain_text(self):
        from input_processor.screening import parse_multilingual_input

        assert parse_multilingual_input("VeriMedia is a content moderation tool.") is None

    def test_splits_en_fr_tagged_text(self):
        from input_processor.screening import parse_multilingual_input

        text = "[EN]\nVeriMedia is a tool.\n[FR]\nVeriMedia est un outil."
        result = parse_multilingual_input(text)
        assert result is not None
        assert set(result.keys()) == {"EN", "FR"}
        assert "VeriMedia is a tool." in result["EN"]
        assert "VeriMedia est un outil." in result["FR"]

    def test_splits_three_language_input(self):
        from input_processor.screening import parse_multilingual_input

        text = "[EN]\nHello world.\n[FR]\nBonjour monde.\n[AR]\nمرحبا بالعالم."
        result = parse_multilingual_input(text)
        assert result is not None
        assert set(result.keys()) == {"EN", "FR", "AR"}
        assert result["EN"] == "Hello world."
        assert result["FR"] == "Bonjour monde."
        assert result["AR"] == "مرحبا بالعالم."

    def test_skips_unknown_tag_with_warning(self, capsys):
        from input_processor.screening import parse_multilingual_input

        text = "[EN]\nHello.\n[XX]\nUnknown language text."
        result = parse_multilingual_input(text)
        assert result is not None
        assert "EN" in result
        assert "XX" not in result
        captured = capsys.readouterr()
        assert "XX" in captured.err
        assert "skipping" in captured.err.lower()

    def test_returns_none_for_invalid_patterns(self):
        from input_processor.screening import parse_multilingual_input

        # [english] — more than 2 letters; [F] — 1 letter; [123] — digits
        assert parse_multilingual_input("[english]\nSome text here.") is None
        assert parse_multilingual_input("[F]\nSome text here.") is None
        assert parse_multilingual_input("[123]\nSome text here.") is None

    def test_returns_none_for_empty_segments(self):
        from input_processor.screening import parse_multilingual_input

        # Tag present but no text following it
        assert parse_multilingual_input("[EN]\n") is None

    def test_returns_none_when_only_unknown_tags(self, capsys):
        from input_processor.screening import parse_multilingual_input

        text = "[XX]\nOnly unknown tag."
        result = parse_multilingual_input(text)
        assert result is None


# ---------------------------------------------------------------------------
# run_council: multilingual bundle integration
# ---------------------------------------------------------------------------


class TestRunCouncilMultilingual:
    _MOCK_RESOLUTION = {
        "final_decision": "PASS",
        "decision_tier": "auto",
        "decision_rule_triggered": "rule_5",
        "expert_summary": {"expert_1": "LOW", "expert_2": "LOW", "expert_3": "LOW"},
        "council_reasoning": "All experts LOW.",
        "governance_action": {},
    }

    def _expert_output(self, eid):
        return {
            "expert_id": eid,
            "expert_name": "Test Expert",
            "expert_risk_level": "LOW",
            "dimension_scores": [],
            "aggregation_trace": "Rule 5 fired: no HIGH/MEDIUM. expert_risk_level = LOW.",
            "multilingual_flag_applied": False,
        }

    def test_multilingual_tagged_input_produces_bundle(self):
        """[EN]/[FR] tagged input → multilingual_bundle is a list in council output."""
        from output.final_output import run_council

        tagged = "[EN]\nVeriMedia is a test tool.\n[FR]\nVeriMedia est un outil de test."

        def mock_translate(text, lang):
            if lang == "eng_Latn":
                return text, 1.0
            return f"translated:{text}", 0.9

        with (
            patch("input_processor.multilingual.translate_to_english", side_effect=mock_translate),
            patch("experts.expert_1_security.run_expert_1", return_value=self._expert_output("expert_1")),
            patch("experts.expert_2_content.run_expert_2", return_value=self._expert_output("expert_2")),
            patch("experts.expert_3_governance.run_expert_3", return_value=self._expert_output("expert_3")),
            patch("council.arbitration.run_arbitration", return_value={"expert_levels": {}}),
            patch("council.resolution.run_resolution", return_value=self._MOCK_RESOLUTION),
            patch("output.final_output.write_audit_log", return_value="test-audit.json"),
        ):
            result = run_council("test_agent", tagged)

        assert "multilingual_bundle" in result
        assert isinstance(result["multilingual_bundle"], list)
        # FR segment should appear in bundle
        langs = [item["source_language"] for item in result["multilingual_bundle"]]
        assert "fra_Latn" in langs

    def test_plain_text_input_produces_no_bundle(self):
        """Plain text (no tags) → multilingual_bundle is None in council output."""
        from output.final_output import run_council

        plain = "VeriMedia is a content moderation tool that processes text data."

        def mock_translate(text, lang):
            return text, 1.0

        with (
            patch("input_processor.multilingual.translate_to_english", side_effect=mock_translate),
            patch("experts.expert_1_security.run_expert_1", return_value=self._expert_output("expert_1")),
            patch("experts.expert_2_content.run_expert_2", return_value=self._expert_output("expert_2")),
            patch("experts.expert_3_governance.run_expert_3", return_value=self._expert_output("expert_3")),
            patch("council.arbitration.run_arbitration", return_value={"expert_levels": {}}),
            patch("council.resolution.run_resolution", return_value=self._MOCK_RESOLUTION),
            patch("output.final_output.write_audit_log", return_value="test-audit.json"),
        ):
            result = run_council("test_agent", plain)

        assert result.get("multilingual_bundle") is None
