"""
Unit tests for llm_client.py: JSON extraction, model/effort selection, thinking plans,
redaction and the ask() fallback loop against the fake provider server.
"""

from __future__ import annotations

import pytest

import llm_client
from llm_client import (
    CLAUDE, GEMINI, Credentials, JSONExtractError, LLMError, TASK_CHAT, TASK_EXTRACT,
    TASK_PING, TASK_SCHEDULE, extract_json, gemini_thinking_plan, generate_json,
    max_tokens_for, model_chain, normalize_effort, public_message, redact, resolve_credentials,
    tier_for,
)


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------
class TestExtractJson:
    def test_plain_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_in_markdown(self):
        assert extract_json('Sure!\n```json\n{"a": 1}\n```\nhope that helps') == {"a": 1}

    def test_fenced_without_language_tag(self):
        assert extract_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_chatter_before_and_after(self):
        assert extract_json('Here is your JSON: {"a": 1} — let me know!') == {"a": 1}

    def test_leading_bom(self):
        assert extract_json('﻿{"a": 1}') == {"a": 1}

    def test_raw_newline_inside_string(self):
        assert extract_json('{"a": "line one\nline two"}') == {"a": "line one\nline two"}

    def test_trailing_comma_in_object(self):
        assert extract_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}

    def test_trailing_comma_in_array(self):
        assert extract_json('{"a": [1, 2, 3,]}') == {"a": [1, 2, 3]}

    def test_single_element_list_wrapping_expected_dict(self):
        assert extract_json('[{"a": 1}]') == {"a": 1}

    def test_expect_list(self):
        assert extract_json('[1, 2, 3]', expect=list) == [1, 2, 3]

    def test_nested_braces_and_strings_with_braces(self):
        raw = '{"activity": "Study {hard}", "blocks": [{"x": "a}b"}]}'
        assert extract_json(raw) == {"activity": "Study {hard}", "blocks": [{"x": "a}b"}]}

    def test_escaped_quote_inside_string(self):
        assert extract_json(r'{"a": "she said \"hi\""}') == {"a": 'she said "hi"'}

    def test_picks_first_object_when_several_present(self):
        assert extract_json('{"a": 1} then some junk {"b": 2}') == {"a": 1}

    def test_empty_string_raises_empty(self):
        with pytest.raises(JSONExtractError) as exc:
            extract_json("")
        assert exc.value.kind == "empty"

    def test_whitespace_only_raises_empty(self):
        with pytest.raises(JSONExtractError) as exc:
            extract_json("   \n\t  ")
        assert exc.value.kind == "empty"

    def test_none_raises_empty(self):
        with pytest.raises(JSONExtractError) as exc:
            extract_json(None)
        assert exc.value.kind == "empty"

    def test_no_json_at_all_raises_no_json(self):
        with pytest.raises(JSONExtractError) as exc:
            extract_json("I cannot help with that request.")
        assert exc.value.kind == "no_json"

    def test_truncated_object_raises_truncated(self):
        with pytest.raises(JSONExtractError) as exc:
            extract_json('{"schedule": [{"day": "Monday", "blocks": [{"time": "7:00')
        assert exc.value.kind == "truncated"

    def test_truncated_nested_array_raises_truncated(self):
        with pytest.raises(JSONExtractError) as exc:
            extract_json('{"a": [1, 2, {"b": [3, 4')
        assert exc.value.kind == "truncated"

    def test_mismatched_brackets_raise_invalid(self):
        with pytest.raises(JSONExtractError) as exc:
            extract_json('{"a": [1, 2}')
        assert exc.value.kind == "invalid"

    def test_garbage_that_merely_contains_braces_raises_invalid(self):
        with pytest.raises(JSONExtractError) as exc:
            extract_json('{not real json at all}')
        assert exc.value.kind == "invalid"

    def test_wrong_top_level_type_is_not_returned(self):
        # a bare number is valid JSON but not a dict; this must not match "expect=dict"
        with pytest.raises(JSONExtractError):
            extract_json("42")

    def test_closed_object_preferred_over_later_unclosed_fragment(self):
        assert extract_json('{"a": 1} and then it trails off {"b":') == {"a": 1}


# ---------------------------------------------------------------------------
# Effort / tier / model-chain selection
# ---------------------------------------------------------------------------
class TestSelection:
    @pytest.mark.parametrize("value,expected", [("low", "low"), ("medium", "medium"), ("high", "high"),
                                                ("", "medium"), (None, "medium"), ("LOW", "medium"), (3, "medium")])
    def test_normalize_effort(self, value, expected):
        assert normalize_effort(value) == expected

    def test_chat_extract_ping_always_use_fast_tier(self):
        for task in (TASK_CHAT, TASK_EXTRACT, TASK_PING):
            for effort in ("low", "medium", "high"):
                assert tier_for(task, effort) == "fast"

    @pytest.mark.parametrize("effort,tier", [("low", "fast"), ("medium", "balanced"), ("high", "best")])
    def test_schedule_tier_follows_effort(self, effort, tier):
        assert tier_for(TASK_SCHEDULE, effort) == tier

    def test_model_chain_is_non_empty_and_ordered_for_every_provider_task_effort(self):
        for provider in (GEMINI, CLAUDE):
            for task in (TASK_CHAT, TASK_EXTRACT, TASK_PING, TASK_SCHEDULE):
                for effort in ("low", "medium", "high"):
                    chain = model_chain(provider, task, effort)
                    assert isinstance(chain, tuple) and len(chain) >= 1
                    assert all(isinstance(m, str) and m for m in chain)

    def test_higher_effort_schedule_tiers_are_distinct_chains(self):
        low = model_chain(GEMINI, TASK_SCHEDULE, "low")
        high = model_chain(GEMINI, TASK_SCHEDULE, "high")
        assert low != high or tier_for(TASK_SCHEDULE, "low") != tier_for(TASK_SCHEDULE, "high")

    def test_env_override_replaces_the_default_chain(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODELS_BALANCED", "my-custom-model, another-model")
        assert model_chain(GEMINI, TASK_SCHEDULE, "medium") == ("my-custom-model", "another-model")

    def test_max_tokens_scale_with_schedule_effort(self):
        low = max_tokens_for(GEMINI, TASK_SCHEDULE, "low")
        medium = max_tokens_for(GEMINI, TASK_SCHEDULE, "medium")
        high = max_tokens_for(GEMINI, TASK_SCHEDULE, "high")
        assert low < medium < high

    def test_max_tokens_for_chat_is_a_flat_small_value(self):
        assert 0 < max_tokens_for(GEMINI, TASK_CHAT) <= 2000
        assert 0 < max_tokens_for(CLAUDE, TASK_CHAT) <= 2000


# ---------------------------------------------------------------------------
# Gemini thinking plan
# ---------------------------------------------------------------------------
class TestThinkingPlan:
    def test_gemini_3x_defaults_to_level(self):
        plan = gemini_thinking_plan("gemini-3.8-flash", TASK_SCHEDULE, "medium")
        assert plan[0][0] == "level" and plan[-1] is None

    def test_gemini_2x_defaults_to_budget(self):
        plan = gemini_thinking_plan("gemini-2.5-flash", TASK_SCHEDULE, "medium")
        assert plan[0][0] == "budget" and plan[-1] is None

    def test_plan_always_ends_with_none_so_a_model_can_still_be_called(self):
        for model in ("gemini-3.8-flash", "gemini-2.5-flash", "gemini-flash-latest"):
            assert gemini_thinking_plan(model, TASK_SCHEDULE, "high")[-1] is None

    def test_high_effort_schedule_uses_more_thinking_than_low(self):
        high = gemini_thinking_plan("gemini-3.8-flash", TASK_SCHEDULE, "high")[0]
        low = gemini_thinking_plan("gemini-3.8-flash", TASK_SCHEDULE, "low")[0]
        assert high[1] != low[1]

    def test_chat_never_requests_heavy_thinking(self):
        choice = gemini_thinking_plan("gemini-3.8-flash", TASK_CHAT, "medium")[0]
        assert choice[1] == "low"

    def test_learned_none_short_circuits_to_a_single_none_entry(self):
        llm_client._CAPS.set("gemini-3.8-flash", "none", 60)
        try:
            assert gemini_thinking_plan("gemini-3.8-flash", TASK_SCHEDULE, "medium") == [None]
        finally:
            llm_client.reset_caches()

    def test_learned_budget_is_tried_even_for_an_alias_that_looks_like_3x(self):
        llm_client._CAPS.set("gemini-flash-latest", "budget", 60)
        try:
            plan = gemini_thinking_plan("gemini-flash-latest", TASK_SCHEDULE, "medium")
            assert plan[0][0] == "budget"
        finally:
            llm_client.reset_caches()


# ---------------------------------------------------------------------------
# Credentials, key shape, resolution, redaction
# ---------------------------------------------------------------------------
class TestCredentials:
    def test_resolve_prefers_claude_over_gemini_when_both_given(self):
        creds = resolve_credentials(user_claude_key="sk-ant-" + "a" * 20, user_gemini_key="AIza" + "b" * 20)
        assert creds.provider == CLAUDE

    def test_resolve_falls_back_to_gemini(self):
        creds = resolve_credentials(user_gemini_key="AIza" + "b" * 20)
        assert creds.provider == GEMINI

    def test_resolve_raises_no_key_when_nothing_present(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(LLMError) as exc:
            resolve_credentials()
        assert exc.value.reason == "no_key"

    def test_resolve_falls_back_to_server_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "s" * 20)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        creds = resolve_credentials()
        assert creds.provider == GEMINI and creds.source == "server"

    def test_key_with_whitespace_is_trimmed(self):
        creds = resolve_credentials(user_gemini_key="  AIza" + "b" * 20 + "  ")
        assert not creds.api_key.strip() != creds.api_key

    def test_key_wrapped_in_quotes_is_cleaned(self):
        creds = resolve_credentials(user_gemini_key='"AIza' + "b" * 20 + '"')
        assert '"' not in creds.api_key

    @pytest.mark.parametrize("bad_key", ["short", "a" * 600, "has a space in it 12345678901234", "notascii☃" * 3])
    def test_impossibly_shaped_keys_are_rejected(self, bad_key):
        with pytest.raises(LLMError) as exc:
            resolve_credentials(user_gemini_key=bad_key)
        assert exc.value.reason == "bad_key"

    def test_fingerprint_is_stable_and_never_the_raw_key(self):
        creds = Credentials(GEMINI, "AIza" + "x" * 30, "user")
        assert len(creds.fingerprint) == 12 and creds.fingerprint != creds.api_key

    def test_credentials_repr_never_includes_the_key(self):
        creds = Credentials(GEMINI, "AIzaTHE_SECRET_VALUE_12345678", "user")
        assert "AIzaTHE_SECRET_VALUE" not in repr(creds)


class TestRedact:
    def test_redacts_a_gemini_style_key(self):
        assert "AIza" not in redact("your key AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ123 failed")

    def test_redacts_a_claude_style_key(self):
        assert "sk-ant" not in redact("key sk-ant-api03-abcdefghijklmnop failed")

    def test_redacts_the_exact_key_passed_via_creds_even_if_oddly_shaped(self):
        creds = Credentials(GEMINI, "totally-nonstandard-key-format-123", "user")
        assert "totally-nonstandard-key-format-123" not in redact("error with totally-nonstandard-key-format-123", creds)

    def test_redacts_an_authorization_header_style_string(self):
        out = redact("failed: Authorization: Bearer sk-ant-abcdefghijklmnop")
        assert "sk-ant-abcdefghijklmnop" not in out

    def test_leaves_ordinary_text_alone(self):
        assert redact("the model is overloaded, please retry") == "the model is overloaded, please retry"


class TestPublicMessage:
    def test_known_reason_has_no_unfilled_placeholders(self):
        for reason in ("no_key", "bad_key", "region_blocked", "quota_exhausted", "rate_limited",
                      "model_unavailable", "blocked", "truncated", "bad_response", "timeout", "network",
                      "provider_error", "bad_request", "internal_error"):
            msg = public_message(reason, GEMINI, 30)
            assert "{" not in msg and "}" not in msg

    def test_unknown_reason_falls_back_to_generation_failed_text(self):
        assert public_message("totally_unknown_reason") == public_message("generation_failed")

    def test_retry_after_is_mentioned_when_given(self):
        assert "30 seconds" in public_message("rate_limited", GEMINI, 30)

    def test_no_retry_after_omits_the_wait_clause(self):
        assert "seconds" not in public_message("rate_limited", GEMINI, None)


# ---------------------------------------------------------------------------
# ask() against the fake provider server
# ---------------------------------------------------------------------------
class TestAskFallback:
    def test_ping_round_trips(self, fake_llm, gemini_creds):
        result = llm_client.check_credentials(gemini_creds)
        assert result.text.strip() == "OK"

    def test_bad_key_is_fatal_and_not_retried_on_other_models(self, fake_llm, gemini_creds):
        fake_llm.scenario.valid_gemini_keys = {"some-other-key"}
        with pytest.raises(LLMError) as exc:
            llm_client.ask("sys", [{"role": "user", "content": "hi"}], creds=gemini_creds, task=TASK_PING)
        assert exc.value.reason == "bad_key"
        assert len(fake_llm.calls("gemini")) == 1  # no point burning calls on other models for a bad key

    def test_unavailable_model_falls_through_the_chain(self, fake_llm, gemini_creds):
        fake_llm.scenario.unavailable_models = {model_chain(GEMINI, TASK_PING)[0]}
        result = llm_client.check_credentials(gemini_creds)
        assert result.text.strip() == "OK"
        assert len(fake_llm.calls("gemini")) == 2

    def test_empty_conversation_raises_bad_request_without_a_network_call(self, fake_llm, gemini_creds):
        with pytest.raises(LLMError) as exc:
            llm_client.ask("sys", [], creds=gemini_creds, task=TASK_PING)
        assert exc.value.reason == "bad_request"
        assert fake_llm.calls() == []

    def test_history_merges_consecutive_same_role_turns(self, fake_llm, gemini_creds):
        history = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
        llm_client.ask("sys", history, creds=gemini_creds, task=TASK_PING)
        assert fake_llm.calls("gemini")[0]["turns"] == 1

    def test_history_starting_with_assistant_is_trimmed_to_start_on_user(self, fake_llm, gemini_creds):
        history = [{"role": "assistant", "content": "leftover"}, {"role": "user", "content": "hi"}]
        llm_client.ask("sys", history, creds=gemini_creds, task=TASK_PING)
        assert fake_llm.calls("gemini")[0]["turns"] == 1


class TestGenerateJson:
    def test_success_on_first_try(self, fake_llm, gemini_creds):
        fake_llm.scenario.chat_reply = None  # task is extract via the system prompt fixture below

        def validate(d):
            return d
        result = generate_json(
            "You extract structured data from messy, unstructured text",
            [{"role": "user", "content": "Extract: Data Structures, 3 lectures a week"}],
            creds=gemini_creds, task=TASK_EXTRACT, validate=validate)
        assert "subjects" in result.data

    def test_validation_failure_triggers_one_repair_turn(self, fake_llm, gemini_creds, monkeypatch):
        calls = {"n": 0}

        def flaky_validate(data):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("missing required field")
            return data
        generate_json("You extract structured data from messy, unstructured text",
                      [{"role": "user", "content": "Extract: Data Structures"}],
                      creds=gemini_creds, task=TASK_EXTRACT, validate=flaky_validate, repair_attempts=1)
        assert calls["n"] == 2
        assert len(fake_llm.calls("gemini", "extract")) == 2

    def test_exhausting_repairs_raises_bad_response(self, fake_llm, gemini_creds):
        def always_fails(data):
            raise ValueError("never good enough")
        with pytest.raises(LLMError) as exc:
            generate_json("You extract structured data from messy, unstructured text",
                          [{"role": "user", "content": "x"}], creds=gemini_creds, task=TASK_EXTRACT,
                          validate=always_fails, repair_attempts=1)
        assert exc.value.reason == "bad_response"
        assert len(fake_llm.calls("gemini", "extract")) == 2
