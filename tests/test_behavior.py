import pytest
from companion.behavior import GesturePolicy
from companion.capabilities import AVATAR_ACTIONS
from companion.text import SpeechText


class FixedRandom:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def random(self):
        self.calls += 1
        return self.value


@pytest.mark.parametrize("intent,gesture", [("greeting", "wave"), ("explaining", "explain")])
def test_meaningful_actions_do_not_depend_on_randomness(intent, gesture):
    rng = FixedRandom(1)
    policy = GesturePolicy(rng)
    for _ in range(5):
        assert policy.choose("one", intent, "neutral") == gesture
    assert rng.calls == 0


@pytest.mark.parametrize("mood", ["concerned", "annoyed", "sleepy"])
@pytest.mark.parametrize(
    "intent", ["greeting", "explaining", "acknowledging", "uncertain", "conversation"]
)
def test_serious_replies_never_acquire_random_or_playful_actions(mood, intent):
    policy = GesturePolicy(FixedRandom(0))
    assert all(policy.choose("one", intent, mood) is None for _ in range(20))


@pytest.mark.parametrize(
    "intent,gesture,chance", [("acknowledging", "nod", 0.30), ("uncertain", "shrug", 0.35)]
)
def test_optional_actions_have_probability_and_both_time_and_turn_cooldowns(
    intent, gesture, chance
):
    now = [0]
    rng = FixedRandom(0)
    policy = GesturePolicy(rng, lambda: now[0])
    assert policy.choose("one", intent, "neutral") == gesture
    now[0] = 30
    assert all(policy.choose("one", intent, "neutral") is None for _ in range(3))
    assert rng.calls == 1  # No extra rolls while cooling down.
    assert policy.choose("one", intent, "neutral") == gesture
    assert all(policy.choose("one", intent, "neutral") is None for _ in range(5))
    now[0] = 60
    rng.value = chance
    assert policy.choose("one", intent, "neutral") is None
    rng.value = chance - 0.001
    assert policy.choose("one", intent, "neutral") == gesture
    # A new conversation has an independent cooldown.
    assert policy.choose("two", intent, "neutral") == gesture


@pytest.mark.parametrize(
    "roll,expected",
    [(0, "step"), (0.029, "step"), (0.03, "shift"), (0.089, "shift"), (0.09, None), (0.99, None)],
)
def test_casual_movement_is_rare_and_never_on_opening_turns(roll, expected):
    policy = GesturePolicy(FixedRandom(roll))
    assert all(policy.choose("one", "conversation", "neutral") is None for _ in range(3))
    assert policy.choose("one", "conversation", "neutral") == expected
    if expected:
        assert all(policy.choose("one", "conversation", "neutral") is None for _ in range(10))


def test_unknown_intents_and_bashfulness_do_not_produce_shrugs():
    policy = GesturePolicy(FixedRandom(0))
    for intent in (None, "twirl", "run-code", ""):
        assert policy.choose("one", intent, "neutral") is None
    assert policy.choose("one", "uncertain", "embarrassed") is None


@pytest.mark.parametrize(
    "intent", ["greeting", "explaining", "acknowledging", "uncertain", "conversation"]
)
def test_intent_header_can_split_at_every_stream_boundary(intent):
    source = f"[mood=curious] [intent={intent}] 반가워요! こんにちは。 Hello."
    for split in range(len(source) + 1):
        parser = SpeechText()
        shown, spoken = "", []
        for index, chunk in enumerate((source[:split], source[split:], "")):
            visible, clauses = parser.feed(chunk, final=index == 2)
            shown += visible
            spoken += clauses
        assert shown.strip() == "반가워요! こんにちは。 Hello."
        assert spoken == ["반가워요!", "こんにちは。", "Hello."]
        assert parser.expression == "curious"
        assert parser.intent == intent
        assert parser.header_complete


@pytest.mark.parametrize("header", ["", "[intent=invalid] "])
def test_missing_and_invalid_intents_fall_back_without_leaking_to_speech(header):
    parser = SpeechText()
    visible, clauses = parser.feed(f"[mood=neutral] {header}Hello.", final=True)
    assert visible.strip() == "Hello."
    assert clauses == ["Hello."]
    assert parser.intent is None


def test_legacy_reply_without_intent_preserves_literal_brackets():
    parser = SpeechText()
    display = ""
    for char in "[mood=neutral] [2 + 2] equals four.":
        visible, _clauses = parser.feed(char)
        display += visible
    visible, _clauses = parser.feed("", final=True)
    assert (display + visible).strip() == "[2 + 2] equals four."
    assert parser.intent is None


def test_truncated_intent_is_flushed_and_mood_is_available_early():
    parser = SpeechText()
    assert parser.feed("[mood=happy]") == ("", [])
    assert parser.started and not parser.header_complete and parser.expression == "happy"
    assert parser.feed(" [intent=gree") == ("", [])
    assert parser.feed("", final=True) == ("", [])
    assert parser.header_complete and parser.intent is None


def test_repeated_model_metadata_never_leaks_to_text_or_voice_at_any_boundary():
    source = "[mood=annoyed] [intent=conversation] Please stop. [mood=happy] [intent=greeting] That is my lunch. [optional]"
    expected = "Please stop.   That is my lunch. [optional]"
    for size in (1, 2, 5, 12, len(source)):
        parser = SpeechText()
        display, spoken = "", []
        for offset in range(0, len(source), size):
            visible, clauses = parser.feed(source[offset : offset + size])
            display += visible
            spoken += clauses
        visible, clauses = parser.feed("", final=True)
        display += visible
        spoken += clauses
        assert display.strip() == expected
        assert spoken == ["Please stop.", "That is my lunch.", "[optional]"]
        assert parser.expression == "annoyed" and parser.intent == "conversation"


def test_body_metadata_buffer_is_bounded_and_recovers_after_overlong_tag():
    parser = SpeechText()
    parser.feed("[mood=neutral] [intent=conversation] Hello.")
    assert parser.feed("[mood=" + "x" * 1000) == ("", [])
    assert len(parser.body_pending) <= 64
    visible, spoken = parser.feed("] Still here. [intent=unfinished", final=True)
    assert visible == " Still here. "
    assert spoken == ["Still here."]


@pytest.mark.parametrize("action", AVATAR_ACTIONS)
def test_explicit_actions_bypass_optional_randomness_and_cooldowns(action):
    rng = FixedRandom(1)
    policy = GesturePolicy(rng)
    for _ in range(5):
        assert policy.choose("one", "conversation", "happy", action) == action
    assert rng.calls == 0


@pytest.mark.parametrize("action", ["dance", "laugh", "invalid", "run-code"])
def test_unsupported_actions_fail_closed_even_with_greeting_intent(action):
    assert GesturePolicy().choose("one", "greeting", "happy", action) is None


def test_requested_nod_is_allowed_but_serious_replies_stay_still():
    policy = GesturePolicy()
    assert policy.choose("one", "conversation", "neutral", "nod") == "nod"
    for mood in ("concerned", "annoyed", "sleepy"):
        for action in AVATAR_ACTIONS:
            assert policy.choose("one", "conversation", mood, action) is None


@pytest.mark.parametrize("action", [*AVATAR_ACTIONS, "none", "dance"])
def test_action_header_is_parsed_once_and_never_spoken_at_any_stream_split(action):
    source = (
        f"[mood=happy] [intent=conversation] [action={action}] One little spin! [action=wave] 하하!"
    )
    for split in range(len(source) + 1):
        parser = SpeechText()
        display, spoken = "", []
        for i, chunk in enumerate([source[:split], source[split:], ""]):
            visible, ready = parser.feed(chunk, final=i == 2)
            display += visible
            spoken += ready
        assert display.strip() == "One little spin!  하하!"
        assert spoken == ["One little spin!", "하하!"]
        assert parser.action == (action if action != "dance" else "invalid")
        assert parser.header_complete


def test_action_header_waits_until_complete_and_truncated_metadata_is_suppressed():
    parser = SpeechText()
    parser.feed("[mood=happy] [intent=conversation]")
    assert not parser.header_complete
    assert parser.feed(" [action=tw") == ("", [])
    assert not parser.header_complete
    assert parser.feed("irl] Yay!") == ("Yay!", ["Yay!"])
    assert parser.action == "twirl" and parser.header_complete
    for size in (3, 1000):
        parser = SpeechText()
        visible, _ = parser.feed("[mood=happy] [intent=conversation] [action=" + "x" * size)
        final, _ = parser.feed("", final=True)
        assert not visible and not final
