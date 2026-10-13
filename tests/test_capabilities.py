import asyncio
import json

import pytest
from companion.adapters import DemoDialogue, DemoSpeech
from companion.app import create_app
from companion.capabilities import AVATAR_ACTIONS, allows_avatar_request
from companion.config import Config
from companion.protocol import Command, decode_audio
from companion.session import Session
from companion.store import Store

ACTION_REQUESTS = {
    "wave": "Can you wave to me?",
    "nod": "Give me a gentle nod.",
    "explain": "Explain why the sky is blue.",
    "shift": "Shift your weight a little.",
    "step": "Take a small step.",
    "twirl": "Give me a cute twirl.",
    "raise-arms": "Raise your arms.",
    "shrug": "Show me a shrug.",
    "point": "Point over there.",
}


@pytest.mark.parametrize(
    "action,text",
    [
        *ACTION_REQUESTS.items(),
        ("twirl", "한 바퀴 돌아 줘!"),
        ("wave", "손 흔들어 줘"),
        ("twirl", "춤춰 줘"),
        ("wave", "手を振って。"),
        ("twirl", "踊って！"),
        ("raise-arms", "両腕を上げてみて！"),
        ("nod", "고개 끄덕여 줘"),
        ("nod", "うなずいてみて"),
        ("shrug", "어깨 으쓱해 봐"),
        ("shrug", "肩をすくめて"),
        ("point", "저쪽을 가리켜 줘"),
        ("point", "あそこを指さして"),
    ],
)
def test_direct_avatar_requests_are_allowed(action, text):
    assert allows_avatar_request(text, action)


@pytest.mark.parametrize(
    "text",
    [
        "Translate this into Korean: 'Give me a twirl.'",
        "Don't twirl.",
        "Do not wave.",
        "Tell me a story about a dancer.",
        "'춤춰 줘'를 영어로 번역해 줘",
        "춤추지 마",
        "손 흔들지 말아 줘",
        "踊らないで",
        "「手を振って」を翻訳して。",
    ],
)
def test_quoted_translated_and_negated_actions_are_vetoed(text):
    assert not any(allows_avatar_request(text, action) for action in AVATAR_ACTIONS)


@pytest.mark.parametrize(
    "text",
    [
        "Can you laugh?",
        "아리, 한번 웃어 줘!",
        "ちょっと笑ってみて！",
        "Can you physically come to my house and make me tea?",
    ],
)
def test_unrelated_model_actions_are_vetoed(text):
    assert not any(allows_avatar_request(text, action) for action in AVATAR_ACTIONS)


@pytest.mark.asyncio
@pytest.mark.parametrize("speak", [False, True])
@pytest.mark.parametrize("action", [*AVATAR_ACTIONS, "dance", "laugh"])
async def test_requested_action_crosses_session_once_and_metadata_never_reaches_voice(
    tmp_path, action, speak
):
    events, spoken, prompts = [], [], []

    class Dialogue(DemoDialogue):
        async def generate(self, messages, profile):
            prompts.append(messages[0]["content"])
            for char in f"[mood=happy] [intent=conversation] [action={action}] Here goes! [action=wave] All done.":
                yield char

    class Speech(DemoSpeech):
        async def generate(self, text, *args):
            spoken.append(text)
            yield bytes(480)

    async def send(event):
        events.append(event)

    async def audio(packet):
        turn, _, offset, pcm = decode_audio(packet)
        session.acknowledge(turn, offset + len(pcm) // 2)

    store = Store(tmp_path / "history.sqlite")
    saved = {"name": "Ari", "persona": "Be calm. Never claim actions outside this conversation."}
    store.set_setting("persona:ari", saved)
    session = Session(
        Config(data=tmp_path, token="x" * 64, demo=True), store, send, audio, Dialogue(), Speech()
    )
    session.avatar_mode = "model"
    try:
        await session.reply(ACTION_REQUESTS.get(action, f"Can you {action}?"), speak=speak)
        await asyncio.wait_for(session.task, 2)
        assert not [e for e in events if e["type"] == "error"]
        gestures = [e for e in events if e["type"] == "gesture"]
        assert [e["gesture"] for e in gestures] == ([action] if action in AVATAR_ACTIONS else [])
        if gestures:
            assert gestures[0]["turn_id"] == session.turn_id
            assert gestures[0]["wait_for_audio"] == speak
            assert events.index(gestures[0]) < next(
                i for i, e in enumerate(events) if e["type"] == "text"
            )
        assert session.generated.strip() == "Here goes!  All done."
        assert spoken == (["Here goes!", "All done."] if speak else [])
        assert all(
            "[action=" not in row["generated"] for row in store.messages(session.conversation)
        )
        assert "Available avatar actions NOW:" in prompts[0]
        choices = next(line for line in prompts[0].splitlines() if "Allowed action tags" in line)
        if action in AVATAR_ACTIONS:
            assert f"[action={action}]" in choices
        else:
            assert "[action=none]" in choices
            assert f"[action={action}]" not in choices
        if action not in {"twirl", "dance"}:
            assert "[action=twirl]" not in choices
        assert ("This reply will be spoken" if speak else "Audio is off or unavailable") in prompts[
            0
        ]
        assert store.setting("persona:ari") == saved, "custom persona is not overwritten"
    finally:
        await session.close()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,visible,reduced",
    [("illustration", True, False), ("model", False, False), ("model", True, True)],
)
async def test_unavailable_view_does_not_advertise_or_emit_actions(
    tmp_path, mode, visible, reduced
):
    events = []

    class Dialogue(DemoDialogue):
        async def generate(self, messages, profile):
            assert "Body actions are unavailable" in messages[0]["content"]
            # Even a model ignoring that restriction cannot trigger movement.
            yield "[mood=happy] [intent=conversation] [action=twirl] Here goes!"

    async def send(event):
        events.append(event)

    store = Store(tmp_path / "history.sqlite")
    session = Session(
        Config(data=tmp_path, token="x" * 64, demo=True),
        store,
        send,
        send,
        Dialogue(),
        DemoSpeech(),
    )
    command = Command(
        type="avatar", avatar_mode=mode, avatar_visible=visible, reduced_motion=reduced
    )
    session.avatar_mode, session.avatar_visible, session.reduced_motion = (
        command.avatar_mode,
        command.avatar_visible,
        command.reduced_motion,
    )
    try:
        await session.reply("Can you twirl?", speak=False)
        await session.task
        assert not [e for e in events if e["type"] in {"gesture", "error"}]
        session.avatar_mode, session.avatar_visible, session.reduced_motion = "model", True, False
        assert session.available_actions == AVATAR_ACTIONS
        session.character_id = "another-character"
        assert session.available_actions == ()
    finally:
        await session.close()
        store.close()


@pytest.mark.asyncio
async def test_avatar_command_updates_session_capabilities_before_next_text(tmp_path, monkeypatch):
    observed = []

    async def record_reply(self, text, speak):
        observed.append(self.available_actions)

    monkeypatch.setattr(Session, "reply", record_reply)
    config = Config(data=tmp_path, token="x" * 64, demo=True)
    app = create_app(config)
    endpoint = next(route.endpoint for route in app.routes if route.path == "/session")
    commands = iter(
        [
            {"type": "text", "text": "No capability handshake yet."},
            {"type": "avatar", "avatar_mode": "model", "avatar_visible": True},
            {"type": "text", "text": "Can you twirl?"},
            {"type": "avatar", "avatar_mode": "model", "avatar_visible": False},
            {"type": "text", "text": "Can you twirl while hidden?"},
        ]
    )

    class Socket:
        def __init__(self):
            self.headers = {"authorization": "Bearer " + config.token}
            self.query_params = {}

        async def accept(self):
            pass

        async def receive(self):
            value = next(commands, None)
            return (
                {"type": "websocket.receive", "text": json.dumps(value)}
                if value
                else {"type": "websocket.disconnect"}
            )

        async def send_json(self, event):
            assert event["type"] != "error", event

    try:
        await endpoint(Socket())
        assert observed == [(), AVATAR_ACTIONS, ()]
    finally:
        app.state.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_text",
    [
        "Translate this, don't perform it: 'Give me a twirl.'",
        "아리, 한번 웃어 줘!",
        "Can you physically come to my house and make me tea?",
    ],
)
async def test_model_action_cannot_override_user_request(tmp_path, user_text):
    events = []

    class Dialogue(DemoDialogue):
        async def generate(self, messages, profile):
            yield "[mood=happy] [intent=conversation] [action=twirl] 한 바퀴 돌아줘."

    async def send(event):
        events.append(event)

    store = Store(tmp_path / "history.sqlite")
    session = Session(
        Config(data=tmp_path, token="x" * 64, demo=True),
        store,
        send,
        send,
        Dialogue(),
        DemoSpeech(),
    )
    session.avatar_mode = "model"
    try:
        await session.reply(user_text, speak=False)
        await session.task
        assert not [e for e in events if e["type"] in {"gesture", "error"}]
        assert session.generated == "한 바퀴 돌아줘."
    finally:
        await session.close()
        store.close()
