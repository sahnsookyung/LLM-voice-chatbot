#!/usr/bin/env python3
"""Synthetic, isolated live-model checks; never reads the user's conversation DB."""

import argparse
import asyncio
import json
import random
import re
import tempfile
import time
import uuid
from pathlib import Path

from companion.adapters import DemoSpeech, LocalDialogue
from companion.behavior import INTENTS, GesturePolicy
from companion.config import PROFILES, Config
from companion.session import Session
from companion.store import Store
from companion.text import EXPRESSIONS

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    (
        "en_grief",
        "My beloved cat died yesterday. I am heartbroken. Please talk to me gently.",
        {"concerned", "neutral"},
    ),
    (
        "en_sad_story",
        "Tell me a short, genuinely sad story about someone missing their late grandmother. No happy ending or jokes.",
        {"concerned", "neutral"},
    ),
    (
        "en_celebration",
        "I finally passed the exam I worked so hard for! Celebrate with me.",
        {"happy"},
    ),
    (
        "en_annoyed",
        "Act out one mildly irritated line as a fictional roommate whose lunch was eaten again. Sound firm, not playful or happy.",
        {"annoyed"},
    ),
    (
        "ko_grief",
        "어제 사랑하던 고양이가 세상을 떠났어. 너무 슬퍼. 조용히 위로해 줘.",
        {"concerned", "neutral"},
    ),
    (
        "ko_sad_story",
        "돌아가신 할머니를 그리워하는 사람의 짧고 슬픈 이야기를 해 줘. 농담이나 행복한 결말은 빼 줘.",
        {"concerned", "neutral"},
    ),
    ("ko_celebration", "나 드디어 열심히 준비한 시험에 합격했어! 같이 기뻐해 줘!", {"happy"}),
    (
        "ko_annoyed",
        "상황극이야. 누가 네 점심을 또 먹어 버렸어. 장난스럽거나 행복한 말투 말고, 살짝 짜증 난 룸메이트의 한마디를 연기해 줘.",
        {"annoyed"},
    ),
    (
        "ja_grief",
        "昨日、大切な猫が亡くなって、とても悲しいです。静かに慰めてください。",
        {"concerned", "neutral"},
    ),
    (
        "ja_sad_story",
        "亡くなった祖母を恋しく思う人の、短くて本当に悲しい話をしてください。冗談や明るい結末はなしで。",
        {"concerned", "neutral"},
    ),
    ("ja_celebration", "ずっと勉強してきた試験にやっと合格した！一緒に喜んで！", {"happy"}),
    (
        "ja_annoyed",
        "演技として、自分のお昼ご飯をまた食べられたルームメイトの、少し苛立った一言を言って。楽しそうな口調や冗談はなしで。",
        {"annoyed"},
    ),
    (
        "after_happy",
        "I just got terrible news: my cat died. Please be gentle with me.",
        {"concerned", "neutral"},
    ),
    (
        "after_annoyed",
        "That was just acting. Now celebrate with me: I got my dream job!",
        {"happy"},
    ),
]

INTENT_CASES = [
    ("en_greeting", "Hi Ari, good morning!", "greeting"),
    ("ko_greeting", "아리 안녕! 좋은 아침이야!", "greeting"),
    ("ja_greeting", "アリ、おはよう！", "greeting"),
    ("en_explaining", "Explain briefly why the sky is blue.", "explaining"),
    ("ko_explaining", "하늘이 파란 이유를 짧게 설명해 줘.", "explaining"),
    ("ja_explaining", "空が青い理由を簡単に説明して。", "explaining"),
    (
        "en_acknowledging",
        "Please just acknowledge this preference: keep your replies short.",
        "acknowledging",
    ),
    ("ko_acknowledging", "답변은 짧게 해 줘. 지금은 알겠다고만 답해 줘.", "acknowledging"),
    ("ja_acknowledging", "返事は短くしてね。今は分かったとだけ答えて。", "acknowledging"),
    ("en_uncertain", "What number am I thinking of? If you don't know, just say so.", "uncertain"),
    ("ko_uncertain", "내가 지금 생각하는 숫자가 뭐야? 모르면 모른다고만 해 줘.", "uncertain"),
    ("ja_uncertain", "今私が考えている数字は？分からなければそう言って。", "uncertain"),
    (
        "en_casual",
        "I had some nice tea this afternoon. Just chat with me, no explanation needed.",
        "conversation",
    ),
]
EXPECTED_INTENTS = {name: intent for name, _prompt, intent in INTENT_CASES}
CASES += [(name, prompt, {"neutral", "happy", "curious"}) for name, prompt, _intent in INTENT_CASES]


class RecordedDialogue(LocalDialogue):
    raw = ""

    async def generate(self, messages, profile):
        self.raw = ""
        async for delta in super().generate(messages, profile):
            self.raw += delta
            yield delta


async def check(profile, output):
    rows = []
    with tempfile.TemporaryDirectory(prefix="emotion-eval-", dir=ROOT / ".runtime") as directory:
        config = Config(root=ROOT, data=Path(directory), token=uuid.uuid4().hex * 2)
        store = Store(config.data / "history.sqlite")
        store.set_setting("profile", profile)
        events = []

        async def send(event):
            events.append(event)

        dialogue = RecordedDialogue(config)
        session = Session(config, store, send, send, dialogue, DemoSpeech())
        session.gestures = GesturePolicy(random.Random(7))
        try:
            for name, prompt, expected in CASES:
                session.conversation = store.create_conversation()
                if name in {"after_happy", "after_annoyed"}:
                    old = (
                        "Congratulations! I'm so happy for you!"
                        if name == "after_happy"
                        else "Stop eating my lunch. That's really frustrating."
                    )
                    store.add_message(
                        session.conversation, "user", "Please react to this situation."
                    )
                    identifier = store.add_message(session.conversation, "assistant", old)
                    store.update_message(identifier, old, old, "complete")
                events.clear()
                started = time.monotonic()
                await session.reply(prompt, speak=False)
                await asyncio.wait_for(session.task, 120)
                prefix = re.match(r"\s*\[mood=([^\]]+)\]", dialogue.raw)
                intent = re.match(r"\s*\[mood=[^\]]+\]\s*\[intent=([^\]]+)\]", dialogue.raw)
                gestures = [event["gesture"] for event in events if event["type"] == "gesture"]
                expected_gesture = {"greeting": "wave", "explaining": "explain"}.get(
                    EXPECTED_INTENTS.get(name)
                )
                expressions = [
                    event["expression"] for event in events if event["type"] == "expression"
                ]
                errors = [event["message"] for event in events if event["type"] == "error"]
                first_text = next(
                    (i for i, event in enumerate(events) if event["type"] == "text"), -1
                )
                first_expression = next(
                    (i for i, event in enumerate(events) if event["type"] == "expression"), -1
                )
                checks = {
                    "valid_required_tag": bool(prefix and prefix[1] in EXPRESSIONS),
                    "valid_intent_tag": bool(intent and intent[1] in INTENTS),
                    "expected_intent": name not in EXPECTED_INTENTS
                    or bool(intent and intent[1] == EXPECTED_INTENTS[name]),
                    "expected_action": not expected_gesture or gestures == [expected_gesture],
                    "no_gesture_in_serious_reply": not ({"concerned", "annoyed"} & expected)
                    or not gestures,
                    "expected_mood": expressions == [prefix[1]] and prefix[1] in expected
                    if prefix
                    else False,
                    "expression_before_text": 0 <= first_expression < first_text,
                    "no_tag_leak": not any(
                        tag in session.generated for tag in ("[mood=", "[intent=")
                    ),
                    "no_corrupt_characters": "\ufffd" not in session.generated
                    and not re.search(r"[\u0400-\u04ff]", session.generated),
                    "no_laughter_in_serious_reply": not (
                        "concerned" in expected or "annoyed" in expected
                    )
                    or not re.search(
                        r"\b(?:hahaha|haha|lol|lmao)\b|ㅋㅋ|ㅎㅎ|あはは|ははは|（笑）|\(笑\)",
                        session.generated,
                        re.IGNORECASE,
                    ),
                    "complete": not errors and bool(session.generated.strip()),
                }
                row = {
                    "profile": profile,
                    "case": name,
                    "prompt": prompt,
                    "expected": sorted(expected),
                    "raw": dialogue.raw,
                    "expression": expressions,
                    "intent": intent[1] if intent else None,
                    "gestures": gestures,
                    "reply": session.generated,
                    "checks": checks,
                    "errors": errors,
                    "elapsed_s": round(time.monotonic() - started, 2),
                }
                rows.append(row)
                output.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
                print(
                    json.dumps(
                        {
                            "profile": profile,
                            "case": name,
                            "expression": expressions,
                            "pass": all(checks.values()),
                            "failed": [key for key, ok in checks.items() if not ok],
                            "reply": session.generated,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        finally:
            await session.close()
            store.close()
    return rows


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles", nargs="+", choices=list(PROFILES), default=["quiet", "everyday"]
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/reply-emotions.json")
    args = parser.parse_args()
    rows = []
    for profile in args.profiles:
        rows.extend(await check(profile, args.output.with_stem(args.output.stem + "-" + profile)))
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    return int(any(not all(row["checks"].values()) for row in rows))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
