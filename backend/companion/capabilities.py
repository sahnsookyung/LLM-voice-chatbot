"""Capabilities of the shipped companion, not the language model in isolation."""

import re

AVATAR_ACTIONS = (
    "wave",
    "nod",
    "explain",
    "shift",
    "step",
    "twirl",
    "raise-arms",
    "shrug",
    "point",
)


_ACTION_WORDS = {
    "wave": r"\bwave\b|손.*흔들|手.*振っ",
    "nod": r"\bnod\b|고개.*끄덕|頷|うなず",
    "explain": r"\bexplain\b|설명|説明",
    "shift": r"\bshift\b|체중.*옮|무게.*옮|重心.*移",
    "step": r"\bstep\b|한\s*발|한\s*걸음|一歩",
    "twirl": r"\b(?:twirl|spin|pirouette|dance)\b|춤|한\s*바퀴|돌아\s*줘|踊|ダンス|くるり|回って",
    "raise-arms": r"\b(?:raise|lift).*(?:arms|hands)\b|팔.*(?:올|들어)|손.*들어|(?:腕|手).*上げ",
    "shrug": r"\bshrug\b|어깨.*으쓱|肩.*すく",
    "point": r"\bpoint\b|가리켜|指.*(?:差|さし|さして)",
}


def allows_avatar_request(text, action):
    """Require matching movement words before trusting an explicit model tag.

    This is not a language classifier. Ambiguous mixed requests can be retried
    as a direct command; an unwanted movement is worse than a missed one.
    """
    pattern = _ACTION_WORDS.get(action)
    if not pattern:
        return False
    text = text.replace("’", "'")
    blocked = re.search(
        r"\b(?:translate|translation|quote|story|meaning|don't|do not|never|stop)\b"
        r"|번역|무슨 뜻|이야기|하지\s*말|지\s*마|말아"
        r"|翻訳|訳して|意味|物語|しないで|やめて|ないで",
        text,
        re.IGNORECASE,
    )
    return not blocked and bool(re.search(pattern, text, re.IGNORECASE))


def capability_prompt(actions, speaking, requested_actions=()):
    voice = (
        "This reply will be spoken by the app's synthesized voice."
        if speaking
        else "Audio is off or unavailable for this reply; do not claim the user can hear it."
    )
    movement = (
        f"Available avatar actions NOW: {', '.join(actions)}. "
        "To perform one, put its exact ID in the action tag. A twirl is one cute spin. "
        "For a dance request, say briefly that you can do a twirl for now and select twirl; "
        "do not pretend a full dance routine exists."
        if actions
        else "Body actions are unavailable in the current view (hidden/illustration/reduced "
        "motion or a different character). Use action=none. For movement requests, explain "
        "briefly that visible Ari 3D with motion enabled is needed; do not claim to move."
    )
    examples = ""
    if "twirl" in requested_actions:
        examples += (
            "User: Give me a twirl. Reply: [mood=happy] [intent=conversation] [action=twirl] One little spin!\n"
            "User: 춤춰 줘. Reply: [mood=happy] [intent=conversation] [action=twirl] 아직 춤 대신 한 바퀴 돌 수 있어요!\n"
        )
    if "wave" in requested_actions:
        examples += (
            "User: 手を振って。 Reply: [mood=happy] [intent=conversation] [action=wave] はーい！\n"
        )
    if not actions:
        examples += (
            "User: Give me a twirl. Reply: [mood=neutral] [intent=conversation] [action=none] "
            "Show my 3D avatar with motion enabled, and I can do a twirl for you.\n"
        )
    return (
        "CURRENT APP CAPABILITIES:\n"
        "You are a virtual character with an on-screen avatar and synthesized speech. "
        "Speak naturally as the character. Being virtual permits expressive acting; "
        "it does not make you text-only. " + voice + " " + movement + "\n"
        "For a friendly request to laugh, give one brief natural chuckle with mood=happy "
        "and action=none, for example Haha! The speech engine can synthesize a chuckle "
        "and the avatar can smile; a dedicated laugh-body animation is not implemented. "
        "The app supplies speech delivery from your mood tag; do not write audio-effect "
        "tags or stage directions in the spoken words. Speak in first person. "
        "Ordinary replies need no laughter, technical explanation or identity disclaimer. "
        "If asked, be honest that you are virtual. You cannot browse the web, control other "
        "apps or perform real-world physical actions. Older assistant claims of being "
        "unable to speak or animate are outdated.\n"
        "After mood and intent, add exactly one action tag. Choose a supported action "
        "only for a direct request to perform it now, including 'can you wave?'. "
        "Quotes, stories, negations and general discussion use action=none. Select at most "
        "one action; never promise an unimplemented action or a sequence. The app handles "
        "ordinary conversational gestures itself. Concerned, annoyed and sleepy replies "
        "always use action=none. Use mood=happy for lighthearted acting, never mood=playful.\n"
        "Examples for the CURRENT view:\n"
        "User: Can you laugh? Reply: [mood=happy] [intent=conversation] [action=none] Haha!\n"
        + examples
        + "Only actions matching the user's request are permitted for this reply. "
        + f"Allowed action tags for THIS reply: {', '.join(f'[action={a}]' for a in ('none', *requested_actions))}.\n"
    )
