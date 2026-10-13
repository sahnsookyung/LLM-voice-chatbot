"""Bounded conversational gestures; no extra model call or animation timer."""

import random
import time

INTENTS = {"greeting", "explaining", "acknowledging", "uncertain", "conversation"}


class GesturePolicy:
    def __init__(self, rng=None, clock=time.monotonic):
        self.rng = rng if rng is not None else random.Random()
        self.clock = clock
        self.conversation = None
        self.turn = 0
        self.last_optional_turn = -4
        self.last_optional_time = -float("inf")

    def choose(self, conversation, intent, expression):
        if conversation != self.conversation:
            self.conversation = conversation
            self.turn = 0
            self.last_optional_turn = -4
            self.last_optional_time = -float("inf")
        self.turn += 1
        # A missing/invalid intent fails quietly. Empathy and irritation must not
        # acquire a cheerful wave, a dismissive shrug, or random footwork.
        if intent not in INTENTS or expression in {"concerned", "annoyed", "sleepy"}:
            return None
        if intent == "greeting":
            return "wave"
        if intent == "explaining":
            return "explain"
        now = self.clock()
        if self.turn - self.last_optional_turn < 4 or now - self.last_optional_time < 20:
            return None
        gesture = None
        if intent == "acknowledging":
            gesture = "nod" if self.rng.random() < 0.30 else None
        elif intent == "uncertain" and expression in {"neutral", "curious"}:
            gesture = "shrug" if self.rng.random() < 0.35 else None
        elif (
            intent == "conversation"
            and expression in {"neutral", "happy", "curious"}
            and self.turn >= 4
        ):
            roll = self.rng.random()
            gesture = "step" if roll < 0.03 else "shift" if roll < 0.09 else None
        if gesture:
            self.last_optional_turn = self.turn
            self.last_optional_time = now
        return gesture
