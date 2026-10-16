"""Incremental Fish adapter, derived from mlx-audio 94c7716 (MIT; see licenses/mlx-audio-MIT.txt).

Kept in the repository, never patched into site-packages. Sampling and token
limits match the pinned provider. Codec state is fresh for each provider segment.
"""

import copy
import hashlib

import mlx.core as mx
import numpy as np
from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import (
    IM_END_TOKEN,
    RAS_WIN_SIZE,
    Conversation,
    Message,
    TextPart,
    VQPart,
    _sample_logits,
)

from .fish_codec_stream import CodecStream
from .resampling import StreamingResampler


class PrefixCache:
    """One immutable reference/instruction prefix, at most 128 MiB of KV tensors."""

    def __init__(self):
        self.key = self.saved = None

    def prefill(self, owner, conversation, prompt):
        prefix = Conversation(list(conversation.messages[:-1])).encode_for_inference(
            owner.tokenizer, num_codebooks=owner.model.num_codebooks
        )[None]
        length = prefix.shape[-1]
        # Only reuse a byte-identical prefix, including reference and instructions.
        if not bool(mx.all(prompt[..., :length] == prefix).item()):
            cache = owner.model.make_cache()
            return owner.model(prompt, cache=cache), cache
        key = id(owner.model), hashlib.sha256(np.asarray(prefix).tobytes()).digest()
        if key != self.key:
            cache = owner.model.make_cache()
            owner.model(prefix, cache=cache)
            states = [item.state for item in cache]
            mx.eval(states)
            size = sum(value.nbytes for state in states for value in state if value is not None)
            self.saved = copy.deepcopy(cache) if size <= 128 * 1024 * 1024 else None
            self.key = key if self.saved is not None else None
        else:
            cache = copy.deepcopy(self.saved)
        return owner.model(prompt[..., length:], cache=cache), cache


def make_fast_decode(model):
    """Compile one residual frame; all nine sampling steps retain their RNG state.

    The fast cache is private to a single semantic frame in the upstream model.
    Compiling that same bounded loop reduces CPU/GPU dispatch without changing
    the transformer weights, sampling settings, or semantic token limit.
    """

    def frame(hidden, semantic, temperature, top_p, top_k):
        cache = model.make_fast_cache()
        model.fast_forward_cached(hidden, cache)
        value = model.fast_embeddings(semantic)
        tokens = [semantic]
        for _ in range(model.num_codebooks - 1):
            logits = model.fast_forward_cached(value, cache)
            token = _sample_logits(logits, temperature=temperature, top_p=top_p, top_k=top_k)
            tokens.append(token)
            value = model.fast_embeddings(token)
        return mx.stack(tokens, axis=1)

    return mx.compile(frame, inputs=mx.random.state, outputs=mx.random.state)


def code_batches(
    self,
    conversation: Conversation,
    batch_text: str,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    temperature: float,
):
    if self.tokenizer is None:
        raise ValueError("Tokenizer not loaded. Call post_load_hook first.")

    prompt_conversation = Conversation(list(conversation.messages))
    prompt_conversation.append(
        Message(
            role="assistant",
            parts=[],
            modality="voice",
            add_im_start=True,
            add_im_end=False,
        )
    )
    prompt = prompt_conversation.encode_for_inference(
        self.tokenizer, num_codebooks=self.model.num_codebooks
    )
    prompt = prompt[None, :, :]

    if not hasattr(self, "_ari_prefix_cache"):
        self._ari_prefix_cache = PrefixCache()
    result, cache = self._ari_prefix_cache.prefill(self, conversation, prompt)
    logits = result.logits[:, -1]
    hidden_state = result.hidden_states[:, -1]

    previous_semantic_tokens: list[int] = []
    generated_steps = []
    produced = False
    im_end_id = self.tokenizer.get_token_id(IM_END_TOKEN)
    text_token_count = len(self.tokenizer.encode(batch_text))
    semantic_token_budget = min(
        max_new_tokens,
        max(32, text_token_count * 12),
    )

    for _ in range(semantic_token_budget):
        semantic_token = self._sample_semantic(
            logits=logits,
            previous_semantic_tokens=previous_semantic_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
        semantic_token_id = int(semantic_token[0].item())
        if semantic_token_id == im_end_id:
            break

        previous_semantic_tokens.append(semantic_token_id)
        previous_semantic_tokens = previous_semantic_tokens[-RAS_WIN_SIZE:]

        semantic_code = (semantic_token - self.config.semantic_start_token_id).astype(mx.int32)
        semantic_code = mx.clip(semantic_code, 0, self.config.audio_decoder_config.vocab_size - 1)
        if not hasattr(self, "_ari_fast_decode"):
            self._ari_fast_decode = make_fast_decode(self.model)
        previous_codebooks = self._ari_fast_decode(
            hidden_state, semantic_code, temperature, top_p, top_k
        )

        generated_steps.append(previous_codebooks[0])
        produced = True
        if len(generated_steps) == 4:
            yield mx.stack(generated_steps, axis=1).astype(mx.int32)
            generated_steps.clear()

        next_input = mx.concatenate(
            [semantic_token[:, None].astype(mx.int32), previous_codebooks], axis=1
        )
        next_result = self.model(next_input[:, :, None], cache=cache)
        logits = next_result.logits[:, -1]
        hidden_state = next_result.hidden_states[:, -1]

    if not produced:
        raise RuntimeError("No audio tokens were generated")
    if generated_steps:
        yield mx.stack(generated_steps, axis=1).astype(mx.int32)


def stream_pcm(model, text, ref_audio, ref_text, instruct=None, weights=None):
    prompt_texts, prompt_tokens = model._prepare_reference_prompt(ref_audio, ref_text)
    conversation = model._build_conversation(prompt_texts, prompt_tokens, instruct=instruct)
    for batch in model._split_generation_text(text, 300):
        conversation.append(
            Message(role="user", parts=[TextPart(batch)], add_im_start=True, add_im_end=True)
        )
        codec = CodecStream(model.codec, weights)
        resampler = StreamingResampler(model.sample_rate, 24000)
        pending = bytearray()
        codes = []
        for frames in code_batches(model, conversation, batch, 512, 0.7, 30, 0.7):
            codes.append(frames)
            samples = resampler.feed(np.asarray(codec.decode(frames[None])))
            pending.extend((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
            while len(pending) >= 3840:
                yield bytes(pending[:3840])
                del pending[:3840]
        codec.flush()
        samples = resampler.feed([], final=True)
        pending.extend((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
        while pending:
            yield bytes(pending[:3840])
            del pending[:3840]
        conversation.append(
            Message(
                role="assistant",
                parts=[VQPart(mx.concatenate(codes, axis=1))],
                modality="voice",
                add_im_start=True,
                add_im_end=True,
            )
        )
