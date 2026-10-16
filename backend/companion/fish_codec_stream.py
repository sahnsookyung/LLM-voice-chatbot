"""Stateful decoder for mlx-audio 94c7716's causal Fish S1 DAC.

This adapter keeps bounded convolution overlap and attention KV history. It
decodes each code frame once; it does not replay an ever-growing utterance.
Equations and layer ordering follow mlx-audio's MIT; see licenses/mlx-audio-MIT.txt implementation.
"""

import math

import mlx.core as mx
from mlx_audio.codec.models.fish_s1_dac.fish_s1_dac import _normalize_weight, apply_rotary_emb


class CodecStream:
    def __init__(self, codec, weights=None):
        self.codec = codec
        self.state = {}
        self.weights = weights if weights is not None else {}
        self.frames = 0

    def weight(self, module):
        key = id(module)
        if key not in self.weights:
            self.weights[key] = (
                module.weight_g * module.weight_v / _normalize_weight(module.weight_v)
            )
            mx.eval(self.weights[key])
        return self.weights[key]

    def call(self, m, x):
        name = type(m).__name__
        if name in {"CausalWNConv1d", "CausalWNConvTranspose1d"}:
            # Fold once for inference; never mutate the provider's loaded weights.
            return self.convolve(m.conv, x, self.weight(m), m.bias)
        if name in {"CausalConvNet", "CausalTransConvNet"}:
            return self.convolve(m, x, m.conv.weight, m.conv.bias)
        if name == "WNConv1d":
            if m.kernel_size != 1 or m.stride != 1:
                raise ValueError("Unexpected noncausal convolution in streaming decoder")
            y = mx.conv1d(x.swapaxes(1, 2), self.weight(m).transpose(0, 2, 1), groups=m.groups)
            if m.bias is not None:
                y = y + m.bias
            return y.swapaxes(1, 2)
        if name in {"Decoder", "DecoderBlock", "ResidualUnit"}:
            y = x
            for child in m.model if name == "Decoder" else m.block:
                y = self.call(child, y)
            return x + y if name == "ResidualUnit" else y
        if name == "ConvNeXtBlock":
            y = self.call(m.dwconv, x).swapaxes(1, 2)
            y = m.pwconv2(m.act(m.pwconv1(m.norm(y))))
            if m.gamma is not None:
                y = m.gamma * y
            return x + y.swapaxes(1, 2)
        if name == "WindowLimitedTransformer":
            return self.transformer(m, x)
        if name in {"Snake1d", "Identity", "Tanh"}:
            return m(x)
        raise ValueError(f"Unsupported streaming codec layer: {name}")

    def convolve(self, m, x, weight, bias):
        c, key = m.conv, id(m)
        if type(m).__name__ == "CausalConvNet":
            if c.stride != 1:
                raise ValueError("Decoder convolution must have unit stride")
            history = self.state.get(key)
            if history is None:
                history = mx.zeros((*x.shape[:-1], m.padding), dtype=x.dtype)
            joined = mx.concatenate([history, x], axis=-1)
            y = mx.conv1d(
                joined.swapaxes(1, 2),
                weight.transpose(0, 2, 1),
                dilation=c.dilation,
                groups=c.groups,
            )
            self.state[key] = joined[..., -m.padding :] if m.padding else joined[..., :0]
            if bias is not None:
                y = y + bias
            return y.swapaxes(1, 2)
        y = mx.conv_transpose1d(
            x.swapaxes(1, 2),
            weight.transpose(1, 2, 0),
            stride=c.stride,
            dilation=c.dilation,
            groups=c.groups,
        ).swapaxes(1, 2)
        overlap = self.state.get(key)
        if overlap is not None and overlap.shape[-1]:
            y = mx.concatenate(
                [y[..., : overlap.shape[-1]] + overlap, y[..., overlap.shape[-1] :]], axis=-1
            )
        length = x.shape[-1] * c.stride
        self.state[key] = y[..., length:]
        y = y[..., :length]
        if bias is not None:
            y = y + bias[None, :, None]
        return y

    def transformer(self, m, x):
        if not m.causal or type(m.look_ahead_conv).__name__ != "Identity":
            raise ValueError("Codec lookahead requires a different streaming contract")
        if m.channels_first:
            x = x.swapaxes(1, 2)
        x = m.input_proj(x)
        position, caches = self.state.get(id(m), (0, [None] * len(m.layers)))
        n = x.shape[1]
        row = mx.arange(position, position + n)
        # Compute only the new RoPE positions; do not wrap a fixed table on long audio.
        freqs = 1.0 / (
            m.config.rope_base
            ** (mx.arange(0, m.config.head_dim, 2, dtype=mx.float32) / m.config.head_dim)
        )
        angle = row.astype(mx.float32)[:, None] * freqs[None, :]
        rope = mx.stack([mx.cos(angle), mx.sin(angle)], axis=-1)
        window = m.window_size or m.config.block_size
        for i, layer in enumerate(m.layers):
            a = layer.attention
            norm = layer.attention_norm(x)
            qkv = a.wqkv(norm)
            size = a.n_local_heads * a.head_dim
            q = qkv[..., :size].reshape(x.shape[0], n, a.n_head, a.head_dim)
            k = qkv[..., size : 2 * size].reshape(x.shape[0], n, a.n_local_heads, a.head_dim)
            v = qkv[..., 2 * size : 3 * size].reshape(x.shape[0], n, a.n_local_heads, a.head_dim)
            if a.pos_embed_type == "rope":
                q, k = apply_rotary_emb(q, rope), apply_rotary_emb(k, rope)
            q, k, v = (value.transpose(0, 2, 1, 3) for value in (q, k, v))
            if caches[i] is not None:
                old_k, old_v = caches[i]
                k, v = mx.concatenate([old_k, k], axis=2), mx.concatenate([old_v, v], axis=2)
            col = mx.arange(position + n - k.shape[2], position + n)
            mask = (col[None, :] <= row[:, None]) & (col[None, :] >= row[:, None] - window + 1)
            mask = mx.where(mask, 0.0, -1e9).astype(x.dtype)[None, None]
            caches[i] = (k[:, :, -(window - 1) :], v[:, :, -(window - 1) :])
            if a.n_local_heads != a.n_head:
                k, v = (mx.repeat(value, a.n_head // a.n_local_heads, axis=1) for value in (k, v))
            y = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=1 / math.sqrt(a.head_dim), mask=mask
            )
            y = a.wo(y.transpose(0, 2, 1, 3).reshape(x.shape[0], n, -1))
            h = x + layer.attention_layer_scale(y)
            x = h + layer.ffn_layer_scale(layer.feed_forward(layer.ffn_norm(h)))
        self.state[id(m)] = (position + n, caches)
        x = m.output_proj(m.norm(x))
        return x.swapaxes(1, 2) if m.channels_first else x

    def decode(self, codes):
        q = self.codec.quantizer
        semantic = None
        residual = 0.0
        for i in range(codes.shape[1]):
            quantizer = (
                q.semantic_quantizer.quantizers[0] if i == 0 else q.quantizer.quantizers[i - 1]
            )
            indices = mx.clip(codes[:, i], 0, quantizer.codebook_size - 1)
            part = self.call(quantizer.out_proj, quantizer.decode_code(indices))
            if i == 0:
                semantic = 0.0 + part
            else:
                residual = residual + part
        z = semantic + residual
        z = self.call(q.post_module, z)
        for block in q.upsample:
            for layer in block:
                z = self.call(layer, z)
        audio = self.call(self.codec.decoder, z)
        self.frames += codes.shape[-1]
        mx.eval(audio, self.state)
        return audio[0, 0]

    def flush(self):
        # The whole causal decoder trims transposed-convolution right overlap.
        # No extra zero frames may be decoded at end; that would lengthen speech.
        self.state.clear()
