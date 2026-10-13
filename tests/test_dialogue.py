import json
from types import SimpleNamespace

import httpx
import pytest
from companion.adapters import LocalDialogue, SetupRequired
from companion.config import Config


@pytest.mark.asyncio
async def test_long_context_trim_preserves_a_stable_prefix(tmp_path):
    prefixes = []

    async def handle(request):
        payload = json.loads(request.content)
        if request.url.path == "/tokenize":
            messages = payload["messages"]
            prefixes.append(messages[1]["content"])
            return httpx.Response(200, json={"count": len(messages) * 100})
        return httpx.Response(
            200, text='data: {"choices":[{"delta":{"content":"Hello."}}]}\n\ndata: [DONE]\n\n'
        )

    d = LocalDialogue(Config(data=tmp_path, token="t" * 64))
    await d.client.aclose()
    d.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    d.worker = SimpleNamespace(url="http://test")

    async def prepare(profile):
        pass

    d.prepare = prepare
    messages = [{"role": "system", "content": "System"}]
    for i in range(30):
        messages.extend(
            [
                {"role": "user", "content": f"user-{i}"},
                {"role": "assistant", "content": f"assistant-{i}"},
            ]
        )
    messages.append({"role": "user", "content": "Current"})
    assert "".join([x async for x in d.generate(messages, "quiet")]) == "Hello."
    skipped = d.history_skip
    assert skipped > 0
    prefix = prefixes[-1]
    messages.extend(
        [{"role": "assistant", "content": "Hello."}, {"role": "user", "content": "Next"}]
    )
    _ = [x async for x in d.generate(messages, "quiet")]
    assert d.history_skip == skipped
    assert prefixes[-1] == prefix
    d.invalidate_cache()
    _ = [x async for x in d.generate(messages, "quiet")]
    assert prefixes[-1] != prefix
    assert d.history_skip == len(messages) - 1 - 9
    await d.client.aclose()


@pytest.mark.asyncio
async def test_oversized_current_message_is_rejected(tmp_path):
    d = LocalDialogue(Config(data=tmp_path, token="t" * 64))
    await d.client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "responses,expected,calls",
    [
        (["", "Recovered."], "Recovered.", 2),
        (["", ""], "", 2),
        (["A partial sentence"], "A partial sentence", 1),
    ],
)
async def test_only_empty_dialogue_retries_with_fresh_prefix(tmp_path, responses, expected, calls):
    requests = []

    async def handle(request):
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 50})
        requests.append(json.loads(request.content))
        content = responses[len(requests) - 1]
        delta = json.dumps({"choices": [{"delta": {"content": content}}]})
        return httpx.Response(200, text=f"data: {delta}\n\ndata: [DONE]\n\n")

    d = LocalDialogue(Config(data=tmp_path, token="t" * 64))
    await d.client.aclose()
    d.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    d.worker = SimpleNamespace(url="http://test")

    async def prepare(profile):
        pass

    d.prepare = prepare
    messages = [{"role": "system", "content": "System"}, {"role": "user", "content": "Hello"}]
    assert "".join([x async for x in d.generate(messages, "quiet")]) == expected
    assert len(requests) == calls
    if calls == 2:
        assert requests[0]["messages"][0] != requests[1]["messages"][0]
        assert requests[0]["messages"][-1] == requests[1]["messages"][-1]
    await d.client.aclose()
    d.client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"count": 5000}))
    )
    d.worker = SimpleNamespace(url="http://test")

    async def prepare(profile):
        pass

    d.prepare = prepare
    with pytest.raises(SetupRequired, match="context"):
        _ = [
            x
            async for x in d.generate(
                [{"role": "system", "content": "X"}, {"role": "user", "content": "Y"}], "quiet"
            )
        ]
    await d.client.aclose()
