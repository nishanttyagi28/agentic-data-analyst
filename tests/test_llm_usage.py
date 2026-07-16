from types import SimpleNamespace

from agents import llm_client


class FakeCompletions:
    def create(self, **_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        )


def test_capture_usage_preserves_chat_contract(monkeypatch):
    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(llm_client, "get_groq_client", lambda: (fake, None))
    with llm_client.capture_llm_usage() as usage:
        response, error = llm_client.chat_completion([{"role": "user", "content": "hi"}])
        llm_client.chat_completion([{"role": "user", "content": "again"}])
    assert response == "ok"
    assert error is None
    assert usage.prompt_tokens == 22
    assert usage.completion_tokens == 14
    assert usage.total_tokens == 36
    assert usage.calls == 2


def test_calls_outside_capture_remain_supported(monkeypatch):
    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(llm_client, "get_groq_client", lambda: (fake, None))
    assert llm_client.chat_completion([]) == ("ok", None)
