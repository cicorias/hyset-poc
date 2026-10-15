from collections.abc import Iterator

import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage

from hyset.data import Tool
from hyset.reward import RewardRequest, make_reward
from hyset.reward.llm import LangChainReward, parse_score
from hyset.reward.sim import SimulatedExecutionReward

FLIGHT = Tool("flight", "Flight Search", "search flights", "travel", "flight_search")
FLIGHT2 = Tool("flight2", "Other Flights", "also flights", "travel", "flight_search")
HOTEL = Tool("hotel", "Hotels", "book hotels", "travel", "hotel_booking")
JUNK = Tool("junk", "Jokes", "random jokes", "misc", "filler:jokes")


def req(tools, gt=(FLIGHT, HOTEL)):
    return RewardRequest("q1", "find flights and a hotel", list(tools), list(gt))


def test_sim_reward_shapes():
    sim = SimulatedExecutionReward(noise=0.0)
    full, half, alt, extra = sim.score(
        [req([FLIGHT, HOTEL]), req([FLIGHT]), req([FLIGHT2, HOTEL]), req([FLIGHT, HOTEL, JUNK])]
    )
    assert full == 1.0
    assert half == 0.25  # (1/2) ** sharpness 2
    assert alt == 1.0  # a functionally equivalent tool earns full credit
    assert extra == pytest.approx(0.95)


def test_sim_noise_is_deterministic():
    sim = SimulatedExecutionReward(noise=0.5, seed=1)
    r = [req([FLIGHT, HOTEL])] * 3
    assert len(set(sim.score(r))) == 1


def test_make_reward():
    assert make_reward("none") is None
    assert make_reward("sim").name == "sim"
    with pytest.raises(ValueError):
        make_reward("bogus")


@pytest.mark.parametrize(
    "text,want",
    [
        ('{"score": 0.7, "reason": "ok"}', 0.7),
        ('```json\n{"score": 1, "reason": "x"}\n```', 1.0),
        ('Thinking... {"reason": "partial", "score": 0.5}', 0.5),
        ("score: 2", 1.0),
        ("no idea", None),
    ],
)
def test_parse_score(text, want):
    assert parse_score(text) == want


def test_judge_mode_with_fake_model():
    fake = FakeListChatModel(responses=['{"score": 0.8, "reason": "covers both"}'])
    reward = LangChainReward(agent_model=fake, mode="judge", max_concurrency=1)
    assert reward.score([req([FLIGHT, HOTEL])]) == [0.8]


class ToolCallingFake(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        self.__dict__["bound"] = [t.name for t in tools]
        return self


def _script() -> Iterator[AIMessage]:
    yield AIMessage(
        content="",
        tool_calls=[{"name": "flight", "args": {"request": "flights to Tokyo"}, "id": "c1"}],
    )
    yield AIMessage(content="Found flights; no hotel tool needed.")
    yield AIMessage(content='{"score": 0.5, "reason": "hotel part skipped"}')


def test_agent_mode_runs_mock_tools_then_judges():
    fake = ToolCallingFake(messages=_script())
    reward = LangChainReward(agent_model=fake, mode="agent", max_concurrency=1)
    transcript, answer = reward.run_agent(req([FLIGHT, HOTEL]))
    assert "CALL flight" in transcript and "Simulated flight_search" in transcript
    assert answer.startswith("Found flights")
    assert sorted(fake.__dict__["bound"]) == ["flight", "hotel"]
    assert reward._judge("sys", "user") == 0.5


def test_provider_errors_become_none():
    class Boom(FakeListChatModel):
        def invoke(self, *a, **k):
            raise RuntimeError("network down")

    reward = LangChainReward(agent_model=Boom(responses=["x"]), mode="judge")
    assert reward.score([req([FLIGHT])]) == [None]
