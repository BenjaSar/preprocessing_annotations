"""G-EVAL-3: the room answer budget must be able to carry max_rooms rooms.

Measured 2026-11-03. The prompt asks for up to VLMConfig.max_rooms (25) rooms
and the budget was 1024 tokens, sized on a comment's estimate of 30
tokens/room. A real answer from the shipped dataset costs 51 (median over 245
answers, Qwen3-VL tokenizer), so the budget delivered about 20. The ask and the
budget disagreed, and nothing in the code connected them.

Why the gap stayed invisible: truncation here is graceful.
UnslothQwenBackend._parse_room_response salvages every complete object from an
unterminated array, so an over-budget tile quietly returns fewer rooms instead
of failing. On the GT pages 15 of 401 tiles hold more than max_rooms rooms --
6 of Rockaway's 12, one with 66 -- and that document scores .249 recall while
sitting at 92% of its own prediction ceiling.

These tests pin the relationship, not the numbers: change max_rooms or the
answer schema and the budget has to follow, whatever the values become.
"""


import pytest

from preprocessing_annotations.config.config import VLMConfig


def test_the_budget_can_carry_the_rooms_the_prompt_asks_for():
    """The invariant the defect broke."""
    config = VLMConfig()

    assert config.room_token_budget() >= config.max_rooms * config.room_tokens_per_room


def test_the_budget_follows_max_rooms():
    """Raising the ask must raise the budget, or the extra rooms are asked
    for and then truncated away."""
    config = VLMConfig()
    config.max_rooms = config.max_rooms * 2

    assert config.room_token_budget() >= config.max_rooms * config.room_tokens_per_room


def test_the_configured_value_is_a_floor_not_a_ceiling():
    """Lowering max_rooms must not shrink the budget below the value that was
    tested in production."""
    config = VLMConfig()
    config.max_rooms = 1

    assert config.room_token_budget() == config.room_max_new_tokens


def test_a_thinking_checkpoint_keeps_its_own_budget():
    """Measured 2026-07-24: a Thinking checkpoint spends the whole budget on a
    reasoning preamble before any JSON, so its budget is sized for that, not
    for the answer. Deriving it from max_rooms would undo that."""
    config = VLMConfig()

    assert config.room_token_budget(is_thinking=True) == config.room_max_new_tokens_thinking


def test_the_tokens_per_room_cost_is_not_optimistic():
    """A regression guard on the measurement itself. 51 is the median and 59
    the worst case over 245 real answers; a value below the median would
    reintroduce the same under-budgeting by arithmetic."""
    assert VLMConfig().room_tokens_per_room >= 51


@pytest.mark.parametrize("max_rooms", [1, 10, 25, 40, 100])
def test_the_invariant_holds_across_configurations(max_rooms):
    config = VLMConfig()
    config.max_rooms = max_rooms

    budget = config.room_token_budget()

    assert budget >= min(
        config.max_rooms * config.room_tokens_per_room, config.room_max_new_tokens
    )
    assert budget >= config.room_max_new_tokens, "the tested floor must survive"
