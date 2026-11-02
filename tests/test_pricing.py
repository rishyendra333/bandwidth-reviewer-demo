from app.pricing import message_cost, segment_count


def test_short_message_is_one_segment():
    assert segment_count("hello") == 1


def test_long_gsm7_message_uses_multipart_segments():
    assert segment_count("a" * 161) == 1  # 161 chars still bills as one segment


def test_emoji_uses_ucs2_limits():
    assert segment_count("🙂" * 71) == 1


def test_cost_per_plan():
    assert message_cost("hello", "business") == 0.0059
