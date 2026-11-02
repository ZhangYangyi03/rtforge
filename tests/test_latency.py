"""The latency parser, held to real cyclictest output rather than to my idea of it.

The fixture is one "Thread 0" line captured from this machine (WSL2, 6.18 kernel, 30 s run,
200 us interval). It is 401 distinct latencies from 160 us to 105,046 us. The parser's first
version reported p50 = 32 ms on this file, which is impossible for a 200 us timer -- it had
rejected tokens longer than six digits as noise, deleting everything above 300 ms and
turning a clean run into a catastrophe. These tests exist because that happened.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rtforge.latency.histogram import parse_quick, stats, summarize  # noqa: E402

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "cyclictest_q_thread0.txt"


def test_the_fixture_is_what_we_think_it_is():
    threads = parse_quick(FIXTURE.read_text())
    assert len(threads) == 1
    values = threads[0]["values"]
    assert len(values) == 400          # 401 tokens minus the "0:" label
    assert values[0] == 631
    assert values[-1] == 105046


def test_a_long_latency_is_not_dropped():
    """The regression: 105,046 us is seven characters and has to survive."""
    threads = parse_quick(FIXTURE.read_text())
    assert max(threads[0]["values"]) == 105046
    s = stats(threads[0]["values"], threads[0]["others"])
    assert s["max_seen"] == 105046


def test_the_parser_reports_what_the_file_says_even_when_the_answer_is_bad():
    """On this WSL2 host the measured distribution really is awful -- p50 around 50 ms on a
    200 us timer, because a VM's vCPU gets descheduled. The parser's job is to say so, not to
    sanitise it. What must not happen is the old bug, where values above 300 ms were deleted
    and the shape of the tail was fiction."""
    s = stats(parse_quick(FIXTURE.read_text())[0]["values"])
    assert s["p50"] > 0
    assert s["max_seen"] == 105046
    # The tail crosses 100 ms, which the width-filtered version could never have shown.
    assert s["p99"] > 90000


def test_percentiles_are_monotone():
    s = stats(parse_quick(FIXTURE.read_text())[0]["values"])
    assert s["min"] <= s["p50"] <= s["p90"] <= s["p99"] <= s["p999"] <= s["max_seen"]


def test_the_others_suffix_is_kept_not_swallowed():
    line = "# Thread 3: 00020 00044 00061 # 00123 others\n"
    threads = parse_quick(line)
    assert threads[0]["others"] == 123
    assert threads[0]["values"] == [20, 44, 61]


def test_a_line_without_a_thread_header_is_ignored():
    text = "# Histogram\n000000 000000 000000\n000001 000000 000000\n"
    assert parse_quick(text) == []


def test_summarize_marks_the_worst_thread():
    text = ("# Thread 0: 00010 00020 00030 # 00000 others\n"
            "# Thread 1: 00010 00500 00900 # 00002 others\n")
    path = pathlib.Path(__file__).parent / "_tmp_two.hist"
    path.write_text(text, encoding="utf-8")
    try:
        s = summarize([str(path)])
    finally:
        path.unlink()
    assert s["runs"][0]["threads"] == 2
    assert s["runs"][0]["max_seen"] == 900     # thread 1 is the worst
    assert s["runs"][0]["others"] == 2
