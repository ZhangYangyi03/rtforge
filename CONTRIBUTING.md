# Contributing to rtforge

Thanks for looking. The single most valuable contribution to this repository is
**running the checks on your own machine and telling us what happened** —
especially if it failed. This portfolio has strong internal verification and
almost no external reproduction; an independent result, positive or negative,
is worth more than a star.

## Ground rules

- Apache-2.0 licensed. By opening a pull request you agree your contribution is under the same license.
- Bug reports, reproduction attempts and arguments about the claims are contributions. You do not need to write code.
- One change per pull request.
- If a number in the README disagrees with the code, that is a bug. Please open an issue rather than editing the prose to match.

## What this repository is

A real-time anomaly-detection research pipeline whose report must run end to end from a fresh clone.

## Reproduce it

No API key or special hardware is needed for the checks below (standard library
plus, where noted, pytest):

```bash
python scripts/report.py
git diff --exit-code -- results/report.txt
```

The second command should print nothing: `results/report.txt` is a committed artefact and is byte-identical on a fresh clone, regardless of the working directory or the operating system. The script works from any directory (it inserts the repository root into `sys.path` itself), and writes LF newlines so Windows and Linux agree.

One line in the report depends on local build state: if `cpp/build/rtforge_selftest` exists, it says `measured C++ self-test: N checks, M failures`; on a fresh clone it says `absent ... not built here`. The C++ self-test itself is a separate CI job that builds the tree with cmake. If you built the C++ tree locally, expect the report to differ — that is by design, please say so in your reproduction report rather than "fixing" the committed file.

CI runs the end-to-end report step, the deterministic-artefact check, the pytest latency-parser suite and the error budget, plus a separate cmake/g++ job for the dynamics self-test and feedforward sweep.

## Reporting an independent reproduction

Use the issue template "Independent reproduction". What makes a report actionable:

1. The commit hash or version you checked out.
2. `python -c "import sys, platform; print(sys.version, platform.platform())"`.
3. The exact commands and their full output (a failed check with a traceback is a good report).
4. Expected vs actual, in one sentence.
5. Whether all checks passed, some failed, numbers differed, or you could not run them.

Both "all checks passed on Ubuntu 24.04 / Python 3.12" and "check 3 fails on my
machine" are useful. Please do not fix a failing check silently in a PR — report
it first so the discrepancy is recorded.

## Claim scope

Where a document says a claim is exploratory, a working hypothesis, or not
human-adjudicated, please hold us to it. A pull request that strengthens a claim
without new evidence will be asked for the evidence.
