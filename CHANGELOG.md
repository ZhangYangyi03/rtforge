# Changelog

All notable changes are documented here. This file starts with the
2026-11 GitHub portfolio audit: the fixes below were made in response to
`GitHub-项目综合评价.md` and are intentionally itemised so a reviewer can
verify each one against the commit.

## [Unreleased] — 2026-11 portfolio audit fixes

- Fixed scripts/report.py inserting the repository root into sys.path so 'python scripts/report.py' works from any working directory (CI was red on all 4 runs).
- Made the report write to <repo>/results/report.txt instead of a cwd-relative path.
- Set PYTHONPATH in the CI workflow as a second layer of protection.
- Made `results/report.txt` reproducible on a fresh clone. The committed report had a line `measured C++ self-test: 16 checks, 0 failures` produced on a machine with `cpp/build/rtforge_selftest` present; a fresh clone silently omitted it. `render()` now always emits an explicit `absent ... not built here` line instead of dropping the row, `scripts/report.py` pins LF newlines so Windows and Linux produce identical bytes, and CI asserts `git diff --exit-code -- results/report.txt` after regenerating. Verified byte-identical when run from the repo root and from `/tmp`.
- Added governance files: `CONTRIBUTING.md`, this changelog, `CITATION.cff` and the "Independent reproduction" issue template.

## Audit context

- Full portfolio assessment: `GitHub-项目综合评价.md` (overall 6.8/10; external validation 1.5/10).
- Canonical maintainer identity: `ZhangYangyi03` / Yangyi Zhang.
- This changelog does not claim external validation. See the "Independent reproduction" issue template.
