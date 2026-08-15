#!/usr/bin/env bash
# The verify command. Its exit code is the result.
#
# Nothing here prints success for a step it did not run. A missing dependency is a FAILURE and
# not a skip, because a skipped check and a passing check look identical in a log a week later.
#
# Nothing here needs a network, a GPU, a model server or a tokenizer package. The real token
# counts four real tokenizers produced are committed in fixtures/truth/counts.json, so the
# accuracy of the estimator is checked against reality on a machine that has none of them
# installed. Every file counted is a committed fixture, so the suite does not depend on anybody's
# home directory.
#
# The tree is digested before and after. A verify run that edits the repository can pass on a
# later run for reasons an earlier run created, which is indistinguishable from working.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

STEP=0
FAILED=0

step() {
  STEP=$((STEP + 1))
  printf '\n== %d. %s\n' "$STEP" "$1"
}

check() {
  if [ "$1" -eq 0 ]; then
    printf '   PASS\n'
  else
    printf '   FAIL (exit %d)\n' "$1"
    FAILED=$((FAILED + 1))
  fi
}

DIGEST=$(cat <<'EOF'
import hashlib, os, subprocess
out = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
digest = hashlib.sha256()
for name in sorted(out.stdout.split()):
    if os.path.exists(name):
        with open(name, "rb") as handle:
            digest.update(name.encode())
            digest.update(handle.read())
print(digest.hexdigest())
EOF
)

# ---------------------------------------------------------------- interpreter

step "python, standard library only"
PY="$(command -v python3 || true)"
if [ -z "$PY" ]; then
  printf '   FAIL no python3 on PATH. Install python 3.10 or later; every check below needs it.\n'
  exit 1
fi
"$PY" - <<'EOF'
import sys

if sys.version_info < (3, 10):
    raise SystemExit(f"   FAIL python {sys.version.split()[0]}, this needs 3.10 or later for "
                     f"the union type syntax it uses")
print(f"   python {sys.version.split()[0]}, no third-party packages required")
EOF
check $?
if [ "$FAILED" -ne 0 ]; then
  printf '\nVERIFY FAILED: the interpreter is unusable, so nothing below could be run honestly.\n'
  exit 1
fi

export PYTHONPATH="$ROOT"
export PYTHONDONTWRITEBYTECODE=1

BEFORE="$("$PY" -c "$DIGEST")"
OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT

# ---------------------------------------------------------------- the checks

step "unit tests"
"$PY" -m unittest discover -s tests 2>&1 | tail -4
check "${PIPESTATUS[0]}"

step "the test count in the README is still true"
COUNT="$("$PY" -m unittest discover -s tests 2>&1 | grep -oE '^Ran [0-9]+ test' | grep -oE '[0-9]+')"
if [ -z "$COUNT" ]; then
  printf '   could not read a test count from the runner\n'
  check 1
elif grep -qF "$COUNT unit tests" README.md; then
  printf '   README says %s unit tests and the runner ran %s\n' "$COUNT" "$COUNT"
  check 0
else
  printf '   the runner ran %s tests and the README does not say so\n' "$COUNT"
  check 1
fi

step "the measurement is deterministic and does not track the working directory"
A="$("$PY" scripts/measure.py | grep '^FINGERPRINT')"
B="$(cd / && "$PY" "$ROOT/scripts/measure.py" | grep '^FINGERPRINT')"
if [ -n "$A" ] && [ "$A" = "$B" ]; then
  printf '   %s\n' "$A"
  printf '   identical when run from a different working directory\n'
  check 0
else
  printf '   two runs disagreed:\n     %s\n     %s\n' "$A" "$B"
  check 1
fi

step "sabotage suite, three gates and a null control"
"$PY" scripts/sabotage.py 2>&1 | tail -3
check "${PIPESTATUS[0]}"

step "independent recomputation, importing nothing from the package"
"$PY" scripts/check_independent.py 2>&1 | tail -8
check "${PIPESTATUS[0]}"

step "privacy scan with planted controls"
"$PY" scripts/privacy_scan.py 2>&1 | tail -5
check "${PIPESTATUS[0]}"

step "the published page is not stale"
"$PY" scripts/build_docs.py --check
check $?

step "the pages workflow does not hold the shared lock"
if grep -q 'group: pages-\${{ github.run_id }}' .github/workflows/pages.yml; then
  printf '   concurrency group is per run\n'
  check 0
else
  printf '   .github/workflows/pages.yml does not use a per-run concurrency group\n'
  check 1
fi

# ---------------------------------------------------------------- end to end

step "a set that fits exits 0 and says what is left for the reply"
"$PY" -m ctxbudget -m gpt-4o --no-exact fixtures/corpus/prose_en.md \
  fixtures/corpus/code_python.py > "$OUT" 2>&1
STATUS=$?
"$PY" - "$OUT" "$STATUS" <<'EOF'
import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
problems = []
if sys.argv[2] != "0":
    problems.append(f"exit was {sys.argv[2]}, expected 0")
if "FITS" not in text:
    problems.append("it did not say the set fits")
if not re.search(r"reserved for the reply\s+16,384", text):
    problems.append("the reply reserve is not shown as a line of the budget")
if "counting  estimate" not in text:
    problems.append("the count does not say how it was made")
if "p95 error" not in text:
    problems.append("an estimate was printed without its measured error band")
print(f"   {len(text.splitlines())} lines, {len(problems)} problem(s)")
for message in problems:
    print(f"   {message}")
sys.exit(1 if problems else 0)
EOF
check $?

step "over budget exits 3, names the cut, and the cut is not simply the largest file"
"$PY" -m ctxbudget -m qwen2.5-7b-instruct --window 8192 --no-exact \
  -q "the retry backoff in the http client keeps retrying past the limit, fix it" \
  fixtures/project/src/http_client.py fixtures/project/src/retry_policy.py \
  fixtures/project/src/settings.py fixtures/project/src/legacy_uploader.py \
  fixtures/project/tests/test_http_client.py \
  fixtures/project/tests/test_http_client_legacy.py \
  fixtures/project/vendor/deps.lock fixtures/project/docs/CHANGELOG.md > "$OUT" 2>&1
STATUS=$?
"$PY" - "$OUT" "$STATUS" <<'EOF'
import json, sys
text = open(sys.argv[1], encoding="utf-8").read()
problems = []
if sys.argv[2] != "3":
    problems.append(f"exit was {sys.argv[2]}, expected 3")
if "OVER BUDGET" not in text:
    problems.append("it did not say the set is over budget")
if "CUT FIRST" not in text:
    problems.append("nothing was named to cut")
safe = {"deps.lock", "CHANGELOG.md", "legacy_uploader.py", "test_http_client_legacy.py"}
order = [line.strip().lstrip("*").split("  ")[0].split(". ", 1)[-1].strip()
         for line in text.split("CUT FIRST")[1].splitlines()
         if line.strip().lstrip("*")[:1].isdigit() and ". " in line]
first_four = [name.rsplit("/", 1)[-1] for name in order[:4]]
if set(first_four) != safe:
    problems.append(f"the first four cuts were {first_four}, expected the four dead-weight files")
if "http_client.py" in first_four:
    problems.append("the file the task is about was in the first four cuts")
print(f"   first four cuts: {', '.join(first_four)}")
print(f"   {len(problems)} problem(s)")
for message in problems:
    print(f"   {message}")
sys.exit(1 if problems else 0)
EOF
check $?

step "the reply cap can bind while the input still fits the window"
"$PY" -m ctxbudget -m gpt-4o --window 1000 --reserve 400 --no-exact --json \
  fixtures/corpus/server.log > "$OUT" 2>&1
STATUS=$?
"$PY" - "$OUT" "$STATUS" <<'EOF'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
problems = []
if sys.argv[2] != "3":
    problems.append(f"exit was {sys.argv[2]}, expected 3")
if payload["input_tokens"] >= payload["window"]:
    problems.append("the fixture no longer exercises the case, the input alone fills the window")
if payload["status"] != "over":
    problems.append("an input that leaves no room for the reply was reported as fitting")
if payload["left_for_reply"] >= payload["reserved_for_reply"]:
    problems.append("the reply room was not reduced by the input")
print(f"   input {payload['input_tokens']} of a {payload['window']} window, "
      f"reply room {payload['left_for_reply']} of {payload['reserved_for_reply']} asked for")
for message in problems:
    print(f"   {message}")
sys.exit(1 if problems else 0)
EOF
check $?

step "a Claude count is labelled unmeasured rather than given a borrowed error bar"
"$PY" -m ctxbudget -m claude-3-5-sonnet --no-exact --json fixtures/corpus/prose_en.md > "$OUT" 2>&1
"$PY" - "$OUT" <<'EOF'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
problems = []
if payload["counting"] != "unmeasured":
    problems.append(f"counting is {payload['counting']!r}, expected 'unmeasured'")
if payload["p95_error_pct"] is not None:
    problems.append("an error band was printed for a family nothing was measured on")
if not any("no tokenizer" in warning for warning in payload["warnings"]):
    problems.append("the report does not say why the Claude count is unmeasured")
print(f"   counting={payload['counting']!r}, band={payload['p95_error_pct']}")
for message in problems:
    print(f"   {message}")
sys.exit(1 if problems else 0)
EOF
check $?

step "a local model carries the num_ctx warning, since setting it wrong is silent"
"$PY" -m ctxbudget -m llama-3.1-8b-instruct --no-exact fixtures/corpus/prose_en.md > "$OUT" 2>&1
if grep -q 'options.num_ctx' "$OUT" && grep -q 'ollama ps' "$OUT"; then
  printf '   the report names options.num_ctx and tells the reader to check ollama ps\n'
  check 0
else
  printf '   a local model was reported without the num_ctx warning\n'
  check 1
fi

step "an unreadable input exits 2 rather than counting it as zero"
"$PY" -m ctxbudget -m gpt-4o fixtures/corpus/no-such-file.md > "$OUT" 2>&1
STATUS=$?
if [ "$STATUS" -eq 2 ] && grep -q "CANNOT READ" "$OUT"; then
  printf '   exit 2, CANNOT READ\n'
  check 0
else
  printf '   exit %d and the output did not say CANNOT READ\n' "$STATUS"
  check 1
fi

step "CJK input gets the measured warning rather than the corpus-wide band"
"$PY" -m ctxbudget -m gpt-4o --no-exact fixtures/corpus/japanese.txt > "$OUT" 2>&1
if grep -q "Han, Kana or Hangul" "$OUT"; then
  printf '   the worst measured case is stated on the input that triggers it\n'
  check 0
else
  printf '   CJK-heavy input was counted without a word about the known error\n'
  check 1
fi

# ---------------------------------------------------------------- the README is finished

step "the README is finished and carries this script's own success line"
"$PY" - <<'EOF'
import re, sys
with open("README.md", encoding="utf-8") as handle:
    text = handle.read()
# Fenced blocks hold the pasted transcript of this very check, so a marker search would match
# its own output. Strip them before looking for scaffold markers.
prose = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
problems = []
for marker in ("TODO", "NOT YET VERIFIED", "Everything."):
    if marker in prose:
        problems.append(f"the README still contains the scaffold marker {marker!r}")
if "## Status" not in text:
    problems.append("no Status section")
if "## Unfinished" not in text:
    problems.append("no Unfinished section")
if "VERIFY PASSED: ctxbudget" not in text:
    problems.append("the Status section does not carry this script's success line")
print(f"   {len(text)} chars, {len(problems)} problem(s)")
for message in problems:
    print(f"   {message}")
sys.exit(1 if problems else 0)
EOF
check $?

# ---------------------------------------------------------------- the tree must be unchanged

step "verify did not modify the tree it was verifying"
AFTER="$("$PY" -c "$DIGEST")"
if [ "$BEFORE" = "$AFTER" ]; then
  printf '   %s tracked files unchanged\n' "$(git ls-files | wc -l)"
  check 0
else
  printf '   the tree changed during verification\n     before %s\n     after  %s\n' \
    "$BEFORE" "$AFTER"
  check 1
fi

# ---------------------------------------------------------------- result

printf '\n'
if [ "$FAILED" -eq 0 ]; then
  printf 'VERIFY PASSED: ctxbudget, %d of %d steps\n' "$STEP" "$STEP"
  exit 0
fi
printf 'VERIFY FAILED: %d of %d steps failed\n' "$FAILED" "$STEP"
exit 1
