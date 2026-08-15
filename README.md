# ctxbudget

What fills a model's context window, what is left for the reply, and what to cut first.

Catalog task: `CLI-034`. One of a public catalog of build ideas:
https://github.com/JesseRWeigel/722-things-to-build

## What this is

A command line tool that counts the tokens in a set of files, subtracts what the reply needs,
and says whether the request fits. When it does not fit, it ranks what to drop.

Three things it insists on:

- **The window is not the input budget.** The reply comes out of the same window, and several
  hosted models cap the reply separately far below the window. A request that fills the window is
  accepted and then truncated mid-sentence, which does not look like an error anywhere.
- **A count carries how it was made.** Either a real tokenizer produced it, or the fitted table
  did and the number comes with the error measured on files the fit never saw. Nothing returns a
  bare integer.
- **Size is not a reason to cut.** The largest file in a context is often the one everything else
  refers to. Ranking is tokens returned per unit of demand lost, where demand is inbound
  references plus overlap with the task, discounted by how much of the file is already duplicated
  in the window.

Python 3.10 or later. No third-party packages, at runtime or to test it.

## Running it

```
$ python3 -m ctxbudget -m gpt-4o fixtures/corpus/prose_en.md fixtures/corpus/code_python.py
model     gpt-4o, 128,000 token window, family o200k_base
reserve   16,384 tokens held back for the reply (the model's own maximum output, from the model table)
counting  estimate, fitted table for o200k_base, p95 error 8.15%

PART                                                 TOKENS    SHARE
fixtures/corpus/code_python.py                          501     0.4%
fixtures/corpus/prose_en.md                             387     0.3%
chat template overhead, 1 message(s), vendor-doc          6     0.0%
------------------------------------------------  ---------  -------
input total                                             894     0.7%
reserved for the reply                               16,384    12.8%
unused                                              110,722    86.5%

FITS. 110,722 tokens spare, and the reply can use its full 16,384.
```

Over budget, with a task so the ranking has something to rank against:

```
$ python3 -m ctxbudget -m qwen2.5-7b-instruct --window 8192 \
    -q "the retry backoff in the http client keeps retrying past the limit, fix it" \
    fixtures/project/src/*.py fixtures/project/tests/*.py \
    fixtures/project/vendor/deps.lock fixtures/project/docs/CHANGELOG.md

OVER BUDGET by 22,523 tokens. The input is 28,667 and only 6,144 is available once 2,048 is kept back for the reply.
  The input alone exceeds the window, so the request is refused outright.

CUT FIRST  (tokens / ((1 + inbound refs + task match) x (1 - redundancy)))
 *1. fixtures/project/vendor/deps.lock  22,497 tokens, score 11,248
      nothing else in the context refers to it; the task mentions limit; 37% of its lines repeat inside it
 *2. fixtures/project/docs/CHANGELOG.md  2,523 tokens, score 2,523
      nothing else in the context refers to it; no term from the task appears in it; 54% of its lines repeat inside it
  3. fixtures/project/src/legacy_uploader.py  622 tokens, score 626
      nothing else in the context refers to it; no term from the task appears in it
  4. fixtures/project/tests/test_http_client_legacy.py  577 tokens, score 316
      1 other part(s) refer to it via FakeSleep, HttpClientTest; the task mentions retry, backoff, http; 85% line overlap with fixtures/project/tests/test_http_client.py
  5. fixtures/project/tests/test_http_client.py  540 tokens, score 296
      1 other part(s) refer to it via FakeSleep, HttpClientTest; the task mentions retry, backoff, http; 85% line overlap with fixtures/project/tests/test_http_client_legacy.py
  6. fixtures/project/src/http_client.py  1,337 tokens, score 122
      3 other part(s) refer to it via GaveUp, HttpClient, describe; the task mentions retry, backoff, http
  Cutting the 2 marked * frees 25,020 tokens and brings the input to 3,647, under the 6,144 available.
```

`src/http_client.py` is the second largest file in that set and the one the task is about.
Ranking by size cuts it fourth; this ranks it last.

An argument can be a directory, which is walked, or `-`, which reads standard input, so a diff
goes in with `git diff | python3 -m ctxbudget -m gpt-4o -`. Files under a directory that are not
UTF-8 text are skipped rather than counted, and every skipped path is printed with its reason. A
file named directly on the command line is not skipped: that is a deliberate request to count it,
so it fails with exit 2 instead.

Useful flags: `--system PATH` counts a system prompt as its own part, `--json` prints the whole
report as JSON, `--explain` shows what dividing characters by four would have said, `--reserve N`
sets how much to hold back for the reply, `--window N` and `--family NAME` describe a model the
table has never heard of, and `--no-exact` ignores an installed tokenizer so you can see what the
shipped table alone produces.

Exit codes: `0` fits, `3` over budget, `2` an input could not be read. The third one matters: a
file that cannot be read is never counted as zero tokens.

## How the counting works, and how wrong it is

`tiktoken` gives exact counts for the two OpenAI encodings when it is installed. It usually is
not, and it never covers Qwen or Llama, so the fallback is a fitted table.

The table prices pretoken chunks. A BPE tokenizer splits text with a fixed regex before it merges
anything, and merges never cross a chunk boundary, so the chunk sequence is a hard skeleton of
the token sequence. Each chunk is priced by its class and byte length. Classes that earn their
place, each because leaving them out was measured to be wrong:

- a word is priced by what it starts with, since a BPE vocabulary has an entry for a space-led
  word but no merge from a comma into one;
- Han and Kana apart from Hangul, because Korean is written with spaces and Han is not;
- chunks inside a hash or base64 blob apart from ordinary words, and by case, because lowercase
  hex has merges and mixed-case base64 does not;
- chunks past their class cap get an intercept as well as a per-byte rate.

Fitted against four real tokenizers on 1,260 text files taken read-only from repositories on the
development machine, 951 of them fitting the table and 309 held out. Error on the held-out files,
whole-file counts:

| family | median | p95 | worst | chars/4 median | chars/4 p95 |
|---|---|---|---|---|---|
| `cl100k_base` | 2.678% | 7.519% | 14.436% | 9.385% | 31.073% |
| `o200k_base` | 2.382% | 8.15% | 14.436% | 8.889% | 30.086% |
| `qwen2.5` | 2.564% | 7.265% | 12.826% | 10.298% | 36.588% |
| `llama3` | 2.646% | 7.297% | 14.436% | 9.385% | 30.769% |

The last two columns are dividing characters by four on the same files.

CJK is measured separately, because the corpus above is mostly ASCII code and its band says
nothing about a file of Japanese. That measurement cannot come from the corpus either: of the
1,260 files, zero are as much as 10% Han, Kana or Hangul. The machine holds Japanese UI strings
inside TypeScript files and no CJK documents. So it is measured on three committed fixtures that
the fit never sees, and the tool quotes that number, with its file count, at anyone whose input
crosses the threshold: median 1.3% and worst 1.8% on `cl100k_base`, worst 10.8% on `qwen2.5`.
Three short documents is a thin measurement and the tool says so rather than implying a corpus
stands behind it.

`fixtures/evidence/calibration.md` has the full breakdown, including per-extension error and the
count of training chunks behind every class. Two classes are fitted on very little: `word_hangul`
on 7 chunks and `word_hangul_sp` on 59, and `word_hangul_px` on none at all, so it is copied from
its base class and the copy is recorded rather than passed off as a fit.

## What is not measured

- **Claude counts.** There is no public Claude tokenizer. Claude models count through the
  `cl100k_base` table as a stand-in, and the report labels them `unmeasured` and prints no error
  band rather than borrowing one.
- **Context windows and reply caps.** `ctxbudget/data/models.json` is vendor documentation read
  on 2026-08-15. Nothing in it was measured here and it ages; `--window` and `--reserve` override
  any of it.
- **Chat template overhead** for the OpenAI families is the documented per-message allowance,
  applied server side and not reproducible locally. For Qwen and Llama the wrapper is a literal
  string and was encoded with the real tokenizer, so those two are measured.

## Verifying it

`bash scripts/verify.sh` needs nothing but Python 3.10 or later: no network, no GPU, no model
server, no tokenizer package. The real counts four tokenizers produced for the committed fixtures
are in `fixtures/truth/counts.json`, so the estimator is checked against reality on a machine that
has none of them installed.

Refitting the table is a separate step and is not part of verify. It needs `tiktoken`,
`tokenizers`, a Qwen and a Llama `tokenizer.json`, and a corpus of real files:

```
python3 scripts/calibrate.py --corpus ~/Projects --exclude ctxbudget --tokenizers-dir DIR
```

## Status

`bash scripts/verify.sh`, exit code 0. Pasted whole, from the run that produced it:

```
== 1. python, standard library only
   python 3.12.3, no third-party packages required
   PASS

== 2. unit tests
----------------------------------------------------------------------
Ran 64 tests in 0.299s

OK
   PASS

== 3. the test count in the README is still true
   README says 64 unit tests and the runner ran 64
   PASS

== 4. the measurement is deterministic and does not track the working directory
   FINGERPRINT 034f7e6733acb1aacc0be6cd3c09a9b3c1dc11955faf33f8b32b5bf3e9e3116f
   identical when run from a different working directory
   PASS

== 5. sabotage suite, three gates and a null control
  error-band-dropped                 guard  applies=yes output unchanged=yes caught=yes

16 of 16 sabotages proven under the three-gate rule
   PASS

== 6. independent recomputation, importing nothing from the package
   difflib puts the two test files at 0.95 similar, the tool claims 0.85 redundancy
   first four cuts that are dead weight: tool 4/4, ranking by size 3/4

== negative controls, these must DISAGREE
   characters over eight lands 69.0% out, outside the 2.0% bound, as it must
   on japanese.txt the scanner is 1.8% out with the tail term and 75.2% out without it

INDEPENDENT CHECK PASSED
   PASS

== 7. privacy scan with planted controls
== the committed tree
   56 tracked files read, 0 containing a NUL byte
   8 patterns, every one of them proven to fire above

PRIVACY SCAN PASSED
   PASS

== 8. the published page is not stale
   docs/index.html matches a rebuild from the data, 13923 chars
   PASS

== 9. the pages workflow does not hold the shared lock
   concurrency group is per run
   PASS

== 10. a set that fits exits 0 and says what is left for the reply
   20 lines, 0 problem(s)
   PASS

== 11. over budget exits 3, names the cut, and the cut is not simply the largest file
   first four cuts: deps.lock, CHANGELOG.md, legacy_uploader.py, test_http_client_legacy.py
   0 problem(s)
   PASS

== 12. the reply cap can bind while the input still fits the window
   input 776 of a 1000 window, reply room 224 of 400 asked for
   PASS

== 13. a Claude count is labelled unmeasured rather than given a borrowed error bar
   counting='unmeasured', band=None
   PASS

== 14. a local model carries the num_ctx warning, since setting it wrong is silent
   the report names options.num_ctx and tells the reader to check ollama ps
   PASS

== 15. an unreadable input exits 2 rather than counting it as zero
   exit 2, CANNOT READ
   PASS

== 16. CJK input gets the measured warning rather than the corpus-wide band
   the worst measured case is stated on the input that triggers it
   PASS

== 17. the README is finished and carries this script's own success line
   13789 chars, 0 problem(s)
   PASS

== 18. verify did not modify the tree it was verifying
   56 tracked files unchanged
   PASS

VERIFY PASSED: ctxbudget, 18 of 18 steps
```

64 unit tests, 16 sabotages under the three-gate rule with a null control, and an independent recount that imports nothing from the package and reproduces every per-file count to within 0.0%.

## Unfinished

- **No installed-model table and no VRAM estimate.** The catalog task asks for a table of which
  locally installed models could take a given context at which setting, with the VRAM that
  context would cost. That needs a running Ollama server to enumerate models and read their layer
  and head counts, which verify deliberately cannot depend on, so it is not built. The KV cache
  arithmetic is the easy half; the honest half is measuring a prediction against real VRAM use,
  and neither is done.
- **Exact counts only for the two OpenAI encodings.** `tokenizers` can load the Qwen and Llama
  vocabularies and the tool does not use it even when it is installed, so those two families are
  always estimated.
- **The CJK number rests on three fixtures.** It is out of sample and it is not a corpus. A
  proper measurement wants CJK documents this machine does not have.
- **The split regex is a port of the `cl100k_base` pattern.** The other three families split
  differently, and the fitted table absorbs the difference rather than reproducing their regexes.
  The error of doing that is measured per family, which is the whole point of the table above, but
  a per-family splitter would be better than measuring the cost of not having one.

## License

MIT.
