# Calibration, measured not asserted

Fitted and measured on 1260 real text files taken read-only from repositories on the machine that built this project. 951 files fitted the table, 309 were held out and never seen by the fit. 60 of the files were pulled in deliberately because they contain CJK text, since a hash-ordered sample of this machine turned up almost none and the CJK buckets were going unfitted. File paths are deliberately not recorded here; only counts and extensions are.

Held-out error, whole-file token count, estimator against the real tokenizer:

| family | tokenizer | files | held-out tokens | median | p95 | worst | chars/4 median | chars/4 p95 |
|---|---|---|---|---|---|---|---|---|
| `cl100k_base` | tiktoken cl100k_base (tiktoken 0.13.0) | 308 | 782,631 | 2.678% | 7.519% | 14.444% | 9.385% | 30.897% |
| `o200k_base` | tiktoken o200k_base (tiktoken 0.13.0) | 308 | 782,687 | 2.382% | 8.15% | 14.444% | 8.889% | 30.0% |
| `qwen2.5` | Qwen2.5 tokenizer.json (tokenizers 0.23.1) | 309 | 825,293 | 2.564% | 7.265% | 12.829% | 10.298% | 36.588% |
| `llama3` | Llama 3 tokenizer.json (tokenizers 0.23.1) | 308 | 780,077 | 2.646% | 7.297% | 14.444% | 9.385% | 30.769% |

The last two columns are the thing this replaces. Dividing characters by four is the usual shortcut, and its error on the same held-out files is there beside ours.

## CJK, measured on its own

The band above comes from a corpus that is mostly ASCII code, and it does not describe a file of Japanese. So the tool quotes a separate number once an input is at least 10% Han, Kana or Hangul, and that number is measured on text of that kind. It cannot come from the sampled corpus: of the 1260 files in it, 0 reach that share. The machine has Japanese UI strings inside TypeScript files and no Japanese documents, so an error measured on those files would be the error on TypeScript. It is measured instead on the committed CJK fixtures, which live inside the excluded project directory and are never part of the fit:

| family | fixtures | median | worst |
|---|---|---|---|
| `cl100k_base` | chinese.txt, japanese.txt, korean.txt | 1.308% | 1.773% |
| `o200k_base` | chinese.txt, japanese.txt, korean.txt | 6.725% | 7.942% |
| `qwen2.5` | chinese.txt, japanese.txt, korean.txt | 3.906% | 10.823% |
| `llama3` | chinese.txt, japanese.txt, korean.txt | 5.108% | 9.218% |

Three short documents is a thin measurement and the tool says so rather than implying a corpus stands behind it. It is still the honest number: the alternative was a constant in the source, which is how this file previously came to claim 37.7% while the shipped table was off by five.

## Corpus shape

| extension | files |
|---|---|
| `.py` | 250 |
| `.json` | 245 |
| `.md` | 185 |
| `.ts` | 144 |
| `.tsx` | 139 |
| `.js` | 69 |
| `.html` | 40 |
| `.yml` | 36 |
| `.c` | 32 |
| `.txt` | 31 |
| `.sh` | 22 |
| `.java` | 15 |
| `.php` | 12 |
| `.rb` | 12 |

## Where each family is worst

- **cl100k_base**: worst single file 14.44% off (`.csv`, 90 true, 77 estimated). Median by extension: `.c` 2.813%, `.css` 1.49%, `.html` 2.284%, `.java` 1.071%, `.js` 2.707%, `.json` 1.76%, `.md` 2.007%, `.php` 6.482%, `.py` 3.472%, `.rb` 4.623%, `.ts` 2.558%, `.tsx` 3.93%, `.txt` 3.689%, `.yml` 1.098%.
- **o200k_base**: worst single file 14.44% off (`.csv`, 90 true, 77 estimated). Median by extension: `.c` 1.752%, `.css` 2.447%, `.html` 2.289%, `.java` 1.235%, `.js` 2.955%, `.json` 2.061%, `.md` 1.995%, `.php` 6.085%, `.py` 3.251%, `.rb` 3.849%, `.ts` 2.573%, `.tsx` 1.973%, `.txt` 6.407%, `.yml` 0.981%.
- **qwen2.5**: worst single file 12.83% off (`.md`, 304 true, 265 estimated). Median by extension: `.c` 2.815%, `.css` 1.409%, `.html` 2.16%, `.java` 1.068%, `.js` 2.707%, `.json` 1.731%, `.md` 2.065%, `.php` 6.469%, `.py` 3.4%, `.rb` 4.593%, `.ts` 2.831%, `.tsx` 3.936%, `.txt` 3.557%, `.yml` 0.963%.
- **llama3**: worst single file 14.44% off (`.csv`, 90 true, 77 estimated). Median by extension: `.c` 2.884%, `.css` 1.637%, `.html` 2.27%, `.java` 1.071%, `.js` 2.707%, `.json` 1.653%, `.md` 2.054%, `.php` 6.482%, `.py` 3.165%, `.rb` 4.623%, `.ts` 2.997%, `.tsx` 3.868%, `.txt` 3.668%, `.yml` 1.135%.

## What this does not cover

- Claude has no public tokenizer, so no Claude number here is measured. The tool labels Claude counts `unmeasured` and refuses to print an error bar it cannot back.
- The chunk regex is a port of the `cl100k_base` split pattern. The other three families split differently, and the fitted table absorbs that difference rather than reproducing their regexes. The error above is the price of that shortcut, measured.
- Binary files, files over 200 kB and files that are not valid UTF-8 were excluded from the corpus and are outside anything measured here.
