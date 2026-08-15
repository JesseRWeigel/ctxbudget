# Calibration, measured not asserted

Fitted and measured on 1260 real text files taken read-only from repositories on the machine that built this project. 952 files fitted the table, 308 were held out and never seen by the fit. 60 of the files were pulled in deliberately because they contain CJK text, since a hash-ordered sample of this machine turned up almost none and the CJK buckets were going unfitted. File paths are deliberately not recorded here; only counts and extensions are.

Held-out error, whole-file token count, estimator against the real tokenizer:

| family | tokenizer | files | held-out tokens | median | p95 | worst | chars/4 median | chars/4 p95 |
|---|---|---|---|---|---|---|---|---|
| `cl100k_base` | tiktoken cl100k_base (tiktoken 0.13.0) | 307 | 780,294 | 3.081% | 8.882% | 18.908% | 9.385% | 31.073% |
| `o200k_base` | tiktoken o200k_base (tiktoken 0.13.0) | 307 | 780,357 | 2.924% | 9.682% | 17.462% | 8.889% | 30.086% |
| `qwen2.5` | Qwen2.5 tokenizer.json (tokenizers 0.23.1) | 308 | 823,116 | 3.064% | 8.882% | 18.097% | 10.298% | 36.717% |
| `llama3` | Llama 3 tokenizer.json (tokenizers 0.23.1) | 307 | 777,740 | 3.081% | 8.882% | 18.867% | 9.385% | 30.769% |

The last two columns are the thing this replaces. Dividing characters by four is the usual shortcut, and its error on the same held-out files is there beside ours.

## Corpus shape

| extension | files |
|---|---|
| `.py` | 248 |
| `.json` | 245 |
| `.md` | 186 |
| `.ts` | 145 |
| `.tsx` | 139 |
| `.js` | 67 |
| `.html` | 41 |
| `.yml` | 36 |
| `.c` | 33 |
| `.txt` | 31 |
| `.sh` | 22 |
| `.java` | 15 |
| `.php` | 12 |
| `.rb` | 12 |

## Where each family is worst

- **cl100k_base**: worst single file 18.91% off (`.json`, 5934 true, 4812 estimated). Median by extension: `.c` 2.437%, `.css` 2.475%, `.csv` 6.317%, `.html` 1.691%, `.java` 4.021%, `.js` 2.564%, `.json` 2.311%, `.md` 3.604%, `.php` 5.764%, `.py` 3.123%, `.rb` 2.306%, `.ts` 2.812%, `.tsx` 2.586%, `.txt` 8.972%, `.yml` 0.404%.
- **o200k_base**: worst single file 17.46% off (`.json`, 5864 true, 4840 estimated). Median by extension: `.c` 1.995%, `.css` 1.991%, `.csv` 5.036%, `.html` 1.6%, `.java` 3.166%, `.js` 2.568%, `.json` 3.079%, `.md` 6.011%, `.php` 4.353%, `.py` 3.1%, `.rb` 2.06%, `.ts` 1.802%, `.tsx` 1.427%, `.txt` 12.896%, `.yml` 1.399%.
- **qwen2.5**: worst single file 18.1% off (`.json`, 6200 true, 5078 estimated). Median by extension: `.c` 2.422%, `.css` 2.34%, `.csv` 4.921%, `.html` 1.606%, `.java` 3.998%, `.js` 2.548%, `.json` 2.143%, `.md` 3.523%, `.php` 5.755%, `.py` 3.05%, `.rb` 2.299%, `.ts` 2.992%, `.tsx` 2.437%, `.txt` 8.248%, `.yml` 0.75%.
- **llama3**: worst single file 18.87% off (`.json`, 5931 true, 4812 estimated). Median by extension: `.c` 2.53%, `.css` 2.335%, `.csv` 6.317%, `.html` 1.695%, `.java` 4.021%, `.js` 2.564%, `.json` 2.273%, `.md` 3.508%, `.php` 5.535%, `.py` 3.123%, `.rb` 2.306%, `.ts` 2.903%, `.tsx` 2.586%, `.txt` 8.976%, `.yml` 0.405%.

## What this does not cover

- Claude has no public tokenizer, so no Claude number here is measured. The tool labels Claude counts `unmeasured` and refuses to print an error bar it cannot back.
- The chunk regex is a port of the `cl100k_base` split pattern. The other three families split differently, and the fitted table absorbs that difference rather than reproducing their regexes. The error above is the price of that shortcut, measured.
- Binary files, files over 200 kB and files that are not valid UTF-8 were excluded from the corpus and are outside anything measured here.
