# Calibration, measured not asserted

Fitted and measured on 1260 real text files taken read-only from repositories on the machine that built this project. 949 files fitted the table, 311 were held out and never seen by the fit. 60 of the files were pulled in deliberately because they contain CJK text, since a hash-ordered sample of this machine turned up almost none and the CJK buckets were going unfitted. File paths are deliberately not recorded here; only counts and extensions are.

Held-out error, whole-file token count, estimator against the real tokenizer:

| family | tokenizer | files | held-out tokens | median | p95 | worst | chars/4 median | chars/4 p95 |
|---|---|---|---|---|---|---|---|---|
| `cl100k_base` | tiktoken cl100k_base (tiktoken 0.13.0) | 310 | 786,077 | 2.993% | 8.229% | 15.556% | 9.378% | 31.073% |
| `o200k_base` | tiktoken o200k_base (tiktoken 0.13.0) | 310 | 786,156 | 2.868% | 8.903% | 15.561% | 8.857% | 30.086% |
| `qwen2.5` | Qwen2.5 tokenizer.json (tokenizers 0.23.1) | 311 | 829,011 | 2.941% | 8.205% | 13.462% | 10.298% | 36.588% |
| `llama3` | Llama 3 tokenizer.json (tokenizers 0.23.1) | 310 | 783,523 | 2.981% | 8.229% | 15.556% | 9.378% | 30.769% |

The last two columns are the thing this replaces. Dividing characters by four is the usual shortcut, and its error on the same held-out files is there beside ours.

## Corpus shape

| extension | files |
|---|---|
| `.py` | 248 |
| `.json` | 246 |
| `.md` | 186 |
| `.ts` | 144 |
| `.tsx` | 139 |
| `.js` | 67 |
| `.html` | 41 |
| `.yml` | 37 |
| `.c` | 32 |
| `.txt` | 31 |
| `.sh` | 22 |
| `.java` | 15 |
| `.php` | 12 |
| `.rb` | 12 |

## Where each family is worst

- **cl100k_base**: worst single file 15.56% off (`.csv`, 90 true, 76 estimated). Median by extension: `.c` 2.405%, `.css` 2.836%, `.csv` 6.675%, `.html` 1.46%, `.java` 3.959%, `.js` 2.312%, `.json` 2.425%, `.md` 3.726%, `.php` 4.937%, `.py` 3.092%, `.rb` 2.68%, `.ts` 2.434%, `.tsx` 2.453%, `.txt` 5.421%, `.yml` 0.202%.
- **o200k_base**: worst single file 15.56% off (`.txt`, 1311 true, 1515 estimated). Median by extension: `.c` 1.948%, `.css` 2.327%, `.csv` 5.396%, `.html` 1.343%, `.java` 3.109%, `.js` 2.226%, `.json` 3.149%, `.md` 5.653%, `.php` 3.743%, `.py` 3.009%, `.rb` 2.219%, `.ts` 1.834%, `.tsx` 1.504%, `.txt` 8.454%, `.yml` 1.166%.
- **qwen2.5**: worst single file 13.46% off (`.csv`, 104 true, 90 estimated). Median by extension: `.c` 2.386%, `.css` 2.681%, `.csv` 5.2%, `.html` 1.321%, `.java` 3.934%, `.js` 2.139%, `.json` 2.128%, `.md` 3.716%, `.php` 4.929%, `.py` 3.069%, `.rb` 2.672%, `.ts` 2.559%, `.tsx` 2.494%, `.txt` 5.714%, `.yml` 0.563%.
- **llama3**: worst single file 15.56% off (`.csv`, 90 true, 76 estimated). Median by extension: `.c` 2.498%, `.css` 2.696%, `.csv` 6.675%, `.html` 1.453%, `.java` 3.959%, `.js` 2.322%, `.json` 2.249%, `.md` 3.679%, `.php` 4.937%, `.py` 3.092%, `.rb` 2.68%, `.ts` 2.653%, `.tsx` 2.453%, `.txt` 5.261%, `.yml` 0.203%.

## What this does not cover

- Claude has no public tokenizer, so no Claude number here is measured. The tool labels Claude counts `unmeasured` and refuses to print an error bar it cannot back.
- The chunk regex is a port of the `cl100k_base` split pattern. The other three families split differently, and the fitted table absorbs that difference rather than reproducing their regexes. The error above is the price of that shortcut, measured.
- Binary files, files over 200 kB and files that are not valid UTF-8 were excluded from the corpus and are outside anything measured here.
