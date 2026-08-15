# Calibration, measured not asserted

Fitted and measured on 1200 real text files taken read-only from repositories on the machine that built this project. 916 files fitted the table, 284 were held out and never seen by the fit. File paths are deliberately not recorded here; only counts and extensions are.

Held-out error, whole-file token count, estimator against the real tokenizer:

| family | tokenizer | files | held-out tokens | median | p95 | worst | chars/4 median | chars/4 p95 |
|---|---|---|---|---|---|---|---|---|
| `cl100k_base` | tiktoken cl100k_base (tiktoken 0.13.0) | 283 | 685,488 | 3.162% | 8.751% | 18.807% | 10.214% | 30.897% |
| `o200k_base` | tiktoken o200k_base (tiktoken 0.13.0) | 283 | 689,006 | 3.043% | 10.107% | 17.326% | 9.535% | 30.741% |
| `qwen2.5` | Qwen2.5 tokenizer.json (tokenizers 0.23.1) | 284 | 730,875 | 3.106% | 8.955% | 18.0% | 11.181% | 36.952% |
| `llama3` | Llama 3 tokenizer.json (tokenizers 0.23.1) | 283 | 685,068 | 3.093% | 8.882% | 18.783% | 10.214% | 30.897% |

The last two columns are the thing this replaces. Dividing characters by four is the usual shortcut, and its error on the same held-out files is there beside ours.

## Corpus shape

| extension | files |
|---|---|
| `.py` | 244 |
| `.json` | 240 |
| `.md` | 177 |
| `.tsx` | 130 |
| `.ts` | 119 |
| `.js` | 65 |
| `.html` | 38 |
| `.c` | 34 |
| `.txt` | 32 |
| `.yml` | 31 |
| `.sh` | 23 |
| `.java` | 15 |
| `.php` | 12 |
| `.rb` | 12 |

## Where each family is worst

- **cl100k_base**: worst single file 18.81% off (`.json`, 5934 true, 4818 estimated). Median by extension: `.c` 2.453%, `.css` 2.331%, `.csv` 6.198%, `.html` 1.721%, `.java` 4.051%, `.js` 2.579%, `.json` 2.342%, `.md` 4.849%, `.php` 5.764%, `.py` 3.2%, `.rb` 2.305%, `.ts` 2.899%, `.tsx` 3.414%, `.txt` 9.013%, `.yml` 0.506%.
- **o200k_base**: worst single file 17.33% off (`.json`, 5864 true, 4848 estimated). Median by extension: `.c` 2.073%, `.css` 1.799%, `.csv` 4.916%, `.html` 1.626%, `.java` 3.303%, `.js` 2.797%, `.json` 3.039%, `.md` 5.333%, `.php` 4.449%, `.py` 3.136%, `.rb` 2.245%, `.ts` 2.275%, `.tsx` 1.82%, `.txt` 12.951%, `.yml` 1.577%.
- **qwen2.5**: worst single file 18.0% off (`.json`, 6200 true, 5084 estimated). Median by extension: `.c` 2.448%, `.css` 2.204%, `.csv` 4.828%, `.html` 1.624%, `.java` 4.03%, `.js` 2.76%, `.json` 2.235%, `.md` 4.711%, `.php` 5.755%, `.py` 3.153%, `.rb` 2.298%, `.ts` 2.786%, `.tsx` 3.315%, `.txt` 8.299%, `.yml` 0.844%.
- **llama3**: worst single file 18.78% off (`.json`, 5931 true, 4817 estimated). Median by extension: `.c` 2.546%, `.css` 2.191%, `.csv` 6.198%, `.html` 1.7%, `.java` 4.16%, `.js` 2.778%, `.json` 2.342%, `.md` 4.856%, `.php` 5.764%, `.py` 3.177%, `.rb` 2.305%, `.ts` 2.899%, `.tsx` 3.414%, `.txt` 9.024%, `.yml` 0.507%.

## What this does not cover

- Claude has no public tokenizer, so no Claude number here is measured. The tool labels Claude counts `unmeasured` and refuses to print an error bar it cannot back.
- The chunk regex is a port of the `cl100k_base` split pattern. The other three families split differently, and the fitted table absorbs that difference rather than reproducing their regexes. The error above is the price of that shortcut, measured.
- Binary files, files over 200 kB and files that are not valid UTF-8 were excluded from the corpus and are outside anything measured here.
