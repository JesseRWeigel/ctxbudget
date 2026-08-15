# Calibration, measured not asserted

Fitted and measured on 1200 real text files taken read-only from repositories on the machine that built this project. 917 files fitted the table, 283 were held out and never seen by the fit. File paths are deliberately not recorded here; only counts and extensions are.

Held-out error, whole-file token count, estimator against the real tokenizer:

| family | tokenizer | files | held-out tokens | median | p95 | worst | chars/4 median | chars/4 p95 |
|---|---|---|---|---|---|---|---|---|
| `cl100k_base` | tiktoken cl100k_base (tiktoken 0.13.0) | 282 | 672,657 | 3.115% | 8.771% | 18.824% | 10.211% | 30.897% |
| `o200k_base` | tiktoken o200k_base (tiktoken 0.13.0) | 282 | 676,217 | 3.043% | 10.06% | 17.343% | 9.192% | 30.741% |
| `qwen2.5` | Qwen2.5 tokenizer.json (tokenizers 0.23.1) | 283 | 715,020 | 3.089% | 8.955% | 18.016% | 11.039% | 36.717% |
| `llama3` | Llama 3 tokenizer.json (tokenizers 0.23.1) | 282 | 672,237 | 3.089% | 8.882% | 18.783% | 10.211% | 30.897% |

The last two columns are the thing this replaces. Dividing characters by four is the usual shortcut, and its error on the same held-out files is there beside ours.

## Corpus shape

| extension | files |
|---|---|
| `.py` | 243 |
| `.json` | 240 |
| `.md` | 178 |
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

- **cl100k_base**: worst single file 18.82% off (`.json`, 5934 true, 4817 estimated). Median by extension: `.c` 2.461%, `.css` 2.331%, `.csv` 6.198%, `.html` 1.71%, `.java` 4.051%, `.js` 2.579%, `.json` 2.377%, `.md` 4.833%, `.php` 5.764%, `.py` 3.177%, `.rb` 2.305%, `.ts` 2.899%, `.tsx` 3.414%, `.txt` 9.016%, `.yml` 0.506%.
- **o200k_base**: worst single file 17.34% off (`.json`, 5864 true, 4847 estimated). Median by extension: `.c` 2.002%, `.css` 1.799%, `.csv` 4.916%, `.html` 1.626%, `.java` 3.303%, `.js` 2.773%, `.json` 3.058%, `.md` 5.318%, `.php` 4.449%, `.py` 3.136%, `.rb` 2.245%, `.ts` 2.275%, `.tsx` 1.732%, `.txt` 12.946%, `.yml` 1.566%.
- **qwen2.5**: worst single file 18.02% off (`.json`, 6200 true, 5083 estimated). Median by extension: `.c` 2.448%, `.css` 2.204%, `.csv` 4.828%, `.html` 1.624%, `.java` 4.138%, `.js` 2.76%, `.json` 2.327%, `.md` 4.686%, `.php` 5.755%, `.py` 3.084%, `.rb` 2.298%, `.ts` 2.786%, `.tsx` 3.315%, `.txt` 8.3%, `.yml` 0.844%.
- **llama3**: worst single file 18.78% off (`.json`, 5931 true, 4817 estimated). Median by extension: `.c` 2.554%, `.css` 2.191%, `.csv` 6.317%, `.html` 1.7%, `.java` 4.16%, `.js` 2.778%, `.json` 2.435%, `.md` 4.833%, `.php` 5.764%, `.py` 3.166%, `.rb` 2.305%, `.ts` 2.899%, `.tsx` 3.414%, `.txt` 9.02%, `.yml` 0.507%.

## What this does not cover

- Claude has no public tokenizer, so no Claude number here is measured. The tool labels Claude counts `unmeasured` and refuses to print an error bar it cannot back.
- The chunk regex is a port of the `cl100k_base` split pattern. The other three families split differently, and the fitted table absorbs that difference rather than reproducing their regexes. The error above is the price of that shortcut, measured.
- Binary files, files over 200 kB and files that are not valid UTF-8 were excluded from the corpus and are outside anything measured here.
