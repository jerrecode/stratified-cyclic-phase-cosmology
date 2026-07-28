# LaTeX paper subproject

`main.tex` is the manuscript root. Scientific figures and compact tables are copied into `generated/` by `scripts/reproduce.py`; they are never manually edited. Paper-specific data descriptions live in `sections/06_data.tex` and `appendices/B_data_manifest.tex`, while the authoritative machine manifest remains `../data/releases.yaml`.

Build with:

```bash
python ../scripts/reproduce.py
latexmk -pdf main.tex
```
