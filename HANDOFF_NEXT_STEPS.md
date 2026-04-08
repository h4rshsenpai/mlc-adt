# Handoff Next Steps

Current state:
- GROOVE is fully present under `data/groove/`.
- `data/drum_classifier.pkl` has been rebuilt from the full GROOVE corpus.
- The model supports five labels: `KD`, `SD`, `HH`, `TT`, `CY`.
- Full-dataset comparison against GROOVE ground truth is complete.

Key finding:
- The near-zero `CY` problem was traced to a train/inference sample-rate mismatch.
- Training extracted features at `22050 Hz` but inference loaded audio at native sample rate.
- After enforcing a shared model sample rate end-to-end, `CY` recovered substantially.

Code changes applied:
- Added shared `MODEL_SAMPLE_RATE = 22050` in `src/classification.py`.
- Updated `detect_onsets()` to load at that target sample rate in `src/onset.py`.
- Updated training audio loading to use `MODEL_SAMPLE_RATE` in `src/scripts/train_classifier.py`.

Full-corpus count comparison (after fix):
- Usable tracks processed: `1090`
- Missing audio rows skipped: `60`
- Predicted label counts:
  - `KD`: `42063`
  - `SD`: `52896`
  - `HH`: `28990`
  - `TT`: `18307`
  - `CY`: `26045`
- Ground-truth label counts:
  - `KD`: `68606`
  - `SD`: `97391`
  - `HH`: `67031`
  - `TT`: `25447`
  - `CY`: `43931`

End-to-end precision/recall (full corpus, onset-matched at ±30ms):
- Micro:
  - Precision: `0.455541`
  - Recall: `0.253527`
- Per class:
  - `KD`: Precision `0.437962`, Recall `0.268519`
  - `SD`: Precision `0.567491`, Recall `0.308221`
  - `HH`: Precision `0.313073`, Recall `0.135400`
  - `TT`: Precision `0.489376`, Recall `0.352065`
  - `CY`: Precision `0.391361`, Recall `0.232023`

Current interpretation:
- `CY` collapse is fixed (moved from `6` predicted to `26045` predicted on full corpus).
- Overall recall is still low, with `HH` recall especially weak.
- Remaining bottlenecks are likely onset matching/coverage and feature-window design rather than label mapping.

What to do next:
1. Tune onset detection for higher recall (especially to recover `HH`): revisit tempo-derived `wait`, and test smaller refractory windows.
2. Try feature-window changes for cymbal/hat transients (e.g., add pre-onset context and/or longer windows).
3. Add a reproducible corpus-eval script in `src/scripts/` so these exact aggregate metrics can be re-run without ad hoc one-liners.
4. Re-run full-corpus precision/recall after each change and track deltas by class (`HH`, `CY`, `TT` first).

Helpful commands:
```bash
UV_CACHE_DIR=$PWD/uv_cache uv run python src/scripts/train_classifier.py
UV_CACHE_DIR=$PWD/uv_cache uv run python src/scripts/test_suite.py
UV_CACHE_DIR=$PWD/uv_cache MPLCONFIGDIR=$PWD/.mplconfig uv run python main.py drums.wav --zoom 0:10
```
