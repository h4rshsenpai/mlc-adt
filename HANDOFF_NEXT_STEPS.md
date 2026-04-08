# Handoff Next Steps

Current state:
- GROOVE is fully present under `data/groove/`.
- `data/drum_classifier.pkl` has been rebuilt from the full GROOVE corpus.
- The model supports five labels: `KD`, `SD`, `HH`, `TT`, `CY`.
- Full-dataset comparison against GROOVE ground truth is complete.

Key finding:
- The classifier is still strongly biased toward `HH`, `KD`, and `SD`.
- `TT` is under-predicted.
- `CY` is almost completely missing in inference output, even though it is common in ground truth.

Useful numbers from the comparison:
- Usable tracks processed: `1090`
- Missing audio rows skipped: `60`
- Predicted label counts:
  - `KD`: `45984`
  - `SD`: `30204`
  - `HH`: `54057`
  - `TT`: `10522`
  - `CY`: `6`
- Ground-truth label counts:
  - `KD`: `68606`
  - `SD`: `97391`
  - `HH`: `67031`
  - `TT`: `25447`
  - `CY`: `43931`

What to do next:
1. Investigate why `CY` is nearly absent at inference time.
2. Compare training-label frequency against prediction frequency to separate class imbalance from feature or windowing problems.
3. Inspect a few high-cymbal GROOVE tracks and check whether onset timing, slice length, or feature extraction is hiding cymbal energy.
4. If needed, try a larger onset window or different feature set for cymbal-heavy hits.
5. Re-run the full corpus comparison after each tweak to measure whether `CY` and `TT` improve without breaking `KD`, `SD`, or `HH`.

Helpful commands:
```bash
UV_CACHE_DIR=$PWD/uv_cache uv run python src/scripts/train_classifier.py
UV_CACHE_DIR=$PWD/uv_cache uv run python src/scripts/test_suite.py
UV_CACHE_DIR=$PWD/uv_cache MPLCONFIGDIR=$PWD/.mplconfig uv run python main.py drums.wav --zoom 0:10
```
