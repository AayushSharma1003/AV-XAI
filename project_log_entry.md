# Session entry — LAV-DF setup notebook design

**Date:** _fill in when you append_
**Phase:** Data setup, notebook 01 rewrite
**Status:** Notebook authored; ready to run on lab machine

## Decisions made this session

### Dataset choice → LAV-DF (confirmed)
- Hosted on HuggingFace (`ControlNet/LAV-DF`), gated — requires HF account + accepting terms.
- Total size: 25.6 GB compressed. Native train/dev/test = 78.7K / 31.5K / 26.1K clips.

### Split strategy → use official splits, NOT custom 80/10/10
- **Rationale:** LAV-DF's official splits are identity-disjoint by construction (inherits VoxCeleb2's actor disjointness). Using them directly:
  1. Gives apples-to-apples comparability to BA-TFD / BA-TFD+ baselines reported in the original LAV-DF paper.
  2. Avoids reinventing actor-disjoint hashing.
- The notebook still **verifies** disjointness as a safety check (asserts zero actor overlap across splits).
- Trade-off accepted: ratios end up ~58/23/19, not 80/10/10. The handoff plan originally said 80/10/10, but that decision was made before knowing LAV-DF already had clean official splits.

### Subsample size → 30K with 4-class balance per split
- `TARGET_TOTAL = 30_000`, distributed as ~17.4K / 6.9K / 5.7K across train / val / test (proportional to native ratio).
- Within each split, equal counts per `modify_type` class (real / visual_modified / audio_modified / both_modified).
- If any cell has fewer entries than the target, the notebook uses all available and prints a warning rather than upsampling (avoids label-leak from duplicates).

### Download strategy → metadata-first, then selective
- Phase B downloads only `metadata.min.json` (33 MB) for schema verification.
- After manifest is built, Phase G downloads **only the ~30K MP4s in the manifest** via `hf_hub_download` with an 8-thread pool.
- Estimated payload: ~5.7 GB raw (down from 25.6 GB if we pulled everything). Saves ~20 GB of disk + ~80% of bandwidth.

## Schema assumptions baked into the notebook
Required fields in each LAV-DF metadata entry (asserted in Phase C):
- `file` (str) — relative path under dataset root
- `modify_type` (str) — one of `real`, `visual_modified`, `audio_modified`, `both_modified`
- `split` (str) — one of `train`, `dev`, `test`
- `original` (str) — VoxCeleb2 source path (used to extract actor_id via `idXXXXX/...` prefix)

Optional fields used if present:
- `fake_periods` (list of `[start_s, end_s]`) — for future temporal localization work
- `n_fakes` (int)
- `audio_model`, `audio_fake_segments`, `visual_fake_segments` (kept in metadata; not yet used)

## Manifest schema (output of notebook 01)
`data/manifest.csv` and `data/splits/{train,val,test}.csv` columns:

| column | type | meaning |
|---|---|---|
| `file` | str | relative path, e.g. `train/000001.mp4` |
| `split` | str | `train` / `val` / `test` (LAV-DF's `dev` renamed → `val`) |
| `modify_type` | str | the 4-class string label |
| `label_4class` | int | 0=real, 1=visual, 2=audio, 3=both |
| `label_binary` | int | 0=real, 1=any-fake |
| `actor_id` | str | VoxCeleb2 speaker id (e.g. `id00012`) |
| `fake_periods` | str | JSON-serialized `[[start, end], ...]` in seconds |
| `n_fakes` | int | count of fake segments |
| `original` | str | source VoxCeleb2 path |

## Open items / things to flag in the paper

- **Split ratio deviation from handoff plan.** Paper section on data should say "we used LAV-DF's official actor-disjoint splits" (cite the LAV-DF CVIU paper), not "we constructed 80/10/10 splits".
- **Subsampling.** Need to justify 30K vs full 136K. Defensible: training compute envelope and the ablation studies we plan. Mention upfront in §3.1.
- **`both_modified` may be under-represented** — if Phase D shows fewer than 7,500 entries in any (split, both_modified) cell, the notebook will warn and use all available. Document the actual final counts in the paper.

## Files written / changed this session

- `notebooks/01_setup_and_download_lavdf.ipynb` (new, 19 cells)
- `project_log.md` (this entry appended)

## Git commit suggestion

```
git add notebooks/01_setup_and_download_lavdf.ipynb project_log.md
git commit -m "nb01: rewrite for LAV-DF with HF gated download + official actor-disjoint splits

- Replace Celeb-DF / DFDC download logic with HuggingFace selective fetch
- Use LAV-DF official train/dev/test splits (already identity-disjoint via VoxCeleb2)
- Subsample to 30K balanced across 4 modify_type classes per split
- Smoke-test pattern: metadata-only download first, schema assert, then selective video pull
- Estimated payload ~5.7 GB (vs 25.6 GB for full dataset)"
git push
```

## Things to verify when running on Windows lab machine

1. `pip install huggingface_hub` in the `df` conda env (probably missing).
2. `huggingface-cli login` once, with the read token from https://huggingface.co/settings/tokens.
3. Make sure you have clicked **Agree and access** at https://huggingface.co/datasets/ControlNet/LAV-DF while logged into the SAME account.
4. Phase B (metadata download) takes ~5 seconds. If it 403s, the issue is one of the three above.
5. Phase G (video download) — bandwidth-bound. With 8 workers, expect 30–90 minutes depending on lab network. Resumable.
