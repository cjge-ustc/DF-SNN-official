# Dataset Preparation

Use the event datasets, not similarly named RGB datasets. Download files from the authors' official pages linked in the root README. Dataset licenses/access terms remain with their providers; this repository does not redistribute data.

## Starting From Original Archives

Create each dataset root and its `download/` directory. Keep the original files with these exact names inside `download/`:

| Dataset | Files expected by the dataset adapter |
|:---|:---|
| HAR-DVS | `MINI_HARDVS_files.zip`, `train_label.txt`, `val_label.txt`, `test_label.txt` |
| DVS128 Gesture | `DvsGesture.tar.gz`, `gesture_mapping.csv`, `LICENSE.txt`, `README.txt` |
| N-Caltech101 | `Caltech101.zip`, `Caltech101_annotations.zip`, `ReadMe(Caltech101)-SINAPSE-G.txt`, `ReadMe(Caltech101).txt` |
| DVS-Lip | `DVS-Lip.zip` |

The HAR-DVS adapter expects the compact event archive containing `MINI_HARDVS_files/action_001.zip` through `action_300.zip`, with event `.npz` samples, plus the three split-label files. Do not use pre-rendered event images or silently substitute another split. Official mirrors can change their packaging: verify it matches this adapter before conversion. Its extraction step needs space for both archives and extracted copies.

The DVS-Lip archive must expand to `DVS-Lip/train/<class>/...` and `DVS-Lip/test/<class>/...`. N-Caltech101 is converted by the dataset loader and split with a fixed 80/20 random split (`split_seed=42`), not an independently downloaded train/test split.

On first use, the loader checks archive MD5s, extracts files, creates `events_np/`, then integrates frames. If an official mirror repackages files and checksums differ, do not disable integrity checks; obtain matching original files or separately validate a converted event cache. Avoid running concurrent conversions into the same root. Interrupted conversion can leave an incomplete directory which the loader will reuse; retain/inspect it before manually rebuilding any cache.

## Existing Prepared Caches

Point `--data-path` at the parent containing **both** `events_np/` and `frames_number_T_split_by_number/`. The loader checks `events_np/` before the frame cache, so a frame-only directory is not sufficient.

```text
HAR-DVS/
  events_np/{train,val,test}/action_001/...npz
  frames_number_8_split_by_number/{train,val,test}/action_001/...npz
DVS128Gesture/
  events_np/{train,test}/<class>/...npz
  frames_number_16_split_by_number/{train,test}/<class>/...npz
N-Caltech101/
  events_np/<class>/...npz
  frames_number_16_split_by_number/<class>/...npz
DVS-Lip/
  events_np/{train,test}/<class>/...npz
  frames_number_16_split_by_number/{train,test}/<class>/...npz
```

Each frame `.npz` contains `frames` with shape `[T, 2, H, W]`. Preserve all classes and class-directory ordering. HAR-DVS must be integrated directly to eight frames with `split_by='number'`, not produced by truncating a sixteen-frame cache. Other presets use sixteen frames. Trainers resize inputs to 128 x 128; no manual resize of the stored events is required.

| Dataset | Train samples | Test samples | Classes |
|:---|---:|---:|---:|
| HAR-DVS | 64,519 | 32,386 | 300 |
| DVS128 Gesture | 1,176 | 288 | 11 |
| N-Caltech101 | 6,967 | 1,742 | 101 |
| DVS-Lip | 14,896 | 4,975 | 100 |

HAR-DVS uses the official train and test splits; validation is not merged into training. Test loaders use `drop_last=False`. Training uses `drop_last=True` and therefore may omit a partial final batch. `tests/check_data.py` verifies these dataset lengths and the first/last samples in each split, without changing files. It does not prove that every sample is readable or run complete model inference.
