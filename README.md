# MNE EEG Preprocessing & ERP Analysis Pipeline

This project adapts an EEG preprocessing and ERP analysis pipeline from MATLAB/FieldTrip to MNE-Python. It exactly matches FieldTrip functions and preprocessing order.

## Pipeline Steps

| Step | Script | MNE (Python) | FieldTrip (MATLAB) |
|------|--------|-------------|-------------------|
| Load data | `20_reading_eeg_data.py` | `mne.io.read_raw_brainvision()` | `ft_preprocessing()` |
| Bandpass filter (1–40 Hz) | `20_reading_eeg_data.py` | `raw.filter()` (IIR Butterworth) | `ft_preprocessing()` |
| Add implicit reference (FCz) | `20_reading_eeg_data.py` | `raw.add_reference_channels('FCz')` | `cfg.implicitref = 'FCz'` |
| Average re-reference | `20_reading_eeg_data.py` | `raw.set_eeg_reference('average')` | `ft_preprocessing()` |
| Epoch (±2.5 s) | `20_reading_eeg_data.py` | `mne.Epochs()` | `ft_redefinetrial()` |
| Downsample (250 Hz) | `20_reading_eeg_data.py` | `epochs.resample(250)` | `ft_resampledata()` |
| ICA fit (eyeblink correction) | `20_reading_eeg_data.py` | `mne.preprocessing.ICA(method='infomax')` | `ft_componentanalysis(cfg.method='runica')` |
| Inspect ICA components | `remove_noise.py` | `ica.plot_components()`, `ica.plot_sources()` | `ft_topoplotIC()`, `ft_databrowser()` |
| Remove ICA components | `remove_noise.py` | `ica.apply()` | `ft_rejectcomponent()` |
| Baseline correction (−200 ms to 0) | `remove_noise.py` | `epochs.apply_baseline((-0.20, 0.0))` | `ft_preprocessing(cfg.demean='yes')` |
| Artifact rejection (±100 µV) | `remove_noise.py` | `epochs.drop_bad(reject=dict(eeg=100e-6))` | `ft_artifact_threshold()` + `ft_rejectartifact()` |
| Spherical spline interpolation | `remove_noise.py` | `epochs.interpolate_bads()` | `ft_channelrepair(cfg.method='spline')` |
| Save cleaned data | `remove_noise.py` | `epochs.save()` → `eeg3_clean/` | `save(...)` → `eeg3_clean/` |

## Scripts

- `20_reading_eeg_data.py` — Preprocessing pipeline: load, filter, re-reference, epoch, downsample, fit ICA (adapts `A4_1_importEEG.m`)
- `remove_noise.py` — ICA inspection & cleaning pipeline: inspect components, apply exclusions, baseline-correct, reject artifacts, interpolate bad channels, save (adapts `A4_2_removeNoise.m`)
- `25_background_filtering.py` — Background notes on filtering theory

## Folder Structure

```
MNE_learn/
├── MNE_preprocessed/          # Output of 20_reading_eeg_data.py
│   ├── MCRL_XXXX-epo.fif      # Epoched data per participant
│   └── MCRL_XXXX-ica.fif      # ICA solution per participant
├── eeg3_clean/                # Output of remove_noise.py
│   ├── MCRL_XXXX-epo-clean.fif  # Cleaned epochs per participant
│   └── exclusionrate.json     # Noise exclusion % per participant (equiv. of exclusionrate.mat)
└── *.py                       # Scripts
```

## remove_noise.py Workflow

1. **First run** (empty `component_exclusions`): inspection plots appear — topomaps with variance %, source time courses, raw EEG browser.
2. **Fill in** `component_exclusions[participant]` at the top of the script (0-based indices!).
3. **Second run**: no plots — exclusions applied, baseline corrected, artifacts rejected, bad channels interpolated, cleaned epochs saved.

## Important Differences from MATLAB to Keep in Mind

- **Indexing**: Python is 0-based, MATLAB is 1-based. ICA component 0 in Python = component 1 in MATLAB.
