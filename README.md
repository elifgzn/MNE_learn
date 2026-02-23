# MNE EEG Preprocessing & ERP Analysis Pipeline

This project adapts an EEG preprocessing and ERP analyses pipeline that was prepared on MATLAB using FieldTrip, to MNE on Python. It exactly matches FieldTrip functions and preprocessing order.

## Pipeline Steps

| Step | MNE (Python) | FieldTrip (MATLAB) |
|------|-------------|-------------------|
| Load data | `mne.io.read_raw_brainvision()` | `ft_preprocessing()` |
| Bandpass filter (1–40 Hz) | `raw.filter()` (IIR Butterworth) | `ft_preprocessing()` |
| Add implicit reference (FCz) | `raw.add_reference_channels('FCz')` | `cfg.implicitref = 'FCz'` |
| Average re-reference | `raw.set_eeg_reference('average')` | `ft_preprocessing()` |
| Epoch (±2.5 s) | `mne.Epochs()` | `ft_redefinetrial()` |
| Downsample (250 Hz) | `epochs.resample(250)` | `ft_resampledata()` |
| ICA (eyeblink correction) | `mne.preprocessing.ICA(method='infomax')` | `ft_componentanalysis(cfg.method='runica')` |

## Scripts

- `20_reading_eeg_data.py` — Main preprocessing pipeline (adapts `A4_1_importEEG.m`)
- `25_background_filtering.py` — Background notes on filtering theory
