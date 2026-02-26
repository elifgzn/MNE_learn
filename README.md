# MNE EEG Preprocessing & ERP Analysis Pipeline

This project adapts an EEG preprocessing and ERP analysis pipeline from MATLAB/FieldTrip to MNE-Python. It exactly matches FieldTrip functions, preprocessing order, and statistical testing methodologies.

## Pipeline Steps

| Step | Script | MNE (Python) | FieldTrip (MATLAB) |
|------|--------|-------------|-------------------|
| Load data | `1_import_raw.py` | `mne.io.read_raw_brainvision()` | `ft_preprocessing()` |
| Bandpass filter (1–40 Hz) | `1_import_raw.py` | `raw.filter()` (IIR Butterworth) | `ft_preprocessing()` |
| Add implicit reference (FCz) | `1_import_raw.py` | `raw.add_reference_channels('FCz')` | `cfg.implicitref = 'FCz'` |
| Average re-reference | `1_import_raw.py` | `raw.set_eeg_reference('average')` | `ft_preprocessing()` |
| Epoch (±2.5 s) | `1_import_raw.py` | `mne.Epochs()` | `ft_redefinetrial()` |
| Downsample (250 Hz) | `1_import_raw.py` | `epochs.resample(250)` | `ft_resampledata()` |
| ICA fit (eyeblink correction) | `1_import_raw.py` | `mne.preprocessing.ICA(method='picard')` | `ft_componentanalysis(cfg.method='runica')` |
| Inspect ICA components | `2_remove_noise.py` | `ica.plot_components()`, `ica.plot_sources()` | `ft_topoplotIC()`, `ft_databrowser()` |
| Remove ICA components | `2_remove_noise.py` | `ica.apply()` | `ft_rejectcomponent()` |
| Baseline correction (−200 ms to 0) | `2_remove_noise.py` | `epochs.apply_baseline((-0.20, 0.0))` | `ft_preprocessing(cfg.demean='yes')` |
| Artifact rejection (> ±100 µV) | `2_remove_noise.py` | `epochs.drop()` via absolute thresholding | `ft_artifact_threshold()` + `ft_rejectartifact()` |
| Spherical spline interpolation | `2_remove_noise.py` | `epochs.interpolate_bads()` | `ft_channelrepair(cfg.method='spline')` |
| ERP extraction & Match Logfile | `3_ERP_calculation.py` | `epochs[mask].average()` -> `Evoked` | `ft_timelockanalysis()` |
| Grand averages & Plotting | `4_ERP_graphs.py` | `mne.grand_average()`, `combine_evoked()` | `ft_timelockgrandaverage()` |
| Cluster Permutation Statistics | `5_ERP_permutation.py` | `permutation_cluster_1samp_test()` | `ft_timelockstatistics()` |

## Dependencies

These scripts were developed and tested with the following environment:
- **Python 3.11+**
- `mne` (Core neuroimaging analysis)
- `numpy`, `pandas`, `scipy` (Data manipulation & statistics)
- `matplotlib` (Plotting)
- `defusedxml` (Required by MNE to safely read BrainVision `.bvef` custom montages)

## Scripts

- `1_import_raw.py` — Preprocessing: load, filter, re-reference, epoch, downsample, fit ICA (adapts `A4_1_importEEG.m`).
- `2_remove_noise.py` — ICA inspection & cleaning: inspect components, apply exclusions, baseline-correct, reject artifacts, interpolate bad channels, save (adapts `A4_2_removeNoise.m`).
- `3_ERP_calculation.py` — ERPs extraction: matches logfile trials, applies behavioral conditions (e.g. High Value), computes Evoked parameters per condition, and saves trial summary counts.
- `4_ERP_graphs.py` — ERP plotting: calculates grand averages, computes difference waves, applies Cousineau-Morey variance correction, and exports line plots & topographical maps (`plot_topomap()`).
- `5_ERP_permutation.py` — Statistics: evaluates Main Effects, Interactions, and Difference waves using Monte-Carlo cluster-based permutation testing. Extracts time limits and Cohen's d effect sizes autonomously for significant clusters.
- `25_background_filtering.py` — Background notes on filtering theory.

## Folder Structure

```
MNE_learn/
├── MNE_preprocessed/          # Output of 1_import_raw.py
│   ├── MCRL_XXXX-epo.fif      # Epoched float32 data per participant
│   └── MCRL_XXXX-ica.fif      # ICA solution per participant
├── eeg3_clean/                # Output of 2_remove_noise.py
│   ├── MCRL_XXXX-epo-clean.fif  # Cleaned epochs per participant
│   └── exclusionrate.json     # Noise exclusion % per participant
├── eeg4_ERPSummaries/         # Output of 3_ERP_calculation.py
│   ├── MCRL_XXXX-erp-ave.fif  # Evoked structures per condition/participant
│   └── MCRL_XXXX-erp-summary.csv # Behavioral trial counts
├── eeg5_figures/              # Output of 4 & 5
│   └── *.svg / *.tiff         # ERP plots, topoplots, and permutation cluster visualizations
└── *.py                       # Scripts
```

## 2_remove_noise.py Workflow

1. **First run** (empty `component_exclusions`): inspection plots appear — topomaps with variance %, source time courses, raw EEG browser.
2. **Fill in** `component_exclusions[participant]` at the top of the script (0-based indices!).
3. **Second run**: no plots — exclusions applied, baseline corrected, artifacts rejected, bad channels interpolated, cleaned epochs saved.

## Important Differences from MATLAB to Keep in Mind

- **Indexing**: Python is 0-based, MATLAB is 1-based. ICA component 0 in Python = component 1 in MATLAB.
- **ICA algorithm**: MNE uses `picard` as a direct, faster optimization of FieldTrip's `runica` Infomax.
- **Downsampling Trial Definitions**: Python inherently keeps correct sample tracking. We do not need offsets to fix sample timings like returning `find(eegdata.time{1}>=0,1)-1`.
- **Plotting & Correction**: Simple line plots of within-subjects data automatically calculate Cousineau-Morey variance scaling.
- **T-test vs F-test**: When comparing two conditions with cluster testing, Python operates using a one-sided paired t-test on the difference, giving computationally identical results to the MATLAB univariate F-test.
