# This script adapts the A4_2_removeNoise script to MNE.
# It loads the saved epoched data and ICA solution, then creates
# topoplots and time-course plots to visually identify which
# ICA components to remove (e.g. eyeblinks, eye movements).

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import read_ica

# ──────────────────────────────────────────────────────────────
# Which participant(s) do you want to inspect?
# Change this list to run one or several participants.
# ──────────────────────────────────────────────────────────────
plist = [31]

# Paths
input_path = r"C:\Users\elifg\Desktop\PHD\MNE_learn\MNE_preprocessed"

# ──────────────────────────────────────────────────────────────
# Loop through selected participants
# ──────────────────────────────────────────────────────────────
for sub in plist:

    sub_id = f"{sub:04d}"
    epo_file = os.path.join(input_path, f"MCRL_{sub_id}-epo.fif")
    ica_file = os.path.join(input_path, f"MCRL_{sub_id}-ica.fif")

    # Check that both files exist
    if not os.path.exists(epo_file):
        print(f"Epochs file not found: {epo_file}")
        continue
    if not os.path.exists(ica_file):
        print(f"ICA file not found: {ica_file}")
        continue

    print(f"\n{'='*60}")
    print(f"  Participant {sub}  (sub_id = {sub_id})")
    print(f"{'='*60}")

    # ── Load data ─────────────────────────────────────────────
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
    ica = read_ica(ica_file)

    print(f"  {len(epochs)} epochs, {epochs.info['sfreq']:.0f} Hz, "
          f"{ica.n_components_} ICA components")

    # ── Compute variance explained per component ──────────────
    # Like FieldTrip, shows what % of total variance each component
    # accounts for. Helpful for deciding which components to remove.
    sources = ica.get_sources(epochs)
    source_data = sources.get_data()           # (n_epochs, n_components, n_times)
    comp_var = np.var(source_data, axis=(0, 2)) # variance per component
    var_pct = (comp_var / comp_var.sum()) * 100  # as percentage

    # ── 1) ICA component topoplots with variance % ────────────
    # Equivalent to FieldTrip:
    #   cfg.component = 1:20;
    #   ft_topoplotIC(cfg, components)
    #
    # This shows the spatial pattern (scalp map) of each component.
    # Look for components with frontal dipolar patterns (eyeblinks)
    # or lateral patterns (horizontal eye movements).
    n_plot = min(20, ica.n_components_)
    figs = ica.plot_components(picks=range(n_plot), show=False,
                               title=f"ICA topomaps — participant {sub}")

    # Patch each subplot title to include variance explained
    if not isinstance(figs, list):
        figs = [figs]
    for fig in figs:
        for ax in fig.axes:
            title = ax.get_title()
            if title.startswith('ICA'):
                comp_idx = int(title[3:])
                ax.set_title(f"{title} ({var_pct[comp_idx]:.1f}%)")
    for fig in figs:
        fig.show()

    # ── 2) ICA component time courses ─────────────────────────
    # Equivalent to FieldTrip:
    #   cfg.viewmode = 'component';
    #   ft_databrowser(cfg, components)
    #
    # This opens an interactive browser showing component activations
    # over time. Eyeblink components typically show sharp spikes.
    ica.plot_sources(epochs, title=f"ICA sources — participant {sub}")

    # ── 3) Original EEG data for comparison ───────────────────
    # Equivalent to FieldTrip:
    #   cfg.viewmode = 'vertical';
    #   ft_databrowser(cfg, eegdata)
    #
    # Browse the epoched EEG to see what the raw signal looks like
    # before any component removal.
    epochs.plot(
        n_channels=20,
        scalings=dict(eeg=100e-6),
        title=f"EEG data BEFORE component removal — participant {sub}"
    )

    print(f"\n  → Inspect the plots above.")
    print(f"  → Note down which component(s) to exclude (e.g. 0, 1).")
    print(f"  → You will mark them for removal in the next step.\n")

    # Keep all plot windows open until you close them
    plt.show()
