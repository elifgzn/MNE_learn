# This script adapts the A4_2_removeNoise script to MNE.
# Workflow:
#   1. Run the script → inspection plots appear for any participant whose
#      component_exclusions list is still empty (i.e. not yet decided).
#   2. Close the plots, fill in component_exclusions for that participant.
#   3. Run again → no plots for that participant; exclusions are applied
#      and cleaned epochs are saved automatically.

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import read_ica

# ──────────────────────────────────────────────────────────────
# Which participant(s) do you want to process?
# ──────────────────────────────────────────────────────────────
plist = [31]

# ──────────────────────────────────────────────────────────────
# Component exclusions — fill these in after inspecting the plots.
# Keys are participant numbers (integers).
# Values are lists of component indices to remove (0-based).
# Leave the list EMPTY ([]) if you haven't decided yet for that
# participant; the script will then show the inspection plots.
# ──────────────────────────────────────────────────────────────
component_exclusions = {
    31: [0, 1, 2],   # e.g. change to [0, 2] once you've inspected
}

# Paths
input_path  = r"C:\Users\elifg\Desktop\PHD\MNE_learn\MNE_preprocessed"
output_path = r"C:\Users\elifg\Desktop\PHD\MNE_learn\eeg3_clean"   # separate folder for clean data
os.makedirs(output_path, exist_ok=True)   # create folder if it doesn't exist

# ──────────────────────────────────────────────────────────────
# Loop through selected participants
# ──────────────────────────────────────────────────────────────
for sub in plist:

    sub_id   = f"{sub:04d}"
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
    ica    = read_ica(ica_file)

    print(f"  {len(epochs)} epochs, {epochs.info['sfreq']:.0f} Hz, "
          f"{ica.n_components_} ICA components")

    # ── Check whether exclusions have been defined ────────────
    excl = component_exclusions.get(sub, None)   # None = key absent
    needs_inspection = (excl is None) or (len(excl) == 0)

    if needs_inspection:
        # ────────────────────────────────────────────────────────
        # INSPECTION MODE — show plots so you can decide which
        # components to exclude, then fill in component_exclusions
        # above and re-run.
        # ────────────────────────────────────────────────────────
        print(f"  No exclusions defined yet → showing inspection plots.")
        print(f"  Close all plots, fill in component_exclusions[{sub}], "
              f"then re-run.\n")

        # Variance explained per component
        sources   = ica.get_sources(epochs)
        src_data  = sources.get_data()              # (epochs, comps, times)
        comp_var  = np.var(src_data, axis=(0, 2))
        var_pct   = (comp_var / comp_var.sum()) * 100

        # 1) Topoplots
        n_plot = min(20, ica.n_components_)
        figs   = ica.plot_components(picks=range(n_plot), show=False,
                                     title=f"ICA topomaps — participant {sub}")
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

        # 2) Component time courses
        ica.plot_sources(epochs, title=f"ICA sources — participant {sub}")

        # 3) Raw EEG for comparison
        epochs.plot(
            n_channels=20,
            scalings=dict(eeg=100e-6),
            title=f"EEG data BEFORE component removal — participant {sub}"
        )

        plt.show()   # block until all windows are closed

    else:
        # ────────────────────────────────────────────────────────
        # CLEANING MODE — apply exclusions and save.
        # ────────────────────────────────────────────────────────
        print(f"  Excluding components: {excl}")

        ica.exclude = excl
        epochs_clean = ica.apply(epochs.copy())

        # ── Baseline correction ────────────────────────────────
        # Equivalent to FieldTrip:
        #   cfg.baselinewindow = [-0.20 0.00];
        #   cfg.demean = 'yes';
        #   dataNoBlinks = ft_preprocessing(cfg, dataNoBlinks);
        #
        # Subtracts the mean of the −200 ms to 0 ms window from every
        # epoch. Applied after ICA (before artifact rejection).
        epochs_clean.apply_baseline(baseline=(-0.20, 0.0))
        print(f"  ✓ Baseline correction applied (−200 ms to 0 ms)")

        # ── Artifact rejection (general noise) ────────────────
        # Equivalent to FieldTrip:
        #   cfg.artfctdef.threshold.channel = 'all';
        #   cfg.artfctdef.threshold.min = -100;   % µV
        #   cfg.artfctdef.threshold.max = 100;
        #   cfg.artfctdef.reject = 'complete';    % reject whole trial
        #
        # Done AFTER ICA removal (baseline already applied), so that
        # blink-inflated amplitudes don't cause unnecessary trial loss.
        # If one channel is noisy all the time → exclude it upstream
        # (in the preprocessing / importEEG step) rather than here.
        n_before = len(epochs_clean)
        reject_criteria = dict(eeg=100e-6)   # ±100 µV absolute threshold
        epochs_clean.drop_bad(reject=reject_criteria)
        n_after   = len(epochs_clean)
        n_dropped = n_before - n_after
        pct_dropped = round((n_dropped / n_before) * 100, 2)

        # Quality guide (mirrors FieldTrip comments):
        #   0–5 %  → very good
        #   5–10 % → ok
        #   10–15% → acceptable
        #   > 15 % → consider removing more ICA components or a channel
        if pct_dropped <= 5:
            quality = "very good"
        elif pct_dropped <= 10:
            quality = "ok"
        elif pct_dropped <= 15:
            quality = "acceptable"
        else:
            quality = "HIGH — consider removing more components or a bad channel"

        print(f"  Noise exclusions: {n_dropped} / {n_before} trials "
              f"({pct_dropped} %)  [{quality}]")

        # ── Optional: browse post-ICA data (artifacts highlighted) ─
        # Equivalent to FieldTrip's ft_databrowser after ft_rejectartifact.
        # Set show_browser = True to open the interactive viewer.
        show_browser = False
        if show_browser:
            epochs_clean.plot(
                n_channels=20,
                scalings=dict(eeg=100e-6),
                title=f"EEG data AFTER component removal — participant {sub}"
            )
            plt.show()

        # ── Spherical spline interpolation ────────────────────
        # Equivalent to FieldTrip:
        #   cfg.method = 'spline';
        #   cfg.missingchannel = addc;
        #   dataAddChan = ft_channelrepair(cfg, dataClean);
        #
        # Replaces channels that were marked as bad (info['bads']) during
        # preprocessing, using the signal from surrounding electrodes.
        # MNE tracks bad channels in epochs.info['bads'] automatically,
        # and interpolate_bads() restores them in the correct channel order
        # — no manual position-sorting needed (unlike the MATLAB version).
        bads = epochs_clean.info['bads']
        if bads:
            print(f"  Interpolating bad channel(s): {bads}")
            epochs_clean.interpolate_bads(reset_bads=True)  # method='spline' by default for EEG
            print(f"  ✓ Spherical spline interpolation done")
        else:
            print(f"  No bad channels to interpolate")

        # ── Save cleaned epochs ───────────────────────────────
        # Equivalent to FieldTrip:
        #   save(['D:/MCRL DATA/eeg3_clean/MCRL_' num2str(pnum)], 'dataClean', 'pExcludeNoise');
        out_file = os.path.join(output_path, f"MCRL_{sub_id}-epo-clean.fif")
        epochs_clean.save(out_file, overwrite=True)
        print(f"  ✓ Cleaned epochs saved → {out_file}")

        # ── Exclusion-rate log ────────────────────────────────
        # Equivalent to FieldTrip:
        #   noiserate(pnum) = pExcludeNoise;
        #   save('exclusionrate', 'noiserate');
        #
        # Keeps a running JSON file (exclusionrate.json) in eeg3_clean.
        # Each run updates the entry for the current participant;
        # entries for other participants are preserved.
        log_file = os.path.join(output_path, "exclusionrate.json")
        if os.path.exists(log_file):
            import json
            with open(log_file, 'r') as f:
                noiserate = json.load(f)
        else:
            import json
            noiserate = {}
        noiserate[str(sub)] = pct_dropped          # key = participant number as string
        with open(log_file, 'w') as f:
            json.dump(noiserate, f, indent=2)
        print(f"  ✓ Exclusion rate logged → {log_file}")

        print(f"\n  DONE! Participant {sub} fully processed.")
