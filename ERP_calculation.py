import os
import numpy as np
import pandas as pd
import mne

# ──────────────────────────────────────────────────────────────
# Which participant(s) do you want to process?
# ──────────────────────────────────────────────────────────────
plist = [31]

# Paths
eeg_path = r"C:\Users\elifg\Desktop\PHD\MNE_learn\eeg3_clean"
log_path = r"C:\Users\elifg\Desktop\LMU\thesis\main_logdat\main_logdat"

# Loop through selected participants
for sub in plist:
    sub_id = f"{sub:04d}"
    epo_file = os.path.join(eeg_path, f"MCRL_{sub_id}-epo-clean.fif")
    log_file = os.path.join(log_path, f"MC_RL{sub}.txt")

    # Check that files exist
    if not os.path.exists(epo_file):
        print(f"Cleaned epochs file not found: {epo_file}")
        continue
    if not os.path.exists(log_file):
        print(f"Behavioral log file not found: {log_file}")
        continue

    print(f"\n{'='*60}")
    print(f"  Participant {sub}")
    print(f"{'='*60}")

    # ── 1. Load data ──────────────────────────────────────────
    # Load cleaned EEG epochs
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
    
    # Load behavioral log file
    logdat = pd.read_csv(log_file)
    print(f"  Loaded EEG: {len(epochs)} trials")
    print(f"  Loaded Log: {len(logdat)} trials")

    # ── 2. Match behavioral and EEG ───────────────────────────
    # The epochs.selection array contains the indices of the trials from the 
    # original trigger list (all events matching S 77, S 88, S 99, S 98) 
    # that survived artifact rejection.
    # We assume these indices map directly to the rows in logdat.
    
    # selection is 0-indexed. logdat.overalltrial is 1-indexed.
    # In 20_reading_eeg_data.py, we created epochs from ALL feedback triggers.
    # So selection [0, 2, 3] means the 1st, 3rd, and 4th feedback triggers were kept.
    # These should correspond to overalltrial 1, 3, 4.
    
    # In MNE, selection refers to the index in the original events array passed to Epochs.
    # Since we passed ONLY the feedback events to Epochs, selection is exactly what we need.
    surviving_indices = epochs.selection
    
    # Filter logdat to only include trials that survived EEG preprocessing
    logdat = logdat.iloc[surviving_indices].copy()
    print(f"  Trials surviving preprocessing: {len(logdat)}")

    # ── 3. Filter for relevant trials ───────────────────────
    # OPTION B): ONLY HV TRIALS (High-Value item chosen)
    # FieldTrip: logdat = logdat(logdat.correctAction == 1,:);
    is_hv = (logdat['correctAction'] == 1)
    logdat = logdat[is_hv].copy()
    
    # Match EEG data accordingly
    # In MNE, we can use the boolean mask directly if it matches the current number of epochs
    epochs = epochs[is_hv.values]
    
    print(f"  Trials after HV filtering: {len(logdat)} (Behavioral == EEG: {len(logdat) == len(epochs)})")

    # ── 4. Sanity Check: Match Triggers ────────────────────
    # Compare logdat.trigger_feedback with epochs.events[:, 2]
    # Note: Trigger values in logdat (77, 88, 99, 98) might be different from 
    # the event IDs MNE assigned if not explicitly handled.
    # In 20_reading_eeg_data.py, event_id was used, mapping 'Stimulus/S 77' to some integer.
    
    # We need to reverse the mapping or check the values.
    # Let's find the values MNE used for these triggers.
    event_id_rev = {v: k for k, v in epochs.event_id.items()}
    
    mismatch_found = False
    for i in range(len(epochs)):
        eeg_trigger_str = event_id_rev[epochs.events[i, 2]] # e.g. 'Stimulus/S 77'
        eeg_trigger_val = int(eeg_trigger_str.split('S ')[1])
        log_trigger_val = logdat.iloc[i]['trigger_feedback']
        
        if eeg_trigger_val != log_trigger_val:
            print(f"  ERROR: Mismatch in Trigger Sequence at trial index {i}!")
            print(f"         EEG Trigger: {eeg_trigger_val}, Log Trigger: {log_trigger_val}")
            mismatch_found = True
            break
            
    if not mismatch_found:
        print(f"  ✓ Sanity Check passed: Logfile and EEG triggers match!")

    # ── 5. Narrow time window ──────────────────────────────
    # FieldTrip: cfg.toilim = [-0.20 1.00];
    epochs.crop(tmin=-0.20, tmax=1.00)
    print(f"  ✓ EEG cropped to window: [-0.20, 1.00] s")

    # ── Summary for this participant ──────────────────────
    num_trials = len(logdat)
    print(f"  DONE! {num_trials} HV trials remaining for ERP analysis.")

print("\nAll selected participants processed.")
