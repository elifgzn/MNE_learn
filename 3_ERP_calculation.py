import os
import sys
import numpy as np
import pandas as pd
import mne

# Force UTF-8 output so special characters (e.g. ✓) print cleanly in
# Windows PowerShell terminals (which default to cp1252/cp850)
sys.stdout.reconfigure(encoding='utf-8')

# ──────────────────────────────────────────────────────────────
# Which participant(s) do you want to process?
# ──────────────────────────────────────────────────────────────
plist = [31]

# Paths
eeg_path = r"C:\Users\elifg\Desktop\PHD\MNE_learn\eeg3_clean"
log_path = r"C:\Users\elifg\Desktop\LMU\thesis\main_logdat\main_logdat"

# Accumulates one summary dict per participant (FieldTrip: behavSummary array)
# so we can do a cross-participant trial-count check after the loop.
all_summaries = []

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
    # FieldTrip equivalent:
    #   remainTNums = ismember(logdat.overalltrial, dataClean.trialinfo(:,2));
    #   logdat = logdat(remainTNums,:);
    #
    # The epochs in the cleaned file are the feedback-trigger trials that
    # survived artifact rejection, in chronological order. Their trigger values
    # are in epochs.events[:,2]. The log has ALL trials in the same chronological
    # order, with trigger values in logdat.trigger_feedback.
    #
    # We match by finding which log rows (by their sequential trigger position)
    # correspond to the surviving EEG epochs.
    #
    # NOTE: epochs.selection is NOT used here because it indexes into the
    # original all-markers events array (fixations, responses, etc.), not just
    # the feedback events. Its values are far out of range for logdat.

    # Build a reverse map: event_id int -> trigger number (e.g. 99 -> 99)
    event_id_rev = {v: int(k.split('S ')[1]) for k, v in epochs.event_id.items()}

    # Get the trigger sequence from surviving EEG epochs (chronological order)
    eeg_triggers = np.array([event_id_rev[e] for e in epochs.events[:, 2]])

    # Get the trigger sequence from the full log (chronological order)
    log_triggers = logdat['trigger_feedback'].values  # shape (352,) for all trials

    # Walk through the log and mark which rows have a matching EEG trigger
    # (same logic as ismember in FieldTrip, but sequence-aware)
    eeg_idx = 0  # pointer into eeg_triggers
    log_survived = np.zeros(len(logdat), dtype=bool)

    for log_idx, ltrig in enumerate(log_triggers):
        if eeg_idx < len(eeg_triggers) and eeg_triggers[eeg_idx] == ltrig:
            log_survived[log_idx] = True
            eeg_idx += 1

    if eeg_idx != len(eeg_triggers):
        print(f"  WARNING: Only matched {eeg_idx}/{len(eeg_triggers)} EEG epochs "
              f"to log rows. Check trigger alignment!")
    else:
        print(f"  ✓ All {eeg_idx} EEG epochs matched to log rows via trigger sequence.")

    logdat = logdat[log_survived].copy()
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

    # ── 6. ERP analysis per condition ─────────────────────
    # FieldTrip equivalent (Step 15):
    #   cfgerp.keeptrials = 'no';              → average across trials (no single-trial output)
    #   cfgerp.trials     = <boolean mask>;    → select trials per condition
    #   erp_dat{cnum}     = ft_timelockanalysis(cfgerp, dataClean);
    #
    # In MNE, epochs[mask].average() is the direct equivalent of
    # ft_timelockanalysis with keeptrials='no': it returns an Evoked object
    # containing the trial-averaged ERP for the selected subset.
    #
    # Conditions: control × feedback_valence
    #   control          1 = high control (free choice)   2 = low control (forced/mixed)
    #   feedbackvalence  1 = positive feedback            2 = negative feedback
    # → 4 conditions total: cc1fb1, cc1fb2, cc2fb1, cc2fb2

    # Output folder — create it if it doesn't exist (FieldTrip assumes it exists;
    # we add os.makedirs to be safe, equivalent to a one-time mkdir in the shell)
    erp_out_path = r"C:\Users\elifg\Desktop\PHD\MNE_learn\eeg4_ERPSummaries"
    os.makedirs(erp_out_path, exist_ok=True)

    # We need a reset positional index on logdat so boolean masks align with
    # epochs (which are already in the same order after HV filtering above).
    logdat_reset = logdat.reset_index(drop=True)

    eeg_dat   = {}   # dict of condition_name → mne.Evoked  (FieldTrip: erp_dat cell array)
    cond_name = []   # list of condition label strings       (FieldTrip: cond_name cell array)
    summary   = {'sub': sub}  # trial counts per condition  (FieldTrip: summary struct)

    cnum = 0
    for control in [1, 2]:           # 1 = high control, 2 = low control
        for fb_val in [1, 2]:        # 1 = positive feedback, 2 = negative feedback
            cnum += 1

            # Build condition label, e.g. 'cc1fb1'
            # FieldTrip: cond_name{cnum} = ['cc' num2str(control) 'fb' num2str(feedback_val)];
            label = f"cc{control}fb{fb_val}"
            cond_name.append(label)

            # Boolean mask over the current (HV-filtered, preprocessed) trials
            # FieldTrip: cfgerp.trials = logdat.condition == control & logdat.feedbackvalence == feedback_val;
            mask = (logdat_reset['condition'] == control) & \
                   (logdat_reset['feedbackvalence'] == fb_val)

            n_trials = mask.sum()

            # Store trial count in summary
            # FieldTrip: summary.(['num_' cond_name{cnum}]) = sum(cfgerp.trials);
            summary[f"num_{label}"] = int(n_trials)

            if n_trials == 0:
                print(f"  WARNING: no trials for condition {label} — skipping.")
                eeg_dat[label] = None
                continue

            # Select the matching epochs and average across trials → Evoked object
            # FieldTrip: erp_dat{cnum} = ft_timelockanalysis(cfgerp, dataClean);
            epochs_cond = epochs[mask.values]
            evoked = epochs_cond.average()
            evoked.comment = label   # label the Evoked so it's identifiable when saved
            eeg_dat[label] = evoked

            print(f"  [{label}]  {n_trials} trials  →  ERP computed")

    # ── 7. Save ERP results ───────────────────────────────
    # FieldTrip: save(['D:/MCRL DATA/eeg4_ERPSummaries/MCRL_' addStr num2str(pnum)], ...
    #                 'eegdat', 'cond_name', 'summary');
    #
    # In MNE, Evoked objects are saved as FIF files (one per participant, containing
    # all 4 conditions). The summary dict is saved separately as a CSV.
    # mne.write_evokeds() is the MNE equivalent of saving eegdat in FieldTrip.

    # Save all 4 Evoked objects in a single FIF file (one file per participant)
    evoked_list = [ev for ev in eeg_dat.values() if ev is not None]
    evoked_file = os.path.join(erp_out_path, f"MCRL_{sub_id}-erp-ave.fif")
    mne.write_evokeds(evoked_file, evoked_list, overwrite=True, verbose=False)
    print(f"  ✓ Saved ERP file: {evoked_file}")

    # Save summary (trial counts per condition) as CSV alongside the ERP file
    summary_file = os.path.join(erp_out_path, f"MCRL_{sub_id}-erp-summary.csv")
    pd.DataFrame([summary]).to_csv(summary_file, index=False)
    print(f"  ✓ Saved summary:  {summary_file}")

    print(f"  DONE! Conditions: {cond_name}  |  Trial counts: "
          f"{ {k: summary[f'num_{k}'] for k in cond_name} }")

    # Append this participant's summary to the cross-participant list
    # FieldTrip: behavSummary = [behavSummary summary];
    all_summaries.append(summary)

print("\nAll selected participants processed.")

# ── Cross-participant trial-count check ───────────────────────
# FieldTrip equivalent:
#   trialnumbers = struct2table(behavSummary);        → pd.DataFrame(all_summaries)
#   rows_with_less_trials = any(trialnumbers < 20,2); → .any(axis=1) on condition columns
#   disp([...]);                                      → print(...)
#
# Identifies any participant who has fewer than 25 trials in at least one
# experimental condition — these may need to be excluded from group analysis.
THRESHOLD = 25

if all_summaries:
    summary_df = pd.DataFrame(all_summaries)  # one row per participant

    # Condition columns only (exclude 'sub' identifier column)
    # FieldTrip: trialnumbers = table2array(trialnumbers) operates on the full table;
    # here we explicitly select only the num_* columns to avoid comparing sub IDs.
    cond_cols = [c for c in summary_df.columns if c.startswith('num_')]
    trial_counts = summary_df[cond_cols]

    # Flag rows where ANY condition has fewer than THRESHOLD trials
    # FieldTrip: rows_with_less_trials = any(trialnumbers < 20, 2);
    flagged_mask = (trial_counts < THRESHOLD).any(axis=1)
    flagged_subs = summary_df.loc[flagged_mask, 'sub'].tolist()
    num_flagged  = len(flagged_subs)

    print(f"\n{'='*60}")
    print(f"  TRIAL COUNT CHECK  (threshold: < {THRESHOLD} trials per condition)")
    print(f"{'='*60}")
    # FieldTrip: disp(['Number of rows with a value less than 20: ', num2str(num_rows)]);
    print(f"  Participants with < {THRESHOLD} trials in >= 1 condition: {num_flagged}")

    if num_flagged == 0:
        print("  All participants meet the minimum trial threshold.")
    else:
        # FieldTrip: disp('Row indices:'); disp(row_indices);
        print("  Flagged participant(s):")
        for psub in flagged_subs:
            psub_id = f"{psub:04d}"
            row = summary_df.loc[summary_df['sub'] == psub, cond_cols].iloc[0]
            low_conds = row[row < THRESHOLD].to_dict()
            print(f"    Sub {psub_id}: {low_conds}")
    print(f"{'='*60}\n")
