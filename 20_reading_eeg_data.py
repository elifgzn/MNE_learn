# This script adapts the A4_1_importEEG script (FieldTrip) to MNE.
# Used functions and workflow are matched.

import os
import gc
import mne

# ──────────────────────────────────────────────────────────────
# Which participant(s) to process?
# Use a single-element list (e.g. [31]) to test one participant,
# or list(range(5, 38)) to run all of them.
# ──────────────────────────────────────────────────────────────
plist = [31]  # <-- change this as needed

# ──────────────────────────────────────────────────────────────
# Bad channels per participant (Step 6 from FieldTrip workflow)
# Identified after visual ICA inspection. If a channel is extremely
# noisy, add it here and re-run. Empty list = no bad channels.
# ──────────────────────────────────────────────────────────────
bad_channels = {
    # 17: ['AF7', 'Iz'],
    # 21: ['P8', 'TP8'],
    # 24: ['PO7'],
    # 26: ['CP1'],
    # 27: ['C6'],
    # 30: ['TP8'],
    31: ['P2'],
}

# Input and output folders
input_path = r"C:\Users\elifg\Desktop\PHD\MNE_learn\eeg1_raweeg"
output_path = r"C:\Users\elifg\Desktop\PHD\MNE_learn\MNE_preprocessed"

# Create output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Loop through selected participants (skip 1-4: bad sampling rates / corrupt data)
for sub in plist:
    
    sub_id = f"{sub:04d}"  # formats 1 -> 0001, 32 -> 0032
    filename = f"MCRL_{sub_id}.vhdr"
    filepath = os.path.join(input_path, filename)
    
    if not os.path.exists(filepath):
        print(f"File not found: {filename}")
        continue
    
    print(f"Loading {filename}")
    
    try:
        # Load BrainVision file
        # This automatically reads the .vhdr, .vmrk, and .eeg files as a single Raw object.
        raw = mne.io.read_raw_brainvision(filepath, preload=True)
        
        # Set standard 10-20 montage so MNE knows electrode positions.
        # CHECK THIS; THAT IT MATCHES OUR ELECTRODE POSITIONS!!
        # This is needed for interpolation and for correct topoplots.
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='warn')
        
        # Remove bad channels (Step 6 from FieldTrip)
        # Mark noisy electrodes, interpolate them from neighbours,
        # then continue with the clean data.
        bads = bad_channels.get(sub, [])
        if bads:
            raw.info['bads'] = bads
            raw.interpolate_bads(reset_bads=True)
            print(f"  Interpolated bad channels: {bads}")
        
        # Apply 1–40 Hz band-pass filter
        # In MNE, l_freq is the lower cutoff frequency and h_freq is the upper cutoff frequency. 
        # So, l_freq specifies the high-pass filter and h_freq specifies the low-pass filter.
        # by default, MNE uses FIR filter. this is very costly takes a lot of time. 
        # fieldtrip default is an IIR butterworth filter. to match our classical fieldtrip workflow and be more efficient, we set an IIR butterworth filter here as well
        raw.filter(
            l_freq=1.,
            h_freq=40.,
            method='iir',
            iir_params=dict(order=4, ftype='butter')
        )
        
        # Re-reference to average of all electrodes
        # Equivalent to FieldTrip:
        #   cfg.implicitref = 'FCz';    → add online reference back as flat channel
        #   cfg.reref       = 'yes';
        #   cfg.refchannel  = 'all';    → common average reference
        #   cfg.refmethod   = 'avg';
        #
        # Step 1: Add the implicit reference channel (FCz) back to the data.
        #   During recording, FCz was the online reference so it's not in the data.
        #   This adds it as a flat (all zeros) channel → 64 becomes 65 channels.
        raw.add_reference_channels('FCz')
        
        # Step 2: Re-reference to the average of all 65 electrodes.
        #   Each channel = original − mean(all channels).
        #   FCz goes from zeros to −mean(all), recovering its actual signal.
        raw.set_eeg_reference('average')
        
        # Epoch the data around feedback onset
        # Equivalent to FieldTrip:
        #   cfg.trialdef.eventtype  = 'Stimulus';
        #   cfg.trialdef.eventvalue = {'S 77', 'S 88', 'S 99', 'S 98'};
        #   cfg.trialdef.prestim    = 2.5;
        #   cfg.trialdef.poststim   = 2.5;
        #
        # Step 1: Extract events from BrainVision annotations.
        #   MNE reads the .vmrk markers as annotations. events_from_annotations
        #   converts them to an (N, 3) array of [sample, 0, event_id].
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        
        # Step 2: Select only our triggers of interest (feedback onset)
        #   S 77 = free choice,  S 88 = forced choice
        #   S 99 = mixed choice, S 98 = mixed choice
        # Note: BrainVision annotations in MNE use the full 'Stimulus/S XX' format.
        wanted_triggers = ['Stimulus/S 77', 'Stimulus/S 88', 'Stimulus/S 99', 'Stimulus/S 98']
        triggers = {}
        for t in wanted_triggers:
            if t in event_id:
                triggers[t] = event_id[t]
            else:
                print(f"  WARNING: trigger '{t}' not found in {filename}")
        
        if not triggers:
            print(f"  SKIPPED — no matching triggers found")
            continue
        
        # Step 3: Create epochs: -2.5 s before to +2.5 s after feedback onset
        epochs = mne.Epochs(
            raw, events, event_id=triggers,
            tmin=-2.5, tmax=2.5,
            preload=True, verbose=False
        )
        
        # Downsample to 250 Hz
        # Equivalent to FieldTrip:
        #   cfg.resamplefs = 250;
        #   cfg.detrend    = 'no';
        #   eegdata = ft_resampledata(cfg, eegdata);
        #
        # MNE automatically applies an anti-aliasing lowpass filter before
        # downsampling (1000 Hz → 250 Hz = 4× reduction).
        # We already filtered at 40 Hz, well below the new Nyquist (125 Hz).
        epochs.resample(250, verbose=False)
        
        print(f"  {len(epochs)} epochs, {epochs.info['sfreq']:.0f} Hz ({dict(epochs.metadata['event_name'].value_counts()) if hasattr(epochs, 'metadata') and epochs.metadata is not None else {k: (epochs.events[:, 2] == v).sum() for k, v in triggers.items()}})")

        # In our fieldtrip workflow, we would now switch to single precision.
        # this means, we would convert our data to float32 from float64 to save memory.
        # we have asked python to do this above before filtering, while loading the data (fmt='single')
        # In MNE, the data stays float64 in memory (for numerical accuracy during processing)
        # but is saved as float32 via fmt='single' in the save step below.

        # In our fieldtrip workflow, there is a block here where we update trigger info according to downsampling:
        # for tnum=1:numtrials
        # tinfoCue.trl(tnum,2) = tinfoCue.trl(tnum,1) + size(eegdata.trial{tnum},2)-1;
        # tinfoCue.trl(tnum,3) = -1*(find(eegdata.time{1}>=0,1)-1); %offset = samples before point 0
        # end
        # this is needed because fieldtrip sometimes tracks time in secodns instead of trials, so we need to match trials and triggers manually. 
        # We don't need to do this in MNE because MNE keeps track of the original sample indices.
        
        # ICA for artifact correction (eyeblinks)
        # Equivalent to FieldTrip:
        #   cfg.method = 'runica';         → uses EEGLAB's runica algorithm
        #   cfg.runica.extended = 0;
        #   components = ft_componentanalysis(cfg, eegdata);
        #
        # In MNE, 'infomax' is the same algorithm as FieldTrip's 'runica'.
        # ICA decomposes the data into independent components. Components
        # capturing eyeblinks/eye movements will be identified and rejected
        # in a later interactive step.
        from mne.preprocessing import ICA
        ica = ICA(
            method='infomax',           # same as runica
            fit_params=dict(extended=False),  # matches cfg.runica.extended = 0
            random_state=42             # for reproducibility
        )
        ica.fit(epochs, verbose=False)
        print(f"  ICA fitted: {ica.n_components_} components")
        
        # Save both the epoched data and the ICA solution
        # (same as FieldTrip saving EEG and COMP separately)
        # Component rejection will happen later in a separate interactive step.
        epo_name = f"MCRL_{sub_id}-epo.fif"
        ica_name = f"MCRL_{sub_id}-ica.fif"
        epochs.save(os.path.join(output_path, epo_name), overwrite=True, fmt='single')
        ica.save(os.path.join(output_path, ica_name), overwrite=True)
        
        print(f"  Saved {epo_name} + {ica_name}")
        
        # Free RAM before loading the next participant
        del raw, epochs, ica
        gc.collect()
        
    except Exception as e:
        print(f"  FAILED {filename}: {e}")

print("All done!")
