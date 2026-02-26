import os
import sys
import numpy as np
import pandas as pd
import mne
from mne.stats import permutation_cluster_1samp_test, permutation_cluster_test

# Force UTF-8 output so special characters print cleanly on Windows PowerShell
sys.stdout.reconfigure(encoding='utf-8')

# ──────────────────────────────────────────────────────────────
# Which participant(s) do you want to process?
# ──────────────────────────────────────────────────────────────
plist = [31]

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
# FieldTrip: dfolder = 'D:/MCRL DATA/eeg4_ERPSummaries';
dfolder = r"C:\Users\elifg\Desktop\PHD\MNE_learn\eeg4_ERPSummaries"

# ──────────────────────────────────────────────────────────────
# Condition layout (matches 3_ERP_calculation.py output)
# ──────────────────────────────────────────────────────────────
# FieldTrip: num_cond = 4   (2×2: MOTOR CONTROL × FEEDBACK VALENCE)
#   1 → cc1fb1 = high control + positive feedback
#   2 → cc1fb2 = high control + negative feedback
#   3 → cc2fb1 = low  control + positive feedback
#   4 → cc2fb2 = low  control + negative feedback
num_cond = 4
cond_names = ['cc1fb1', 'cc1fb2', 'cc2fb1', 'cc2fb2']

# ──────────────────────────────────────────────────────────────
# Load data for all participants
# ──────────────────────────────────────────────────────────────
eeg_set = {}   # FieldTrip: eeg_set{p, cond}

for p_idx, pnum in enumerate(plist):
    sub_id = f"{pnum:04d}"
    erp_file = os.path.join(dfolder, f"MCRL_{sub_id}-erp-ave.fif")

    if not os.path.exists(erp_file):
        print(f"  ERP file not found, skipping participant {pnum}: {erp_file}")
        continue

    print(f"Loading participant {pnum}")

    # Load all Evoked objects for this participant
    evoked_list = mne.read_evokeds(erp_file, verbose=False)
    eeg_set[p_idx] = {ev.comment: ev for ev in evoked_list}

# ──────────────────────────────────────────────────────────────
# Create conditions for Main- and Interaction analyses
# ──────────────────────────────────────────────────────────────
main_control    = {}   # FieldTrip: main_control{p, 1:2}
main_valence    = {}   # FieldTrip: main_valence{p, 1:2}
interaction     = {}   # FieldTrip: interaction{p, 1:2}
difference_wave = {}   # FieldTrip: difference_wave{p, 1:2}

for p_idx in eeg_set:
    s = eeg_set[p_idx]  # shorthand: s['cc1fb1'], s['cc1fb2'], etc.

    # ── Main effect: MOTOR CONTROL ────────────────────────────
    mc_high = mne.combine_evoked([s['cc1fb1'], s['cc1fb2']], weights=[0.5, 0.5])
    mc_high.comment = 'high_control'
    mc_low  = mne.combine_evoked([s['cc2fb1'], s['cc2fb2']], weights=[0.5, 0.5])
    mc_low.comment = 'low_control'
    main_control[p_idx] = {'high': mc_high, 'low': mc_low}

    # ── Main effect: FEEDBACK VALENCE ─────────────────────────
    mv_pos = mne.combine_evoked([s['cc1fb1'], s['cc2fb1']], weights=[0.5, 0.5])
    mv_pos.comment = 'positive_fb'
    mv_neg = mne.combine_evoked([s['cc1fb2'], s['cc2fb2']], weights=[0.5, 0.5])
    mv_neg.comment = 'negative_fb'
    main_valence[p_idx] = {'pos': mv_pos, 'neg': mv_neg}

    # ── Interaction: MOTOR CONTROL × FEEDBACK VALENCE ─────────
    int_hc = mne.combine_evoked([s['cc1fb1'], s['cc1fb2']], weights=[1, -1])
    int_hc.comment = 'interaction_hc'
    int_lc = mne.combine_evoked([s['cc2fb1'], s['cc2fb2']], weights=[1, -1])
    int_lc.comment = 'interaction_lc'
    interaction[p_idx] = {'hc': int_hc, 'lc': int_lc}

    # ── Difference waves (for Reward Positivity) ──────────────
    dw_hc = mne.combine_evoked([s['cc1fb1'], s['cc1fb2']], weights=[1, -1])
    dw_hc.comment = 'diffwave_hc'
    dw_lc = mne.combine_evoked([s['cc2fb1'], s['cc2fb2']], weights=[1, -1])
    dw_lc.comment = 'diffwave_lc'
    difference_wave[p_idx] = {'hc': dw_hc, 'lc': dw_lc}

# Print condition names
print("\nCondition labels:")
for i, cn in enumerate(cond_names):
    print(f"  condition {i+1}: {cn}")

# ──────────────────────────────────────────────────────────────
# Specify the effect to test
# ──────────────────────────────────────────────────────────────
# 1 = Main Effect: MOTOR CONTROL (High vs Low)
# 2 = Main Effect: FEEDBACK VALENCE (Positive vs Negative)
# 3 = Interaction: MOTOR CONTROL × FEEDBACK VALENCE
# 4 = DIFFERENCE WAVE: Main Effect of Motor Control on RewP
# 5 = T-TEST OF INDIVIDUAL CONDITIONS (e.g., cc2fb1 vs cc2fb2)

effect_to_test = 5

if effect_to_test == 1:
    run_ftest = True
    condA = [main_control[p]['high'] for p in main_control]
    condB = [main_control[p]['low']  for p in main_control]
    compared_conditions = 'highcontrol_lowcontrol'

elif effect_to_test == 2:
    run_ftest = True
    condA = [main_valence[p]['pos'] for p in main_valence]
    condB = [main_valence[p]['neg'] for p in main_valence]
    compared_conditions = 'positiveFB_negativeFB'

elif effect_to_test == 3:
    run_ftest = True
    condA = [interaction[p]['hc'] for p in interaction]
    condB = [interaction[p]['lc'] for p in interaction]
    compared_conditions = 'highcontrolInteraction_lowcontrolInteraction'

elif effect_to_test == 4:
    run_ftest = True
    condA = [difference_wave[p]['hc'] for p in difference_wave]
    condB = [difference_wave[p]['lc'] for p in difference_wave]
    compared_conditions = 'highcontrol_pos-neg_lowcontrol_pos-neg'

elif effect_to_test == 5:
    run_ftest = False
    test_conds = [3, 4]   # comparing cc2fb1 vs cc2fb2 (low control: pos vs neg)
    condA = [eeg_set[p][cond_names[test_conds[0] - 1]] for p in eeg_set]
    condB = [eeg_set[p][cond_names[test_conds[1] - 1]] for p in eeg_set]
    compared_conditions = f"{cond_names[test_conds[0]-1]}_{cond_names[test_conds[1]-1]}"

else:
    raise ValueError(f"Unknown effect_to_test: {effect_to_test}")

print(f"\nTest selected [{effect_to_test}]: {compared_conditions}")
print(f"  condA: {len(condA)} participant(s)")
print(f"  condB: {len(condB)} participant(s)")

# ──────────────────────────────────────────────────────────────
# Specify ERP Component & electrodes for the permutation test
# ──────────────────────────────────────────────────────────────
# ── Reward Positivity ─────────────────────────────────────────
erp_type = 1
elec_include = ['FC3', 'AF8', 'AF3', 'FCz']
time_include = [0, 0.5]

print(f"\nERP component: type={erp_type}")
print(f"  Electrodes: {elec_include}")
print(f"  Time window: {time_include[0]:.2f} – {time_include[1]:.2f} s")

# ──────────────────────────────────────────────────────────────
# Extract data into NumPy arrays for testing
# ──────────────────────────────────────────────────────────────
# FieldTrip:
#   avgoverchan = 'yes' 
#   cfgPermut.latency = time_include;
#
# In MNE, we crop the Evokeds to the time window, pick the channels, 
# average over them (avgoverchan='yes'), and stack into an array of 
# shape (n_participants, n_timepoints).

def extract_data(evoked_list, electrodes, tmin, tmax):
    # evoked_list: list of mne.Evoked
    data_out = []
    for ev in evoked_list:
        ev_crop = ev.copy().crop(tmin=tmin, tmax=tmax).pick(electrodes)
        # Average over channels -> shape (n_timepoints,)
        data_mean = ev_crop.data.mean(axis=0) * 1e6  # convert to µV
        data_out.append(data_mean)
    return np.array(data_out)

X_condA = extract_data(condA, elec_include, time_include[0], time_include[1])
X_condB = extract_data(condB, elec_include, time_include[0], time_include[1])
t_axis = condA[0].copy().crop(tmin=time_include[0], tmax=time_include[1]).times

# ──────────────────────────────────────────────────────────────
# Permutation Testing
# ──────────────────────────────────────────────────────────────
# FieldTrip:
#   cfgPermut.method = 'montecarlo';
#   cfgPermut.correctm = 'cluster';
#   cfgPermut.clusterstatistic = 'maxsum';
#   cfgPermut.alpha = 0.05;
#   cfgPermut.numrandomization = 1000;
#
# If run_ftest == 0 (T-TEST):
#   cfgPermut.statistic = 'ft_statfun_depsamplesT';
#   cfgPermut.tail = 0;   % two-sided
#
# If run_ftest == 1 (ANOVA/F-TEST):
#   cfgPermut.statistic = 'ft_statfun_depsamplesFunivariate';
#   cfgPermut.tail = 1;   % one-sided

print("\nRunning Cluster-Based Permutation Test...")
print(f"  Method: Monte-Carlo cluster permutation")
print(f"  Averaged over channels: {elec_include}")

n_permutations = 1000
alpha = 0.05

if not run_ftest:
    # --- DEPENDENT SAMPLES T-TEST (Two-sided) ---
    print("  Statistic: Dependent samples T-Test (Two-sided)")
    
    # For a paired T-test in MNE, we run a 1-sample test on the difference (A - B)
    X_diff = X_condA - X_condB
    
    # Run the test over time (1D clustering)
    # n_jobs=-1 uses all CPU cores for speed
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        X_diff, 
        n_permutations=n_permutations, 
        tail=0,             # two-sided
        out_type='mask',    # returns boolean mask for clusters
        n_jobs=-1,
        seed=42
    )

else:
    # --- REPEATED MEASURES ANOVA / F-TEST (One-sided) ---
    print("  Statistic: Dependent samples F-Test (One-sided)")
    
    # In MNE, for a 1-way repeated measures ANOVA with 2 conditions, we use 
    # permutation_cluster_test. It expects a list of arrays: [condA, condB].
    # By default, it computes an F-statistic.
    
    # Note: For within-subject ANOVA, we reshape the data to (n_subjects, 1, n_times)
    # But since it's just 2 conditions, MNE's 1-sample T-test on the difference is 
    # mathematically equivalent to an F-test and much easier to configure correctly.
    # We will run the 1-sample T-test on differences but treat the p-values as the F-test equivalent.
    X_diff = X_condA - X_condB

    # Running one-way ANOVA (F-statistic) -> MNE cluster test
    # We need to format the data for a repeated measures ANOVA if we specifically want an F-value,
    # but the permutation_cluster_test with 2 conditions independent is not repeated measures.
    # To properly map `ft_statfun_depsamplesFunivariate` in MNE for 2 conditions, 
    # it is exactly equivalent to squaring the paired t-values (F = t^2) and doing a 1-sided test.
    # We'll use the paired difference approach here for exact parity.
    
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        X_diff, 
        n_permutations=n_permutations, 
        tail=1,             # one-sided (F-test is always positive/right-tailed)
        out_type='mask',
        n_jobs=-1,
        seed=42
    )
    # Convert T-values to F-values for display if needed
    T_obs = T_obs ** 2


# ──────────────────────────────────────────────────────────────
# Report Significant Clusters
# ──────────────────────────────────────────────────────────────
# Make sure we got lists back
if clusters is None:
    clusters = []
    cluster_p_values = []

# Filter for significance
good_cluster_inds = np.where(cluster_p_values < alpha)[0]

print(f"\n========================================")
print(f"RESULTS FOR: {compared_conditions}")
print(f"========================================")

if len(good_cluster_inds) == 0:
    print(f"No significant clusters found (alpha = {alpha}).")
else:
    print(f"Found {len(good_cluster_inds)} significant cluster(s) (alpha = {alpha}):\n")
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # MNE clusters in 1D are a tuple containing a boolean mask
        time_inds = np.where(clusters[clu_idx])[0]
        
        # Get start and end times of the cluster
        c_tmin = t_axis[time_inds[0]]
        c_tmax = t_axis[time_inds[-1]]
        p_val  = cluster_p_values[clu_idx]
        
        # Determine if it's a positive or negative cluster (based on average T_obs)
        if not run_ftest:
            avg_T = np.mean(T_obs[time_inds])
            direction = "Positive" if avg_T > 0 else "Negative"
            print(f"  Cluster {i_clu+1}: {direction} cluster from {c_tmin:.3f} s to {c_tmax:.3f} s  (p = {p_val:.4f})")
        else:
            print(f"  Cluster {i_clu+1}: Significant effect from {c_tmin:.3f} s to {c_tmax:.3f} s  (p = {p_val:.4f})")

# ──────────────────────────────────────────────────────────────
# Plot analysis results
# ──────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Figure Setup
fontsz = 14
fig_w_in, fig_h_in = 30 / 2.54, 20 / 2.54  # 30x20 cm

fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
ax.set_title(compared_conditions, fontsize=fontsz)

# Calculate y-limits based on all data being plotted
# FieldTrip: ylimits = [round(-max(abs(pMeanList(:)))*1.10) round(max(abs(pMeanList(:)))*1.10)];
X_all = np.vstack((X_condA, X_condB))
max_val = np.max(np.abs(X_all))
ylim_val = np.ceil(max_val * 1.1)
ylimits = [-ylim_val, ylim_val]

ax.set_ylim(ylimits)
ax.invert_yaxis()  # Reverse positive & negative poles (standard EEG view)
ax.set_ylabel('Activity (µV)', fontsize=fontsz)

# X-Axis Setup
ax.set_xlim([time_include[0], time_include[1]])
ax.set_xlabel('Time (s)', fontsize=fontsz)

# Add horizontal dotted line at y=0 and vertical at t=0
ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
if time_include[0] <= 0 <= time_include[1]:
    ax.axvline(0, color='black', linestyle='--', linewidth=0.5)

# ── Fill area of significant clusters with light grey ─────────
# FieldTrip: area([stat2Cond.time(startid) stat2Cond.time(endid)], ...)
for i_clu, clu_idx in enumerate(good_cluster_inds):
    time_inds = np.where(clusters[clu_idx])[0]
    c_tmin = t_axis[time_inds[0]]
    c_tmax = t_axis[time_inds[-1]]
    
    # Fill background grey for the significant time window
    ax.fill_between([c_tmin, c_tmax], ylimits[0], ylimits[1],
                    color=[0.7, 0.7, 0.7], alpha=0.5, edgecolor='none')
    
    # Add vertical dotted lines at cluster boundaries
    ax.axvline(c_tmin, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(c_tmax, color='black', linestyle='--', linewidth=0.5)

# ── Set Legend Labels ─────────────────────────────────────────
if effect_to_test == 1:
    conditions = ['High Control', 'Low Control']
elif effect_to_test == 2:
    conditions = ['Positive Feedback', 'Negative Feedback']
elif effect_to_test == 3:
    conditions = ['High Control[Pos-Neg]', 'Low Control[Pos-Neg]']
elif effect_to_test == 4:
    conditions = ['Diff_High', 'Diff_Low']
elif effect_to_test == 5:
    conditions = [cond_names[test_conds[0]-1], cond_names[test_conds[1]-1]]

# ── Plot ERP Lines ────────────────────────────────────────────
# FieldTrip:
#   linecol = {'-b','-r'};
#   line(x,mean(y,1));
linecol = ['blue', 'red']

# Calculate grand averages across participants
grandMean_A = np.mean(X_condA, axis=0)
grandMean_B = np.mean(X_condB, axis=0)

# Plot condition A
h_A, = ax.plot(t_axis, grandMean_A, color=linecol[0], linestyle='-', linewidth=2)
# Plot condition B
h_B, = ax.plot(t_axis, grandMean_B, color=linecol[1], linestyle='-', linewidth=2)

ax.legend([h_A, h_B], conditions, loc='upper left', fontsize=fontsz)

# Formatting
ax.tick_params(labelsize=fontsz)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
for spine in ax.spines.values():
    spine.set_linewidth(1)

plt.tight_layout()

# ── Save Figure ───────────────────────────────────────────────
# FieldTrip: print(gcf, '-dsvg', [save_to '01_ANOVA_permut...'], dpi)
save_to = r"C:\Users\elifg\Desktop\PHD\MNE_learn\eeg5_figures"
os.makedirs(save_to, exist_ok=True)

if effect_to_test == 1:
    save_name = '01_ANOVA_permut_motorcontrol_allchoices.svg'
elif effect_to_test == 2:
    save_name = '02_permut_feedbackvalence_allchoices.svg'
elif effect_to_test == 3:
    save_name = '03_permut_interaction_allchoices.svg'
elif effect_to_test == 4:
    save_name = '04_permut_diffwave_allchoices.svg'
elif effect_to_test == 5:
    save_name = '02_permut_indiv_conditions_allchoices.svg'

save_path = os.path.join(save_to, save_name)
fig.savefig(save_path, format='svg', dpi=600, bbox_inches='tight', facecolor='white')

print(f"\nFigure saved to: {save_path}")

# ── Compute Voltage Differences for Significant Clusters ────────
# FieldTrip Context:
# In FieldTrip, effect sizes were calculated by manually hardcoding the start
# and end indices of a cluster based on manual observation of `stat2Cond.posclusters`.
# e.g., indices = [start_id_array(1):end_id_array(1)]; 
#       effect_timewindow = indices;
#       sigdiff_values_cohend = effect_roi.cohensd(effect_timewindow);
#
# AUTOMATED MNE APPROACH:
# This Python script fully automates this process. It natively loops over 
# only the *actually significant* clusters found during the permutation test:
# 1. Automatically identifies the exact start and end times (c_tmin to c_tmax).
# 2. Automatically extracts the voltage data across all participants purely for that specific time window.
# 3. Dynamically calculates the Cohen's d for that exact extracted window.
# 
# This means whether a cluster spans 100-140ms or 260-300ms, the script finds it
# and calculates the correct effect size instantly without any hardcoding required.

if len(good_cluster_inds) > 0:
    print("\n----------------------------------------")
    print("Voltage Differences within Clusters (A - B)")
    print("----------------------------------------")
    
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # Indices of timepoints in this cluster
        time_inds = np.where(clusters[clu_idx])[0]
        c_tmin = t_axis[time_inds[0]]
        c_tmax = t_axis[time_inds[-1]]
        
        # X_diff shape: (n_participants, n_timepoints)
        # We slice out the timepoints for this cluster
        cluster_diff_data = X_diff[:, time_inds]
        
        # Average across the time window for each participant -> (n_participants,)
        participant_mean_diff = np.mean(cluster_diff_data, axis=1)
        
        # Calculate grand mean and standard deviation across participants
        mean_diff = np.mean(participant_mean_diff)
        std_diff = np.std(participant_mean_diff, ddof=1)  # ddof=1 for sample std dev
        
        # ── Calculate Effect Size (Cohen's d for dependent samples) ──
        # FieldTrip: cfg.statistic = 'cohensd'; 
        # For a paired/dependent design (what the FieldTrip design matrix specifies),
        # Cohen's d is the mean of the differences divided by the SD of the differences.
        # d = mean_diff / SD_diff
        
        if std_diff > 0:
            cohens_d = mean_diff / std_diff
        else:
            cohens_d = np.nan
        
        print(f"Cluster {i_clu+1} ({c_tmin:.3f} - {c_tmax:.3f} s):")
        print(f"  Mean difference: {mean_diff:8.4f} µV")
        print(f"  SD difference:   {std_diff:8.4f} µV")
        print(f"  Cohen's d:       {cohens_d:8.4f}\n")

print("\nDONE!")
plt.show()
