import os
import sys
import numpy as np
import pandas as pd
import mne

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

# Path for saving figures
# FieldTrip: save_to = 'D:/MCRL DATA/eeg5_figures'; dpi = '-r600';
save_to = r"C:\Users\elifg\Desktop\PHD\MNE_learn\eeg5_figures"
os.makedirs(save_to, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Electrode and time window selection for Reward Positivity
# ──────────────────────────────────────────────────────────────

electrodes_select = ['Fz', 'FCz', 'FC1', 'FC2']

# Time window for Reward Positivity analysis and topoplots
# FieldTrip: time_select = [0.20 0.30];  % literature-based time window
# (alternative data-driven window for topoplots, also in FieldTrip):
# time_select = [0.26, 0.30]   # test = 1, data-driven for topoplots
time_select = [0.20, 0.30]     # test = 1, literature-based

# For difference wave analysis (comment time_select above and use this):
# FieldTrip: %test = 1.1 / time_select = [0.28 0.32]
# time_select = [0.28, 0.32]

# ──────────────────────────────────────────────────────────────
# Condition layout (matches ERP_calculation.py output)
# ──────────────────────────────────────────────────────────────
# FieldTrip: num_cond = 4
# cond index → label:
#   0 → cc1fb1 = high control + positive feedback
#   1 → cc1fb2 = high control + negative feedback
#   2 → cc2fb1 = low  control + positive feedback
#   3 → cc2fb2 = low  control + negative feedback
num_cond = 4
cond_names = ['cc1fb1', 'cc1fb2', 'cc2fb1', 'cc2fb2']

# ──────────────────────────────────────────────────────────────
# Load data for all participants + build grand-average structures
# ──────────────────────────────────────────────────────────────
# FieldTrip:
#   alleeg{p,cond} = participant.eegdat{cond};
#   behavSummary   = [behavSummary participant.summary];
#
# In MNE, mne.read_evokeds() loads the FIF saved by ERP_calculation.py.
# Each file contains all 4 Evoked objects (one per condition), identified
# by their .comment attribute (set to the condition label in ERP_calculation.py).

# alleeg[p][cond_label] → Evoked for participant p, condition cond_label
alleeg       = {}   # FieldTrip: alleeg{p, cond}
all_summaries = []   # FieldTrip: behavSummary

for p_idx, pnum in enumerate(plist):
    sub_id = f"{pnum:04d}"
    erp_file = os.path.join(dfolder, f"MCRL_{sub_id}-erp-ave.fif")
    csv_file  = os.path.join(dfolder, f"MCRL_{sub_id}-erp-summary.csv")

    if not os.path.exists(erp_file):
        print(f"  ERP file not found, skipping participant {pnum}: {erp_file}")
        continue

    print(f"Loading participant {pnum}")

    # Load all Evoked objects for this participant
    # FieldTrip: participant = load([dfolder '/MCRL_' num2str(pnum)]);
    evoked_list = mne.read_evokeds(erp_file, verbose=False)

    # Index by condition label (stored in evoked.comment during ERP_calculation.py)
    alleeg[p_idx] = {ev.comment: ev for ev in evoked_list}

    # Load trial-count summary
    # FieldTrip: behavSummary = [behavSummary participant.summary];
    if os.path.exists(csv_file):
        summary_row = pd.read_csv(csv_file).iloc[0].to_dict()
        all_summaries.append(summary_row)

    # ── Difference waves ──────────────────────────────────────
    # FieldTrip:
    #   cfg.operation = 'x1-x2';  % positive minus negative
    #   cfg.parameter = 'avg';
    #   difference_wave{p,1} = ft_math(cfg, alleeg{p,1}, alleeg{p,2}); %high control
    #   difference_wave{p,2} = ft_math(cfg, alleeg{p,3}, alleeg{p,4}); %low control
    #
    # MNE equivalent: mne.combine_evoked() with weights [+1, -1]
    # This subtracts negative-feedback ERP from positive-feedback ERP,
    # isolating / enhancing the Reward Positivity (RewP).

    diff_hc = mne.combine_evoked(          # high control: pos - neg
        [alleeg[p_idx]['cc1fb1'], alleeg[p_idx]['cc1fb2']],
        weights=[1, -1]
    )
    diff_hc.comment = 'cc1_diff'

    diff_lc = mne.combine_evoked(          # low control: pos - neg
        [alleeg[p_idx]['cc2fb1'], alleeg[p_idx]['cc2fb2']],
        weights=[1, -1]
    )
    diff_lc.comment = 'cc2_diff'

    # Store difference waves alongside alleeg for grand averaging below
    alleeg[p_idx]['cc1_diff'] = diff_hc
    alleeg[p_idx]['cc2_diff'] = diff_lc

# Print behavioral summary table across participants
# FieldTrip: struct2table(behavSummary)
if all_summaries:
    print("\nBehavioral summary (trial counts per condition):")
    print(pd.DataFrame(all_summaries).to_string(index=False))

# ──────────────────────────────────────────────────────────────
# Grand averages across participants
# ──────────────────────────────────────────────────────────────

# ── Non-difference wave grand averages ───────────────────────
# FieldTrip:
#   cfg = [];
#   for cond=1:num_cond
#       GA_dat{cond} = ft_timelockgrandaverage(cfg, alleeg{:,cond});
#   end
#
# MNE: mne.grand_average() averages across a list of Evoked objects.
# It normalizes by number of trials under the hood (equivalent to FieldTrip's
# ft_timelockgrandaverage with default cfg).

print("\nComputing grand averages...")
GA_dat = {}   # FieldTrip: GA_dat{cond}

for cond_label in cond_names:
    # Collect this condition's Evoked across all loaded participants
    evokeds_this_cond = [alleeg[p][cond_label]
                         for p in alleeg if cond_label in alleeg[p]]
    if evokeds_this_cond:
        GA_dat[cond_label] = mne.grand_average(evokeds_this_cond)
        GA_dat[cond_label].comment = cond_label
        print(f"  GA [{cond_label}]: averaged across {len(evokeds_this_cond)} participant(s)")

# ── Difference wave grand averages ───────────────────────────
# FieldTrip:
#   num_cond_diff = size(difference_wave, 2);
#   cfg = [];
#   for cond=1:num_cond_diff
#       GA_dat_diff{cond} = ft_timelockgrandaverage(cfg, difference_wave{:,cond});
#   end
#   cond_names_diff = {'cc1', 'cc2'};

cond_names_diff = ['cc1_diff', 'cc2_diff']   # FieldTrip: cond_names_diff = {'cc1', 'cc2'}
GA_dat_diff = {}                               # FieldTrip: GA_dat_diff{cond}

for diff_label in cond_names_diff:
    evokeds_diff = [alleeg[p][diff_label]
                    for p in alleeg if diff_label in alleeg[p]]
    if evokeds_diff:
        GA_dat_diff[diff_label] = mne.grand_average(evokeds_diff)
        GA_dat_diff[diff_label].comment = diff_label
        print(f"  GA_diff [{diff_label}]: averaged across {len(evokeds_diff)} participant(s)")

print("\nGrand averages ready.")
print(f"  Electrodes selected : {electrodes_select}")
print(f"  Time window         : {time_select[0]:.2f} - {time_select[1]:.2f} s")

# ──────────────────────────────────────────────────────────────
# Average activity across electrodes per participant / condition
# ──────────────────────────────────────────────────────────────
# FieldTrip:
#   for p=1:length(plist)
#     for cond=1:num_cond
#       avg_subj = ft_timelockgrandaverage(cfg, alleeg{p,cond});  % select electrodes via cfg.channel
#       pMean    = mean(avg_subj.avg, 1);                         % average over electrode dimension
#       pMeanList(p,cond,:) = squeeze(pMean)';                    % [n_participants, n_conds, n_timepoints]
#     end
#   end
#
# In MNE, evoked.data has shape [n_channels, n_timepoints] in Volts.
# We pick() the selected electrodes, then average across channels to get
# one time series per participant × condition.  Multiply by 1e6 → µV.

import matplotlib
matplotlib.use('TkAgg')   # interactive window; change to 'Agg' for headless/batch saving only
import matplotlib.pyplot as plt

loaded_plist = list(alleeg.keys())           # participant indices that were successfully loaded
n_participants = len(loaded_plist)

# Get shared time axis (same for all participants / conditions)
# FieldTrip: x = avg_subj.time;
times = alleeg[loaded_plist[0]][cond_names[0]].times   # in seconds

# Verify that all requested electrodes actually exist in the data
available_ch = alleeg[loaded_plist[0]][cond_names[0]].ch_names
picked_channels = [ch for ch in electrodes_select if ch in available_ch]
missing = [ch for ch in electrodes_select if ch not in available_ch]
if missing:
    print(f"  WARNING: electrode(s) not found in data and will be skipped: {missing}")
print(f"  Using electrodes: {picked_channels}")

n_timepoints = len(times)
num_cond_diff = len(cond_names_diff)

# ── Non-difference wave pMeanList ──────────────────────────────
pMeanList = np.zeros((n_participants, num_cond, n_timepoints))   # [P, C, T]

for p_idx, p_key in enumerate(loaded_plist):
    for cond_idx, cond_label in enumerate(cond_names):
        # Pick selected electrodes and convert to µV
        evoked = alleeg[p_key][cond_label].copy().pick(picked_channels)
        data_uv = evoked.data * 1e6                        # [n_channels, n_timepoints]
        pMeanList[p_idx, cond_idx, :] = np.mean(data_uv, axis=0)   # average over electrodes

# ── Compute grandMean / subjMean for error-bar correction ──────
# FieldTrip:
#   if length(plist) == 1
#     grandMean = squeeze(mean(pMeanList, 2));         % mean over conditions dimension
#   else
#     subjMean = squeeze(mean(pMeanList, 2));           % [P, T] -- mean over conditions
#     grandMean = mean(subjMean, 1);                   % [T]    -- mean over participants
#   end
#
# grandMean is used only in the Cousineau-Morey within-subject error correction
# (see plotting loops below). For single-participant plots there are no error bars.

if n_participants == 1:
    grandMean = np.squeeze(np.mean(pMeanList, axis=1))   # [T]
    subjMean  = grandMean                                 # unused for 1 participant
else:
    subjMean  = np.squeeze(np.mean(pMeanList, axis=1))   # [P, T] -- mean over conditions
    grandMean = np.mean(subjMean, axis=0)                 # [T]    -- mean over participants

# ── Difference wave pMeanList ─────────────────────────────────
pMeanList_diff = np.zeros((n_participants, num_cond_diff, n_timepoints))

for p_idx, p_key in enumerate(loaded_plist):
    for cond_idx, diff_label in enumerate(cond_names_diff):
        evoked_diff = alleeg[p_key][diff_label].copy().pick(picked_channels)
        data_uv = evoked_diff.data * 1e6
        pMeanList_diff[p_idx, cond_idx, :] = np.mean(data_uv, axis=0)

if n_participants == 1:
    grandMean_diff = np.squeeze(np.mean(pMeanList_diff, axis=1))
    subjMean_diff  = grandMean_diff
else:
    subjMean_diff  = np.squeeze(np.mean(pMeanList_diff, axis=1))   # [P, T]
    grandMean_diff = np.mean(subjMean_diff, axis=0)                 # [T]


# ──────────────────────────────────────────────────────────────
# Plot settings
# ──────────────────────────────────────────────────────────────
# FieldTrip: plotType = 2; ylimits = [-3 3];
# 1 = non-difference wave (4 conditions)
# 2 = difference wave (high vs. low control, pos-neg)
plotType = 2

ylimits  = [-3, 3]   # µV y-axis range

# Line colors per condition [purple, blue, red, green]
# FieldTrip: colors = {[0.8,0.47,0.65], [0,0.44,0.69], [1 0 0], [0 1 0]}
colors      = [[0.8, 0.47, 0.65], [0, 0.44, 0.69], [1, 0, 0], [0, 1, 0]]
linestyles  = ['-', '-', '-', '-']   # FieldTrip: linestyle = {'-','-','-','-'}

# Linecol for single-participant simple plots
# FieldTrip: linecol = {'-b','-r','-g','-m'}
linecol_colors = ['blue', 'red', 'green', 'magenta']

# Difference wave colors [blue, purple] for high and low control
# FieldTrip: diffcolours = {'-a','-c'} → commented out: {[1 1 0],[0 0 1]}
diff_colors     = [[0, 0.44, 0.69], [0.8, 0.47, 0.65]]   # blue (HC), purple (LC)
linestyles_diff = ['--', '--']       # FieldTrip: linestyle_diff = {'--','--'}

# Figure dimensions matching FieldTrip's PaperPosition [0 0 30 20] (cm)
fig_w_cm, fig_h_cm = 30, 20
fig_w_in = fig_w_cm / 2.54
fig_h_in = fig_h_cm / 2.54
font_size = 20   # FieldTrip: fontsz = 20

# ──────────────────────────────────────────────────────────────
# PRELIMINARY FIGURE — always shows difference waves
# (equivalent to the first un-labelled figure; hold on; block in FieldTrip)
# ──────────────────────────────────────────────────────────────
plt.close('all')   # FieldTrip: close all

fig0, ax0 = plt.subplots(figsize=(fig_w_in, fig_h_in))

H0 = []
for cond_idx, diff_label in enumerate(cond_names_diff):
    y_diff = pMeanList_diff[:, cond_idx, :]   # [P, T]

    if n_participants == 1:
        # Single participant: plot the raw time series directly
        # FieldTrip: H(cond) = plot(x, y_diff, 'Color', diffcolours{cond});
        h, = ax0.plot(times, y_diff[0, :],
                      linestyle=linestyles_diff[cond_idx],
                      color=diff_colors[cond_idx])
    else:
        # Multiple participants: Cousineau-Morey within-subject correction
        # FieldTrip:
        #   yCorrect_diff = y_diff - subjMean_diff;
        #   yCorrect_diff = (bsxfun(@plus, yCorrect_diff, grandMean_diff)) * (4/3);
        #   errbar = std(yCorrect_diff, 0, 1) / sqrt(size(yCorrect_diff, 1));
        #
        # The *(4/3) factor is the Cousineau-Morey correction for 4 conditions
        # (num_cond / (num_cond - 1) = 4/3).
        yCorrect_diff = y_diff - subjMean_diff
        yCorrect_diff = (yCorrect_diff + grandMean_diff) * (4 / 3)
        errbar = np.std(yCorrect_diff, axis=0, ddof=0) / np.sqrt(n_participants)

        # Simple line plot (error bars commented out below)
        # FieldTrip: H(cond) = plot(x, mean(yCorrect_diff,1), linecol{cond});
        h, = ax0.plot(times, np.mean(yCorrect_diff, axis=0),
                      linestyle=linestyles_diff[cond_idx],
                      color=diff_colors[cond_idx])

        # Shaded error bar (commented out — uncomment to show):
        # FieldTrip: H(cond) = shadedErrorBar(x, mean(y_diff,1), errbar, 'lineProps', ...)
        # ax0.fill_between(times,
        #                  np.mean(yCorrect_diff, axis=0) - errbar,
        #                  np.mean(yCorrect_diff, axis=0) + errbar,
        #                  alpha=0.2, color=diff_colors[cond_idx])

    H0.append(h)

# FieldTrip: set(gca,'YDir','reverse'); ylim(ylimits)
ax0.set_ylim(ylimits)
ax0.invert_yaxis()           # EEG convention: negative up
ax0.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax0.axvline(0, color='black', linestyle='--', linewidth=0.5)
plt.tight_layout()

# ──────────────────────────────────────────────────────────────
# MAIN FIGURE — plotType-controlled, with labels and saving
# (equivalent to the second figure; hold on; block in FieldTrip)
# ──────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(fig_w_in, fig_h_in))

H1 = []

if plotType == 1:
    # ── Non-difference wave: 4 conditions ───────────────────
    labels = ['High Control, Positive FB', 'High Control, Negative FB',
              'Low Control, Positive FB',  'Low Control, Negative FB']
    # FieldTrip: for cond=1:num_cond
    for cond_idx, cond_label in enumerate(cond_names):
        y = pMeanList[:, cond_idx, :]   # [P, T]

        if n_participants == 1:
            # FieldTrip: H(cond) = plot(x, y, linecol{cond});
            h, = ax1.plot(times, y[0, :],
                          linestyle='-', color=linecol_colors[cond_idx])
        else:
            # Cousineau-Morey correction
            # FieldTrip: yCorrect = y - subjMean; yCorrect = (..) * (4/3);
            yCorrect = y - subjMean
            yCorrect = (yCorrect + grandMean) * (4 / 3)
            errbar   = np.std(yCorrect, axis=0, ddof=0) / np.sqrt(n_participants)

            # FieldTrip: H(cond) = plot(x, mean(y,1), linecol{cond});
            h, = ax1.plot(times, np.mean(y, axis=0),
                          linestyle=linestyles[cond_idx],
                          color=colors[cond_idx])

            # Shaded error bars (uncomment to show):
            # FieldTrip: H(cond) = shadedErrorBar(x, mean(y,1), errbar, 'lineProps', ...)
            # ax1.fill_between(times,
            #                  np.mean(y, axis=0) - errbar,
            #                  np.mean(y, axis=0) + errbar,
            #                  alpha=0.2, color=colors[cond_idx])
        H1.append(h)
    save_name = '00_lineplot_allconds_plain.svg'   # FieldTrip: '00_lineplot_allconds_plain'

elif plotType == 2:
    # ── Difference wave: 2 conditions (high vs. low control) ─
    labels = ['High Control [pos-neg]', 'Low Control [pos-neg]']
    # FieldTrip: for cond = 1:num_cond_diff
    for cond_idx, diff_label in enumerate(cond_names_diff):
        y_diff = pMeanList_diff[:, cond_idx, :]   # [P, T]

        if n_participants == 1:
            # FieldTrip: H(cond) = plot(x, y_diff, linecol{cond});
            h, = ax1.plot(times, y_diff[0, :],
                          linestyle=linestyles_diff[cond_idx],
                          color=diff_colors[cond_idx])
        else:
            # Cousineau-Morey correction
            yCorrect_diff = y_diff - subjMean_diff
            yCorrect_diff = (yCorrect_diff + grandMean_diff) * (4 / 3)
            errbar        = np.std(yCorrect_diff, axis=0, ddof=0) / np.sqrt(n_participants)

            # FieldTrip: H(cond) = plot(x, mean(yCorrect_diff,1), linecol{cond});
            h, = ax1.plot(times, np.mean(yCorrect_diff, axis=0),
                          linestyle=linestyles_diff[cond_idx],
                          color=diff_colors[cond_idx])

            # Shaded error bars (uncomment to show):
            # FieldTrip: H(cond) = shadedErrorBar(x, mean(y_diff,1), errbar, 'lineprops', ...)
            # ax1.fill_between(times,
            #                  np.mean(yCorrect_diff, axis=0) - errbar,
            #                  np.mean(yCorrect_diff, axis=0) + errbar,
            #                  alpha=0.2, color=diff_colors[cond_idx])
        H1.append(h)
    save_name = '00_lineplot_diffwave_plain_allchoices.svg'


# ── Formatting ────────────────────────────────────────────────
# FieldTrip:
#   legend(H, labels, 'Location', 'best');
#   set(gca, 'YDir','reverse');   ylim(ylimits)
#   line([x(1) x(end)], [0 0], 'Color','black','LineStyle','--','LineWidth',0.5);
#   line([0 0], ylimits, 'Color','black','LineStyle','--','LineWidth',0.5);
#   xlabel('Time (s)');  xticks(0:0.1:max(x));  ylabel('microVolt')
#   set(gca,'FontSize',20,'FontName','Times New Roman','LineWidth',1)

ax1.legend(H1, labels, loc='best', fontsize=font_size)
ax1.set_ylim(ylimits)
ax1.invert_yaxis()           # EEG convention: negative up
ax1.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax1.axvline(0, color='black', linestyle='--', linewidth=0.5)

ax1.set_xlabel('Time (s)', fontsize=font_size)
ax1.set_ylabel('µV', fontsize=font_size)
ax1.set_xticks(np.arange(0, times[-1] + 0.01, 0.1))   # FieldTrip: xticks(0:0.1:max(x))
ax1.tick_params(labelsize=font_size, width=1)
for spine in ax1.spines.values():
    spine.set_linewidth(1)

# FieldTrip: set(gca,'FontName','Times New Roman')
try:
    plt.rcParams['font.family'] = 'Times New Roman'
except Exception:
    pass   # fall back to matplotlib default if font not installed

# FieldTrip: set(gcf,'Color',[1 1 1])   → white background
fig1.patch.set_facecolor('white')
ax1.set_facecolor('white')

# Title hidden (FieldTrip: set(title_handle,'Visible','off'))
ax1.set_title('')

plt.tight_layout()

# ── Save figure ───────────────────────────────────────────────
# FieldTrip: print(gcf, '-dsvg', [save_to '00_lineplot_...'], dpi)
# dpi=600 matches FieldTrip's -r600; SVG is vector so DPI mainly affects rasterised elements.
save_path = os.path.join(save_to, save_name)
fig1.savefig(save_path, format='svg', dpi=600, bbox_inches='tight',
             facecolor='white')
print(f"\n  Figure saved: {save_path}")

# ──────────────────────────────────────────────────────────────
# Topographical Plots
# ──────────────────────────────────────────────────────────────
# FieldTrip:
#   cfg.highlightchannel = electrodes_select;
#   cfg.layout    = 'layout65.mat';
#   cfg.colorbar  = 'yes';
#   cfg.xlim      = time_select;
#   cfg.colormap  = flipud(brewermap(1000,'RdBu'));
#   cfg.zlim      = [-1.5 1.5];
#   ft_topoplotER(cfg, GA_dat{num_plot});
#
# In MNE, we use the `mask` parameter of plot_topomap() to highlight
# the selected electrodes (equivalent to cfg.highlightchannel).
# MNE handles the 2D projection internally so positions always match.

# Colormap: FieldTrip uses flipud(brewermap(1000,'RdBu')) → reversed RdBu.
# matplotlib's 'RdBu_r' is the reversed version (blue for negative, red for positive).
topo_cmap = plt.cm.RdBu_r

# Color limits matching FieldTrip: cfg.zlim = [-1.5 1.5]   (µV)
# Adjust these if needed for visibility.
topo_vlim = (-1.5, 1.5)   # in µV — will be converted to Volts for MNE

# Figure size: FieldTrip PaperPosition [0 0 8 8] (cm) → 8×8 cm per topo
topo_fig_in = 8 / 2.54   # ~3.15 inches

# Time window for averaging: FieldTrip cfg.xlim = time_select
t_min, t_max = time_select

# Build a boolean mask for the highlighted electrodes.
# FieldTrip: cfg.highlightchannel = electrodes_select; cfg.highlightsymbol = '*'
# MNE: mask = boolean array [n_channels], True = highlight with mask_params style.
def make_highlight_mask(evoked, highlight_names):
    """Return a boolean array (n_channels,) — True for channels to highlight."""
    mask = np.array([ch in highlight_names for ch in evoked.ch_names])
    return mask

# Style for highlighted channels (FieldTrip: cfg.highlightsymbol = '*', cfg.highlightsize = 10)
mask_params = dict(marker='*', markerfacecolor='black', markeredgecolor='black',
                   markersize=10, zorder=10)

if plotType == 1:
    # ── Non-difference wave: one topo per condition ──────────
    topo_labels = ['High Control, Positive FB', 'High Control, Negative FB',
                   'Low Control, Positive FB',  'Low Control, Negative FB']
    topo_prefix = '00_topo_allconds_allchoices'

    for cond_idx, cond_label in enumerate(cond_names):
        evoked_topo = GA_dat[cond_label].copy()
        evoked_topo.crop(tmin=t_min, tmax=t_max)
        topo_data = evoked_topo.data.mean(axis=1)   # [n_channels] in Volts

        fig_topo, ax_topo = plt.subplots(figsize=(topo_fig_in, topo_fig_in))

        # FieldTrip: ft_topoplotER(cfg, GA_dat{num_plot})
        highlight_mask = make_highlight_mask(evoked_topo, picked_channels)
        mne.viz.plot_topomap(
            topo_data, evoked_topo.info,
            axes=ax_topo,
            cmap=topo_cmap,
            vlim=(topo_vlim[0] * 1e-6, topo_vlim[1] * 1e-6),   # µV → V
            mask=highlight_mask,
            mask_params=mask_params,
            show=False,
            contours=6,
        )

        ax_topo.set_title(topo_labels[cond_idx], fontsize=12)
        fig_topo.patch.set_facecolor('white')
        fig_topo.tight_layout()

        # Save as SVG and TIFF
        # FieldTrip: print(gcf, '-dsvg', [...], dpi); print(gcf, '-dtiff', [...], dpi)
        for fmt in ['svg', 'tiff']:
            topo_save = os.path.join(save_to, f"{topo_prefix}_{cond_label}.{fmt}")
            fig_topo.savefig(topo_save, format=fmt, dpi=600,
                             bbox_inches='tight', facecolor='white')
        print(f"  Topo saved: {topo_prefix}_{cond_label} (.svg + .tiff)")

elif plotType == 2:
    # ── Difference wave: one topo per control level ──────────
    topo_labels_diff = ['High Control [pos-neg]', 'Low Control [pos-neg]']
    topo_prefix = '00_difftopo_allconds_allchoices'

    for cond_idx, diff_label in enumerate(cond_names_diff):
        evoked_topo = GA_dat_diff[diff_label].copy()
        evoked_topo.crop(tmin=t_min, tmax=t_max)
        topo_data = evoked_topo.data.mean(axis=1)

        fig_topo, ax_topo = plt.subplots(figsize=(topo_fig_in, topo_fig_in))

        highlight_mask = make_highlight_mask(evoked_topo, picked_channels)
        mne.viz.plot_topomap(
            topo_data, evoked_topo.info,
            axes=ax_topo,
            cmap=topo_cmap,
            vlim=(topo_vlim[0] * 1e-6, topo_vlim[1] * 1e-6),
            mask=highlight_mask,
            mask_params=mask_params,
            show=False,
            contours=6,
        )

        ax_topo.set_title(topo_labels_diff[cond_idx], fontsize=12)
        fig_topo.patch.set_facecolor('white')
        fig_topo.tight_layout()

        for fmt in ['svg', 'tiff']:
            topo_save = os.path.join(save_to, f"{topo_prefix}_{diff_label}.{fmt}")
            fig_topo.savefig(topo_save, format=fmt, dpi=600,
                             bbox_inches='tight', facecolor='white')
        print(f"  Topo saved: {topo_prefix}_{diff_label} (.svg + .tiff)")

plt.show()   # display all figures interactively; close windows to end script


