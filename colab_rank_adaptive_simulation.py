import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set high-quality style for academic publication
sns.set_theme(style='whitegrid', palette='colorblind')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9.5
plt.rcParams['axes.titlesize'] = 10.5
plt.rcParams['axes.labelsize'] = 9.5
plt.rcParams['xtick.labelsize'] = 8.5
plt.rcParams['ytick.labelsize'] = 8.5
plt.rcParams['figure.dpi'] = 300

print("=" * 74)
print("Running Simulation & Curve Generation for Low-Rank Attention Bounds")
print("=" * 74)

# ==============================================================================
# Figure 1: Effective Dimension & Capacity Reduction Percentage
# ==============================================================================
d_in = 4096
d_out = 4096
full_dim = d_in * d_out

ranks = np.arange(1, 129)
p_eff = ranks * (d_out + d_in - ranks)
capacity_cut = (1.0 - p_eff / full_dim) * 100.0

fig, ax1 = plt.subplots(figsize=(6.2, 3.8))

color1 = '#1f77b4'
ax1.set_xlabel('Attention Weight Rank ($s$)', fontweight='bold', labelpad=5)
ax1.set_ylabel(r'Effective Dimension $p_{\mathrm{eff}}(s)$ (Millions)', color=color1, fontweight='bold', labelpad=6)
line1 = ax1.plot(ranks, p_eff / 1e6, color=color1, linewidth=2.2, label=r'Effective Dim $p_{\mathrm{eff}}(s)$')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 1.1)

ax2 = ax1.twinx()
color2 = '#d62728'
ax2.set_ylabel('Parameter Capacity Reduction (%)', color=color2, fontweight='bold', labelpad=6)
line2 = ax2.plot(ranks, capacity_cut, color=color2, linewidth=2.0, linestyle='--', label='Capacity Reduction (%)')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(98.0, 100.05)

# Highlight Target Rank s=16
s_target = 16
p_eff_16 = s_target * (d_out + d_in - s_target)
pct_16 = (1.0 - p_eff_16 / full_dim) * 100.0

ax1.plot(s_target, p_eff_16 / 1e6, 'o', color='#ff7f0e', markersize=7, zorder=5)
ax1.axvline(x=s_target, color='#ff7f0e', linestyle=':', alpha=0.7, linewidth=1.2)

annot_text = rf"Rank $s={s_target}$" + "\n" + rf"$p_{{\mathrm{{eff}}}} = {p_eff_16:,}$" + "\n" + f"({pct_16:.2f}% Cut)"
ax1.annotate(annot_text,
             xy=(s_target, p_eff_16 / 1e6),
             xytext=(32, 0.42),
             arrowprops=dict(facecolor='#ff7f0e', edgecolor='#ff7f0e', shrink=0.08, width=1.0, headwidth=5),
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#fff2cc', edgecolor='#ff7f0e', alpha=0.9),
             fontsize=8.0, fontweight='bold')

plt.title(rf'Effective Dimension $p_{{\mathrm{{eff}}}}(s)$ vs. Rank $s$ ($d_{{\mathrm{{in}}}}=d_{{\mathrm{{out}}}}={d_in}$)',
          fontsize=10.0, fontweight='bold', pad=8)

plt.tight_layout()
plt.savefig('fig1_rank_dimension_reduction.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SUCCESS] Figure 1 generated and saved as 'fig1_rank_dimension_reduction.png'")

# ==============================================================================
# Figure 2: Local Rademacher Complexity Contraction Curves
# ==============================================================================
r_vals = np.linspace(0.001, 1.0, 300)
sample_size = 10000
ranks_to_compare = [1, 4, 16, 64, 4096]

fig, ax = plt.subplots(figsize=(6.2, 3.8))
colors = sns.color_palette('colorblind', len(ranks_to_compare))

for idx, s in enumerate(ranks_to_compare):
    p_eff_val = s * (d_out + d_in - s)
    rad_complexity = np.sqrt((s * (d_out + d_in) * r_vals) / sample_size)
    
    if s == 4096:
        label_str = f'Full Rank $s=4096$ ($d^2$)'
        linestyle = '--'
        linewidth = 1.6
    else:
        label_str = rf'Rank $s={s}$ ($p_{{\mathrm{{eff}}}}={p_eff_val:,}$)'
        linestyle = '-'
        linewidth = 1.8
        
    ax.plot(r_vals, rad_complexity, label=label_str, color=colors[idx], linestyle=linestyle, linewidth=linewidth)

ax.set_xlabel('Excess Risk Level ($r$)', fontweight='bold', labelpad=5)
ax.set_ylabel(r'Local Rademacher Bound $\mathfrak{R}_n(\mathcal{F}(r,s))$', fontweight='bold', labelpad=8)
ax.set_title(rf'Local Rademacher Complexity Shrinkage ($n={sample_size:,}$, $d_{{\mathrm{{in}}}}=d_{{\mathrm{{out}}}}={d_in}$)',
             fontsize=10.0, fontweight='bold', pad=8)

ax.legend(title='Rank Constraint ($s$)', frameon=True, facecolor='white', framealpha=0.92,
          fontsize=8.0, title_fontsize=8.5, loc='upper left', bbox_to_anchor=(0.02, 0.98))

sns.despine(ax=ax)
plt.tight_layout()
plt.savefig('fig2_rademacher_shrinkage.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SUCCESS] Figure 2 generated and saved as 'fig2_rademacher_shrinkage.png'")

print("=" * 74)
print("All plots generated successfully. Ready for evaluation.")
print("=" * 74)
