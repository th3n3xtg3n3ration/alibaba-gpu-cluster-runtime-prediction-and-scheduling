# Şekil & Tablo Kalite Denetimi — 139 Doğrulanmış Bulgu

14 ajanlı workflow (7 inceleme + 7 bağımsız doğrulama). Sadece dosyada/kodda **kanıtıyla doğrulanan** bulgular listelendi; her biri ikinci bir ajan tarafından bağımsız kontrol edildi.

**Dağılım:** {'CRITICAL': 31, 'MAJOR': 66, 'MINOR': 42}

| Tür | Adet |
|---|---|
| other | 26 |
| caption-mismatch | 21 |
| color | 16 |
| axis | 15 |
| label-overlap | 13 |
| naming-inconsistency | 12 |
| units | 10 |
| dpi-quality | 9 |
| legend-error | 9 |
| truncation | 8 |

---

## NB01 — 18 bulgu (4 kritik)

### [CRITICAL] NB01-Figure06.png (thesis/latex/figures/nb01-fig06-arrival-heatmap.png) — caption-mismatch
**Sorun:** The heatmap y-axis is 'Day of Week (0 = Monday)' with title 'Job Arrival Heatmap - Day-of-Week x Hour-of-Day', but the weekday encoding is an epoch artifact that folds trace-day 0 and trace-day 7 into one row. The LaTeX caption, the notebook markdown below the cell, and src/feature_engineering.py all say the opposite (trace-day counter). The pipeline was corrected; the figure never was.

**Kanıt:** VERIFIED BY DIRECT COMPUTATION. (1) Code at notebooks/en/01_data_overview.ipynb cell idx 25 (id cd21) literally reads `df_heat["dow"] = df_heat["arrival_time"].dt.dayofweek  # 0 = Monday` and `.reindex(index=range(7), ...)`, with `ax.set_ylabel("Day of Week (0 = Monday)")`. (2) Ran the pipeline on the real data: dayofweek value_counts gives {0:8022, 1:12312, 2:10871, 3:17677, 4:13550, 5:11513, 6:8239}; trace-day counter gives {0:11106, 1:13551, 2:11512, 3:8239, 4:8023, 5:12311, 6:10871, 7:6571}. day0 + day7 = 11106 + 6571 = 17677 = EXACTLY the dayofweek==3 bucket. The fold is arithmetically pr...

**Düzeltme:** In cd21 replace `.dt.dayofweek` with `_t0 = df_heat["arrival_time"].min(); df_heat["dow"] = ((df_heat["arrival_time"] - _t0).dt.total_seconds() // 86400).astype(int)`, change `reindex(index=range(7))` to `range(8)`, set ylabel to 'Trace Day Index (0 = first day of trace)' and retitle. Mask cells (7, 16..23) as NaN with `mask=heatmap_data.isna()` since day 7 only covers hours 0-15 (trace ends 15:51:32). Re-run and re-export.

**Konum:** `notebooks/en/01_data_overview.ipynb cell idx 25 (id cd21); src/feature_engineering.py:205-208; thesis/latex/chapters/3.dataset_and_workload.tex fig:arrival-heatmap caption`


### [CRITICAL] NB01-Figure05.png (thesis/latex/figures/nb01-fig05-interarrival.png) — truncation
**Sorun:** The shipped PNG is stale - it was produced by the old linear-bin code and does not match the notebook that currently generates it. The thesis body and caption publish a number the code has already retracted in a comment.

**Kanıt:** VERIFIED BY HASH AND BY EYE. sha1 of cell cd19's embedded output = c1d686c85b5f25b819bb34f58861a4d90e86e77c; sha1 of both results/figures/thesis_export/png/NB01-Figure05.png and thesis/latex/figures/nb01-fig05-interarrival.png = f8acd14250b87693f2712f55acc9279bd62d09a7. All five other NB01 figures hash-match their notebook output exactly (Fig01 e0d8d223..., Fig02 3a53934f..., Fig03 6e296a2c..., Fig04 a4e39b91..., Fig06 1caefac1...); ONLY Figure 5 diverges. Rendered both images: shipped PNG title is bare 'Inter-Arrival Time Distribution' with one wide first bar at ~24,600 spanning 1-3.7 s and c...

**Düzeltme:** Re-run scripts/export_thesis_results.py so Fig05 regenerates from cd19's current output (verify with `shasum -a1`). Rewrite tex:94 and the tex:99 caption: peak is ~12,100 pairs at exactly 1 s, not ~25,000 over 1-3 s, and disclose that 21.6% of consecutive pairs arrived in the same second and are excluded from the log axis. Extend tests/test_export_thesis_results.py with an assertion that every exported PNG hash equals its notebook output hash so a stale export fails the build.

**Konum:** `notebooks/en/01_data_overview.ipynb cell idx 23 (id cd19); scripts/export_thesis_results.py:152-160; thesis/latex/chapters/3.dataset_and_workload.tex:94,99`


### [CRITICAL] NB01-Figure04.png (thesis/latex/figures/nb01-fig04-gpu-demand.png) — caption-mismatch
**Sorun:** The notebook commentary attached to this figure states numbers that the same cell's own printed describe() disproves three lines earlier, including a ~43,000-job zero-GPU bar that cannot exist. The thesis repeats the wrong mean.

**Kanıt:** VERIFIED AGAINST PRINTED OUTPUT AND RECOMPUTED FROM DATA. Cell cd17's own stream output prints: count 82184, mean 0.680211, min 0.010000, 25% 0.250000, 50% 0.500000, 75% 1.000000, max 8.0. Markdown cell idx 22 immediately below claims '0 GPU: ~43,000 jobs - median = 0, 25th percentile = 0 (more than half request no GPU)' and 'mean = 0.52'. Recomputed value_counts on the real data: smallest value is 0.01 (216 jobs); there is NO 0.0 bucket. Cell idx 14 markdown states 'Records with zero GPU demand ... have been removed'. Opened the PNG: no zero bar exists (impossible on a log axis anyway). The '...

**Düzeltme:** Delete the fabricated zero-GPU bullet from markdown cell idx 22 and restate from the printed describe(): mean 0.68, median 0.50, p25 0.25, p75 1.00, max 8.0, n = 82,184; note zero-GPU jobs were removed upstream. In thesis:62 change '0.52 GPUs' to '0.68 GPUs' and 'approximately 37,000 jobs requesting one or fewer GPUs' to 'approximately 37,000 jobs (45%) requesting exactly one full GPU; a further 52.5% request a fraction of a device'. Add n to the title via `ax.set_title(f"...(n = {len(job_df):,}; zero-GPU jobs excluded upstream)")`.

**Konum:** `notebooks/en/01_data_overview.ipynb cell idx 21 (id cd17) and markdown cell idx 22; thesis/latex/chapters/3.dataset_and_workload.tex:62`


### [CRITICAL] NB01-Figure01..06 (all six) — dpi-quality
**Sorun:** All six figures are inline notebook renders at figure.dpi = 100. Effective print resolution is 190-280 ppi (below the 300 dpi journal requirement) and in-figure text prints at roughly 3.7-6.2 pt (below the 6-8 pt floor).

**Kanıt:** VERIFIED BY MEASUREMENT. PIL reports dpi=(100,100) on all six PNGs. Pixel dims: Fig01 1383x492, Fig02 983x484, Fig03 1383x484, Fig04 984x484, Fig05 984x484, Fig06 1281x484. thesis/latex/thesis.cls:39 loads a4paper and main.tex:5 uses \documentclass[msc]{thesis}, which selects the \@dtype=\@ne branch at thesis.cls:96-107 with left=3.5cm right=2cm, so \textwidth = 21 - 3.5 - 2 = 15.5 cm = 6.10 in (confirmed by reading both files). Include widths read from chapter 3: Fig01 and Fig03 at \textwidth, Fig02 at 0.85\textwidth, Fig04/Fig05/Fig06 at 0.75\textwidth. Resulting ppi: Fig01/Fig03 227, Fig02 ...

**Düzeltme:** Stop shipping inline 100-dpi renders. In cd02 set savefig.dpi 400 / figure.dpi 150 and explicit font sizes, shrink canvases toward printed size ((14,5)->(7.0,3.0) for the \textwidth figures cd11/cd15, (10,5)->(5.2,3.0) for cd13/cd17/cd19, (14,5)->(6.5,2.8) for cd21), add `fig.savefig(path, dpi=400, bbox_inches="tight")` per figure cell, and have scripts/export_thesis_results.py copy those files instead of scraping the base64 inline PNGs. Drop the fontsize=8 override in cd17 once resized.

**Konum:** `notebooks/en/01_data_overview.ipynb cell idx 3 (id cd02) rcParams block; figure cells cd11/cd13/cd15/cd17/cd19/cd21; thesis/latex/thesis.cls:96-107; thesis/latex/main.tex:5`


### [MAJOR] NB01-Figure06.png (thesis/latex/figures/nb01-fig06-arrival-heatmap.png) — units
**Sorun:** The x-axis label 'Hour of Day' plus the notebook and thesis text assert these hours are UTC wall-clock time. They are not: submit_time is a trace-relative offset normalised to 0, so dt.hour is (seconds since trace start) mod 24 h with unknown phase. The project's own source docstring says exactly this.

**Kanıt:** VERIFIED IN THREE PLACES. (1) Data: job_df describe() in cell cd09's output shows arrival_time min = 1970-01-01 00:00:03 and arrival_sec min = 0.000000 - the epoch anchor proves submit_time is a normalised offset, not a real timestamp. Notebook markdown idx 12 itself says 'normalized Unix timestamps relative to the beginning of the trace'. (2) src/feature_engineering.py:172-179 docstring for hour_of_day: the trace's submit_time is 'a relative offset, not a real timestamp ... so this is a 24-hour cyclical phase of unknown absolute alignment -- still a meaningful diurnal-rhythm signal, just not ...

**Düzeltme:** Change the cd21 label to `ax.set_xlabel("Hour Index within Trace Day (24 h cyclical phase; absolute alignment unknown)")`. Delete the 'correspond to UTC time' sentence from notebook markdown idx 26, the 'valid UTC timestamp' sentence from markdown idx 20, and the same claim at thesis:84, replacing all three with the wording already in the src docstring.

**Konum:** `notebooks/en/01_data_overview.ipynb cell idx 25 (id cd21), markdown idx 20 and idx 26; src/feature_engineering.py:172-179; thesis/latex/chapters/3.dataset_and_workload.tex:84`


### [MAJOR] NB01-Figure04.png (thesis/latex/figures/nb01-fig04-gpu-demand.png) — units
**Sorun:** The LaTeX caption and the figure disagree on what the x-axis measures, and the caption's headline claim is contradicted by the plotted bars and by describe().

**Kanıt:** VERIFIED. thesis/latex/chapters/3.dataset_and_workload.tex:67 caption reads 'Histogram of the number of GPUs requested per job. It is evident that almost every request is for a single GPU.' The figure's own axis label (read from the rendered PNG and from cd17 source) is 'GPU Demand per Job (fraction of a device, log scale)' with tick values 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 8 - fractional device shares, not integer counts. Recomputed exact counts from the data: 1.0 -> 37,033 (45.1%); sub-1.0 total -> 43,176 (52.5%), made up of 0.25 -> 20,615, 0....

**Düzeltme:** Rewrite the caption at thesis:67 to match the axis: 'Distribution of GPU demand per job, expressed as a fraction of a physical device (log-scaled x-axis, n = 82,184). Exactly one full GPU is the single most common request (37,033 jobs, 45%), but the median request is 0.50 of a device: the majority (52.5%) are sub-device shares. Multi-GPU requests (2-8) are rare (2.4%).' Use 'GPU demand (fraction of a device)' consistently in axis label, caption and body; reserve 'number of GPUs' for integer counts.

**Konum:** `thesis/latex/chapters/3.dataset_and_workload.tex:62,67; notebooks/en/01_data_overview.ipynb cell idx 21 (id cd17)`


### [MAJOR] NB01-Figure03.png (thesis/latex/figures/nb01-fig03-arrival-rate.png) — caption-mismatch
**Sorun:** The thesis prose claims a daily cycle and a weekly cyclicality that the plot does not show and the trace length cannot support; the notebook's own commentary for the same figure says the opposite. Separately, both body and caption call this an '8-day trace' while the data span 7.66 days.

**Kanıt:** VERIFIED BY EYE AND BY MEASUREMENT. Rendered the PNG: the series is high-frequency burst noise on a ~200-600 jobs/hr baseline with isolated spikes at ~day 1.45 (~1,350) and ~day 5.7 (~1,080); no 24 h periodicity is visually resolvable. thesis/latex/chapters/3.dataset_and_workload.tex:84 nonetheless says the figure 'clearly illustrates that user-submitted jobs exhibit a clear daily cycle' and that 'the weekly and daily cyclicality patterns of submission ... will still be correct'. Notebook markdown idx 20 for the same figure says 'The arrival process is not homogeneous - it exhibits a bursty st...

**Düzeltme:** Replace the 'clear daily cycle' / 'weekly cyclicality' sentences at thesis:84 with the burstiness claim the notebook actually supports. Change '8 days' / '8-day trace' to '~7.7 days' at thesis:84 and :89 (and check line 164/169, which also say '8-day trace'). Add `ax.set_xlim(0, days.max())` in cd15. If a diurnal claim is wanted, support it with a periodogram or a 24 h-folded mean overlay rather than this plot.

**Konum:** `notebooks/en/01_data_overview.ipynb cell idx 19 (id cd15) and markdown idx 20; thesis/latex/chapters/3.dataset_and_workload.tex:84,89 (also 164,169)`


### [MAJOR] NB01-Figure01.png (thesis/latex/figures/nb01-fig01-runtime-dist.png) — naming-inconsistency
**Sorun:** The suptitle labels the figure '100K' while the panels plot the 82,184-job filtered subset, overstating n by ~22%. It is also the only NB01 figure carrying any dataset identifier.

**Kanıt:** VERIFIED. cd11 sets `plt.suptitle("Job Runtime Distribution - Alibaba PAI 100K", ...)`; confirmed visible in the rendered PNG. The panels plot `job_df["job_runtime"]`, and cell cd09's own printed output says 'Valid GPU jobs : 82,184 (filtered 17,816 invalid rows)'. I re-ran the pipeline and confirmed len(job_df) = 82,184. 100,000/82,184 = 1.217, so '100K' overstates the plotted population by 21.7%. Rendered all six PNGs: Figure 1 is the only one with any dataset label; none of the six states its n.

**Düzeltme:** In cd11 change the suptitle to `plt.suptitle(f"Job Runtime Distribution - Alibaba PAI Trace (n = {len(job_df):,} valid GPU jobs)", ...)` and add n = 82,184 to the LaTeX caption at thesis/latex/chapters/3.dataset_and_workload.tex:45. Apply one rule across NB01 - either every figure states n or none does in-figure and all do in the caption.

**Konum:** `notebooks/en/01_data_overview.ipynb cell idx 15 (id cd11); thesis/latex/chapters/3.dataset_and_workload.tex:45`


### [MAJOR] nb03-fig01-cluster-load.png / nb02-fig03 / nb01-fig04 / nb03-fig02 — naming-inconsistency
**Sorun:** 'GPU Demand' names two quantities two orders of magnitude apart within chapter 3, plus two further spellings.

**Kanıt:** Confirmed, and one spelling more than reported. nb03-fig01 cell cd12 uses label='GPU Demand' and title 'Cluster State: Total GPU Demand Over Time' for cluster_load_gpu, whose plotted range is 0-868 (calibrated from gridlines). nb02-fig03 cell cd12 uses x-label 'GPU Demand (units)' for the per-job gpu_demand, range 0-8. nb01-fig04 (also chapter 3) uses a third string, 'GPU Demand per Job (fraction of a device, log scale)'. nb03-fig02 uses a fourth, the raw identifier 'gpu_demand'. All four figures sit in the same chapter.

**Düzeltme:** Fix one string per quantity: per-job = 'GPU demand (GPUs)', cluster-wide = 'Cluster GPU load (GPUs)'. Define a single LABELS dict in src/visualization.py and import it into every notebook.

**Konum:** `notebooks/en/03_feature_engineering.ipynb cd12 (idx 15); notebooks/en/02_workload_analysis.ipynb cd12 (idx 14); notebooks/en/01_data_overview.ipynb cd17 (idx 21)`


### [MINOR] NB01-Figure02.png (thesis/latex/figures/nb01-fig02-runtime-cdf.png) — color
**Sorun:** The two vertical reference lines are distinguished by hue alone - identical dash pattern, crimson vs orange - which converges under deuteranopia/protanopia and in greyscale, and the legend swatches collapse the same way.

**Kanıt:** VERIFIED IN CODE AND IMAGE. cd13 source: `ax.axvline(..., color="crimson", ls="--", label=...)` and `ax.axvline(..., color="orange", ls="--", label=...)` - same ls, same default lw, differing only in hue. Rendered the PNG and confirmed both lines draw with a visually identical dash pattern and identical line weight; the legend shows two dashed swatches separable only by colour. This is the only NB01 figure that encodes meaning in colour at all (the other five are single-series), and it does so non-accessibly. Mitigating context: the two lines sit far apart on the x-axis (594 s vs 24,110 s) so ...

**Düzeltme:** In cd13 give the lines distinct dash patterns as well as colours: `ls="--", lw=1.6` for the median and `ls=":", lw=1.8` for P95. Optionally add `ax.annotate` text anchors at the top of each line so identification survives greyscale printing.

**Konum:** `notebooks/en/01_data_overview.ipynb cell idx 17 (id cd13)`


### [MINOR] NB01-Figure02.png (thesis/latex/figures/nb01-fig02-runtime-cdf.png) — other
**Sorun:** Legend number formatting is inconsistent with the surrounding thesis text: no thousands separator and no space before the unit.

**Kanıt:** VERIFIED. cd13 uses `f"Median = {np.median(runtimes_sorted):.0f}s"` and `f"P95 = {np.percentile(runtimes_sorted, 95):.0f}s"`. Rendered the PNG and read the legend: 'Median = 594s' and 'P95 = 24110s' - unseparated 5-digit value, unit glued to the number. The thesis body at chapter 3 and the fig:runtime-cdf caption both write '594\,s' and '24,110\,s' (thin space, grouped separator). The cell's own print block below already uses the correct `{:>10,.0f}` grouping, so the figure is the only place with the ungrouped form.

**Düzeltme:** In cd13 change to `f"Median = {np.median(runtimes_sorted):,.0f} s"` and `f"P95 = {np.percentile(runtimes_sorted, 95):,.0f} s"`. Apply the same `{:,.0f} s` pattern to any other numeric annotation in NB01.

**Konum:** `notebooks/en/01_data_overview.ipynb cell idx 17 (id cd13)`


### [MINOR] NB01-Figure01..06 (all six) — other
**Sorun:** Every figure bakes a title into the raster which is then duplicated by the LaTeX \caption underneath, creating a second place where wording can drift out of sync.

**Kanıt:** VERIFIED FOR ALL SIX. Source: cd11 `plt.suptitle` + two `set_title`, cd13/cd15/cd17/cd19/cd21 each `ax.set_title(...)`. Rendered each PNG and confirmed the title is visible in the raster. Concrete duplication pair: the Fig02 image reads 'Cumulative Distribution of Job Runtimes' and the caption directly below it reads 'Cumulative distribution function of job runtimes.' Same for Fig04 ('Distribution of GPU Demand per Job' vs the caption) and Fig06 ('Job Arrival Heatmap - Day-of-Week x Hour-of-Day' vs 'Heatmap of job arrivals per hour of the day and per trace-relative day index'). The Fig06 pair ...

**Düzeltme:** Add a module-level `SHOW_TITLES = False` in cd02 and wrap each title call (`if SHOW_TITLES: ax.set_title(...)`), or strip titles at save time, keeping the LaTeX \caption as the single source of truth.

**Konum:** `notebooks/en/01_data_overview.ipynb cells cd11, cd13, cd15, cd17, cd19, cd21`


### [MINOR] NB01-Figure01.png (thesis/latex/figures/nb01-fig01-runtime-dist.png) — axis
**Sorun:** The left panel's x tick labels are raw six-digit integers with no thousands separator and no offset notation, and the panel is near-degenerate: 84% of the mass falls in the first of 80 linear bins.

**Kanıt:** VERIFIED. Rendered the PNG: left panel x ticks read 0, 100000, 200000, 300000, 400000, 500000, 600000 - seven unseparated six-digit strings, printed at ~4.9 pt given the 0.441 scale factor. cd11 sets no formatter. On degeneracy, I recomputed `np.histogram(job_runtime, bins=80)`: the first bin spans [4.0, 7497.0] and holds 69,343 of 82,184 jobs = 84.4%. The rendered panel confirms this - a single spike near x=0 and essentially flat elsewhere, consuming half the figure width to convey one bar.

**Düzeltme:** In cd11 add `from matplotlib.ticker import FuncFormatter; axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:g}k"))` with `axes[0].set_xlabel("Runtime (thousands of seconds)")`. To make the panel informative, either clip to `axes[0].set_xlim(0, np.percentile(job_df["job_runtime"], 99))` (stating the clip in the caption) or add an inset zoom on the 0-20,000 s band.

**Konum:** `notebooks/en/01_data_overview.ipynb cell idx 15 (id cd11)`


### [MINOR] NB01-Figure05 (current notebook output of cell cd19, not the shipped PNG) — other
**Sorun:** In the corrected log-spaced-bin version the notebook now produces, the low end shows an empty-bin comb: bars at 1, 2, 3 and 4 s are separated by visibly empty bins because logspace bins below ~5 s are narrower than the 1 s integer granularity of the data.

**Kanıt:** VERIFIED BY EYE ON THE EXTRACTED CELL OUTPUT. I decoded cd19's embedded image/png (sha1 c1d686c8..., the version NOT yet shipped to the thesis) and rendered it: distinct isolated bars at x=1 (~12,100), x=2 (~6,800), x=3 (~5,700), x=4 (~4,200) with clearly empty gaps between them, converging into a continuous histogram only above ~5 s. The code is `np.logspace(np.log10(inter_arrivals.min()), np.log10(inter_arrivals.max()), 40)` with min = 1 s; inter-arrival values are integer seconds (confirmed: dt==1 count is exactly 12,110), so bins narrower than 1 s in the low decade can contain no possible ...

**Düzeltme:** Use integer-aware bin edges in cd19: `_edges = np.unique(np.concatenate([np.arange(0.5, 10.5, 1.0), np.logspace(np.log10(10.5), np.log10(inter_arrivals.max()), 25)]))` then `ax.hist(inter_arrivals, bins=_edges, ...)` - one bin per integer second below 10 s, log-spaced above. Fix this together with the stale-export finding so the regenerated figure is correct on first ship.

**Konum:** `notebooks/en/01_data_overview.ipynb cell idx 23 (id cd19)`


### [MINOR] NB01-Figure01..06 (all six) — color
**Sorun:** Colour is assigned ad hoc across the figure set and carries no consistent meaning; the same hue marks different quantities in different figures.

**Kanıt:** VERIFIED BY READING ALL SIX CELLS AND ALL SIX RENDERED PNGs. Explicit colour arguments: cd11 `color="steelblue"` (left panel) and `color="darkorange"` (right panel); cd13 `color="royalblue"` for the CDF plus `color="crimson"` and `color="orange"` for the reference lines; cd15 `color="seagreen"`; cd17 `color="steelblue"`; cd19 `color="mediumpurple"`; cd21 `cmap="YlGnBu"`. Cross-figure collisions confirmed in the images: orange marks the log-scale runtime histogram in Fig01 but the P95 reference line in Fig02; steelblue marks runtime in Fig01 but GPU demand in Fig04. Every one of these plots exc...

**Düzeltme:** Define one role-keyed palette in cd02, e.g. `C = {"runtime": "#2b6cb0", "arrival": "#2f855a", "resource": "#6b46c1", "ref_a": "#c53030", "ref_b": "#b7791f"}`, then use `C["runtime"]` in cd11 and cd13, `C["arrival"]` in cd15 and cd19, `C["resource"]` in cd17, and keep YlGnBu in cd21. One hue per quantity, reused across every figure in the chapter.

**Konum:** `notebooks/en/01_data_overview.ipynb cell idx 3 (id cd02) and all six figure cells`


### [MINOR] NB02-Figure01.png (export-only) and nb01-fig01-runtime-dist.png (in thesis) — units
**Sorun:** Log-transformed runtime axis carries no unit, and the same quantity is named four different ways across chapter 3.

**Kanıt:** Primary claim CONFIRMED, secondary claim REJECTED. Confirmed: the x-label is literally 'log₁₀(Runtime + 1)' with no unit (nb02 cell cd08 and nb01 cell cd11 right panel — the latter IS a thesis figure, nb01-fig01). Runtime is named four ways in chapter 3: 'Runtime (seconds)' (nb01-fig01 left), 'log₁₀(Runtime + 1)' (nb01-fig01 right and NB02-Figure01), 'Runtime (seconds, log scale)' (nb02-fig03), 'job_runtime' (nb03-fig02). REJECTED: the 'about 16% of the axis is empty, a tick at 6 where no job exists' sub-claim does not hold up. The x-limits come from np.histogram's data-driven bin edges plus m...

**Düzeltme:** Set ax.set_xlabel(r'$\\log_{10}(\\mathrm{runtime}\\,[\\mathrm{s}] + 1)$') in nb01 cd11 and nb02 cd08, and adopt one runtime label string across all chapter-3 figures. Do not bother forcing xlim.

**Konum:** `notebooks/en/02_workload_analysis.ipynb cd08 (idx 10); notebooks/en/01_data_overview.ipynb cd11 (idx 15)`


### [MINOR] NB02-Figure01.png vs nb01-fig01-runtime-dist.png — color
**Sorun:** The same variable (job runtime) is drawn in three different colours, including two colours inside one figure with no legend.

**Kanıt:** Confirmed in both code and images. notebooks/en/01_data_overview.ipynb cell cd11 draws the left 'Raw Runtime Distribution' panel with color='steelblue' and the right 'Runtime Distribution (Log10 Scale)' panel with color='darkorange' — same variable, two colours, no legend explaining the switch. Verified visually in nb01-fig01-runtime-dist.png (a thesis figure): left panel blue, right panel orange. notebooks/en/02_workload_analysis.ipynb cell cd08 then draws the identical log10 histogram in color='steelblue', so the same chart is blue in one figure and orange in another. Note the second half of...

**Düzeltme:** Define one palette constant (e.g. PALETTE['runtime']) in src/visualization.py and use it in both nb01 cd11 panels and nb02 cd08; reserve a colour change for an actual change of quantity.

**Konum:** `notebooks/en/01_data_overview.ipynb cd11 (idx 15); notebooks/en/02_workload_analysis.ipynb cd08 (idx 10)`


### [MINOR] NB02-Figure02.png vs nb01-fig03-arrival-rate.png — other
**Sorun:** NB02's arrival-rate chart duplicates NB01's, with only the title wording differing.

**Kanıt:** Duplication CONFIRMED, two qualifications. Both PNGs are 1383x484 at 100 dpi. Pixel diff: 16,556 pixels differ by >20, ~9,000 of them in the title band (y=10-44). Titles verified in code: nb01 cell cd15 'Hourly Job Arrival Rate Over the Trace Duration' vs nb02 cell cd10 'Hourly Job Arrival Rate Over Trace Duration'. Both use figsize=(14,5), color='seagreen', plot + fill_between(alpha=0.15). QUALIFICATION 1: the report's 'differences confined almost entirely to the title band plus antialiasing' is not accurate — there is also a systematic ~5,296-pixel band at y=190-199 spread uniformly across t...

**Düzeltme:** If NB02 need not be self-contained, delete cells cd08/cd10 and cross-reference the NB01 figures; then update EXPECTED_FIGURE_COUNT['NB02'] (currently 3) and the ('NB02', 3) key in THESIS_FIGURE_MAP, which address figures by position.

**Konum:** `notebooks/en/02_workload_analysis.ipynb cd08/cd10 (idx 10, 12); notebooks/en/01_data_overview.ipynb cd15 (idx 19); scripts/export_thesis_results.py:76-85,118-120`


---

## NB02 — 8 bulgu (2 kritik)

### [CRITICAL] nb02-fig03-gpu-vs-runtime.png — caption-mismatch
**Sorun:** Text and caption claim both axes are log-scaled; the x-axis is linear.

**Kanıt:** Verified in code AND image. notebooks/en/02_workload_analysis.ipynb cell cd12 (idx 14) contains only `ax.set_yscale("log")` — no set_xscale call anywhere in the cell. Opened the PNG: x ticks are evenly spaced integers 0,1,2,3,4,5,6,7,8 (linear). thesis/latex/chapters/3.dataset_and_workload.tex:72 reads '...using 5,000 randomly selected samples, with both axes on a log scale' and line 77 caption reads '(5,000-job sample, log scale)'. Both are false for the x-axis.

**Düzeltme:** Either add ax.set_xscale("symlog", linthresh=0.1) + explicit ticks at the real demand levels, or correct tex:72 to 'with the runtime axis on a log scale' and tex:77 to '(5,000-job sample, log-scaled runtime axis)'.

**Konum:** `notebooks/en/02_workload_analysis.ipynb cell cd12 (idx 14); thesis/latex/chapters/3.dataset_and_workload.tex:72,77`


### [CRITICAL] All notebook-derived thesis figures (26 files), incl. nb02-fig03, nb03-fig01, nb03-fig02 — dpi-quality
**Sorun:** Figures are exported at 100 DPI; effective print resolution is 147-243 DPI, below the 300 DPI publisher floor.

**Kanıt:** Confirmed, and broader than reported. PIL scan of ALL 29 files in thesis/latex/figures/: every one of the 26 notebook-derived PNGs carries dpi=(99.9998, 99.9998). Sizes match the report exactly (nb02-fig03 920x584, nb03-fig01 1484x784, nb03-fig02 899x784). Loading the project's own theme in ./venv confirms figure.dpi=100.0 and savefig.dpi='figure'. scripts/export_thesis_results.py base64-decodes the notebook's inline image/png output (lines ~155-165), so it inherits that 100 dpi. Page geometry confirmed: main.tex uses \documentclass[msc], thesis.cls:31 maps msc->\@dtype=\@ne, thesis.cls:96-105...

**Düzeltme:** Set mpl.rcParams['figure.dpi']=300 and ['savefig.dpi']=300 in the setup cell (cd02) of both notebooks and their Turkish twins, re-run, re-run scripts/export_thesis_results.py. Better: also emit PDF via %config InlineBackend.figure_formats.

**Konum:** `notebooks/en/02_workload_analysis.ipynb cd02 (idx 3); notebooks/en/03_feature_engineering.ipynb cd02 (idx 3); scripts/export_thesis_results.py extract function`


### [MAJOR] nb03-fig01-cluster-load.png (worst), nb02-fig03, nb03-fig02 — dpi-quality
**Sorun:** Figures authored much larger than printed size, shrinking tick labels below the legible floor.

**Kanıt:** Core claim confirmed, two numbers corrected. Authored sizes verified in code: nb03-fig01 figsize=(15,8), nb03-fig02 (11,8), nb02-fig03 (10,6). Rendered widths (after bbox tight) 14.84in / 8.99in / 9.20in. Printed at 6.102in (\textwidth) or 4.577in (0.75\textwidth) => scale factors 0.411 / 0.679 / 0.497, matching the report. CORRECTION 1: tick labels are NOT 10pt — loading sns.set_theme(style='whitegrid', palette='muted') plus the notebooks' rcParams block in ./venv gives xtick.labelsize=11.0, ytick.labelsize=11.0, legend.fontsize=11.0. CORRECTION 2: recomputing at 11pt gives printed sizes 4.5p...

**Düzeltme:** Author at final print size: set figure.figsize near (6.1, 3.4) and xtick/ytick/legend fontsize 8, axes.labelsize 9 in each setup cell, size each figure to (6.1, h), and use \includegraphics[width=\textwidth] (drop the 0.75 on nb02-fig03).

**Konum:** `notebooks/en/02_workload_analysis.ipynb cd02/cd12; notebooks/en/03_feature_engineering.ipynb cd02/cd12/cd18`


### [MAJOR] nb02-fig03-gpu-vs-runtime.png — legend-error
**Sorun:** Colorbar is drawn at alpha 0.3 while overplotted markers render at full saturation, so the key does not match any mark.

**Kanıt:** Confirmed by pixel sampling. Code: cell cd12 does `sc = ax.scatter(..., alpha=0.3, s=8, c=sample['gpu_demand'], cmap='plasma')` then `plt.colorbar(sc, ax=ax, label='GPU Demand')` — the colorbar inherits the mappable's alpha. Sampled the colorbar strip at x=845 (extent y=32..522): near value 1 (y=460) it is RGB(201,178,226), pale lavender. The most common dark marker colors in the plot area (x<820) are RGB(75,3,161) with 663 px and RGB(32,6,143) with 614 px — deep saturated purple. The key is quantitatively decoupled from the marks.

**Düzeltme:** Remove the colorbar entirely (it duplicates the x-axis anyway), or build an un-alpha'd ScalarMappable for the colorbar and set per-point alpha through RGBA face colors instead of the scatter-wide alpha kwarg.

**Konum:** `notebooks/en/02_workload_analysis.ipynb cell cd12 (idx 14)`


### [MAJOR] nb02-fig03-gpu-vs-runtime.png — color
**Sorun:** High-GPU markers render pale yellow with 1.14:1 contrast against white — effectively invisible.

**Kanıt:** Confirmed by measurement. Masked yellowish pixels (R>200, G>200, B<200) inside the plot area: only 156 pixels total, x range 513-763 (the 4/5/6/8-GPU columns). Median color RGB(242, 245, 163.5); computed WCAG contrast against white = 1.14:1 (requirement for non-text graphical objects is 3:1). Visual inspection of the PNG confirms the 6- and 8-GPU points are barely discernible. This directly undercuts the body-text claim at 3.dataset_and_workload.tex:72 that 'Higher GPU requests (6-8 GPUs) are rare but appear close to the 10^4 s mark'.

**Düzeltme:** Drop the colormap (colour duplicates the x-axis) and use a single high-contrast mark, or keep viridis with edgecolors='black', linewidths=0.2 so pale markers retain an outline.

**Konum:** `notebooks/en/02_workload_analysis.ipynb cell cd12 (idx 14)`


### [MAJOR] nb02-fig03-gpu-vs-runtime.png — legend-error
**Sorun:** Colorbar double-encodes the x-axis variable and consumes plot width for zero information.

**Kanıt:** Confirmed with one number corrected. Code: `c=sample['gpu_demand']` is the identical variable already on the x-axis (`ax.scatter(sample['gpu_demand'], ...)`). Colorbar label is 'GPU Demand' while the x-axis label is 'GPU Demand (units)' — same quantity, two strings, in one figure. CORRECTION: the colorbar assembly (bar x=838-864, tick labels to ~890, rotated label ~895-905) occupies roughly 108 px of the 920 px width (~12%), not the reported ~140 px (~15%). Separately confirmed that '(units)' is a wrong unit: nb01-fig04 shows gpu_demand takes fractional values (0.01 ... 0.8), so the quantity i...

**Düzeltme:** Delete the plt.colorbar(sc, ...) line and the c=/cmap= arguments; change the x-label to ax.set_xlabel('GPU demand (GPUs)') or match nb01-fig04's 'GPU Demand per Job (fraction of a device)'.

**Konum:** `notebooks/en/02_workload_analysis.ipynb cell cd12 (idx 14)`


### [MAJOR] nb02-fig03-gpu-vs-runtime.png — axis
**Sorun:** Linear integer x-axis collapses all fractional GPU-demand levels into an unresolvable smear; a tick at 7 has no data.

**Kanıt:** Confirmed and UNDERSTATED by the report. The report guessed '4-5 distinct fractional levels'. Reading nb01-fig04-gpu-demand.png (which plots the same variable on a log axis with explicit ticks) shows the real levels are 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 8 — twelve sub-1 levels, all crushed between the '0' and '1' ticks in nb02-fig03. Zooming x=60..230 of nb02-fig03 shows exactly this: overlapping smeared sub-columns. There is no level 7, confirming the empty x=7 tick, and no level 0, so the '0' tick label is not a real data value.

**Düzeltme:** Set the actual demand levels as ticks on a symlog axis, or replace the scatter with a per-level distribution view (stripplot/boxplot over gpu_demand as a category).

**Konum:** `notebooks/en/02_workload_analysis.ipynb cell cd12 (idx 14)`


### [MAJOR] nb02-fig03-gpu-vs-runtime.png — other
**Sorun:** Severe overplotting saturates the dense columns into solid opaque bars, destroying the within-class density the caption asks the reader to see.

**Kanıt:** Confirmed by zoom and pixel sampling. Zoomed crop of x=60..230 (the 0-to-1 region) shows the 0.1, 0.25, 0.5 and 1 columns as solid opaque vertical bars with no visible internal density variation. Pixel counts in the plot area: RGB(75,3,161) 663 px and RGB(32,6,143) 614 px, i.e. large uniform runs of one colour rather than a tonal gradient. Cause is in code: s=8 with alpha=0.3 and a discrete x variable, no jitter. The caption at 3.dataset_and_workload.tex:77 asks the reader to see 'the large variability within each GPU class' — which is exactly what the saturated bars hide.

**Düzeltme:** Add horizontal jitter and drop alpha to ~0.1 with rasterized=True, or overlay a per-level boxplot (showfliers=False) so median and IQR per class survive the print shrink.

**Konum:** `notebooks/en/02_workload_analysis.ipynb cell cd12 (idx 14)`


---

## NB03 — 13 bulgu (1 kritik)

### [CRITICAL] nb03-fig02-correlation.png — caption-mismatch
**Sorun:** Thesis body text states gpu_demand correlation = 0.05; figure and underlying data say 0.06.

**Kanıt:** Strongest verification of the whole set. (a) Zoomed the heatmap's first column: gpu_demand x job_runtime cell reads 0.06, num_cpu x job_runtime reads 0.08. (b) The notebook's own stdout in cell cd18 prints `gpu_demand 0.064362` and `num_cpu 0.080097` — so 0.06 is not a rounding artifact of the annotation. (c) thesis/latex/chapters/3.dataset_and_workload.tex:214 reads '...the Pearson correlation coefficient values for num_cpu and gpu_demand are remarkably small at 0.08 and 0.05, respectively.' num_cpu matches, gpu_demand does not.

**Düzeltme:** Change '0.08 and 0.05' to '0.08 and 0.06' at 3.dataset_and_workload.tex:214, or better emit the rounded values from the notebook into a \newcommand file the .tex includes.

**Konum:** `thesis/latex/chapters/3.dataset_and_workload.tex:214; notebooks/en/03_feature_engineering.ipynb cell cd18 (idx 21)`


### [MAJOR] nb03-fig01-cluster-load.png — label-overlap
**Sorun:** Lower-panel legend box sits on top of the data and washes out the tip of a late-trace peak.

**Kanıt:** Occlusion CONFIRMED by pixel scan at x=1393 (day ~7.5): y=420-427 white, y=428-432 RGB(255,241,224) (orange bleeding through the semi-transparent legend frame), y=433 RGB(214,205,195) (legend bottom border), y=434+ full-strength orange RGB(255,186,102). A zoomed crop of (1300,390)-(1484,470) visibly shows the peak's tip clipped by the legend box. Code confirms axes[1].legend(loc='upper right') with default framealpha. CORRECTION: the report calls this 'the highest excursion in that panel' — it is not. Tracing the topmost orange pixel across the whole lower panel gives y=413 at x=1058 (day ~5.5...

**Düzeltme:** Delete both single-entry legends (see the redundant-legend finding), or move them to loc='upper left' with framealpha=1.0 and add headroom via set_ylim(0, max*1.18).

**Konum:** `notebooks/en/03_feature_engineering.ipynb cell cd12 (idx 15)`


### [MAJOR] nb03-fig01-cluster-load.png — naming-inconsistency
**Sorun:** Both y-axis labels overstate the plotted quantities and contradict the thesis's own feature table.

**Kanıt:** Confirmed; only the cited line numbers were slightly off. Cell cd12 plots job_df['active_job_count'] with ylabel 'Active Job Count', and job_df['cluster_load_gpu'] with ylabel 'Total GPUs Requested'. src/feature_engineering.py defines these at lines 323-324 (the report said 312-313): `df['cluster_load_gpu'] = df['load_gpu'] - df['gpu_demand']` and `df['active_job_count'] = df['active_jobs'] - 1`, with the in-code comment 'subtract to get the background load *excluding* this job'. Table tab:features in 3.dataset_and_workload.tex correctly defines them as 'Background GPU load at job arrival' and...

**Düzeltme:** axes[0].set_ylabel('Concurrent other jobs'); axes[1].set_ylabel('Background GPU demand (GPUs)'); update both titles and the caption at 3.dataset_and_workload.tex:169.

**Konum:** `notebooks/en/03_feature_engineering.ipynb cd12 (idx 15); src/feature_engineering.py:323-324`


### [MAJOR] nb03-fig02-correlation.png — naming-inconsistency
**Sorun:** Every tick label is a raw snake_case Python identifier with no units; day_of_week is actively misleading.

**Kanıt:** Confirmed. All ten row and column labels in the rendered PNG are code identifiers: job_runtime, gpu_demand, num_cpu, num_inst, arrival_sec, hour_of_day, day_of_week, cluster_load_cpu, cluster_load_gpu, active_job_count. Cell cd18 passes the dataframe straight to sns.heatmap with no rename, so the labels come from the column names. No units appear anywhere in the figure. This clashes with the prose labels used by the other chapter-3 figures ('GPU Demand (units)', 'Runtime (seconds, log scale)', 'Total GPUs Requested'). The day_of_week point is verified: Table tab:features in 3.dataset_and_workl...

**Düzeltme:** Rename for display only before plotting (corr = corr.rename(index=LABELS, columns=LABELS)) and add a caption sentence noting the day index is trace-relative.

**Konum:** `notebooks/en/03_feature_engineering.ipynb cell cd18 (idx 21)`


### [MINOR] nb03-fig01-cluster-load.png — legend-error
**Sorun:** Each panel has one series but carries a one-entry legend duplicating the title and y-label; grid is also reconfigured redundantly.

**Kanıt:** Confirmed. Cell cd12: axes[0] plots one line labelled 'Active Jobs' with title 'Cluster State: Active Job Count Over Time' and ylabel 'Active Job Count' — the same information three times — then calls axes[0].legend(loc='upper right'). Same pattern in axes[1]. Verified visually: both legends contain exactly one entry. CORRECTION to the report's phrasing: axes[i].grid(True, alpha=0.3) does not literally 'redraw a second grid over the first' — it reconfigures the existing whitegrid gridlines' alpha. The call is redundant given sns.set_theme(style='whitegrid'), but the described double-drawing do...

**Düzeltme:** Delete both legend() calls and the label= kwargs; drop the redundant grid(True, alpha=0.3) calls.

**Konum:** `notebooks/en/03_feature_engineering.ipynb cell cd12 (idx 15)`


### [MINOR] nb03-fig01-cluster-load.png — caption-mismatch
**Sorun:** Body text asserts business-hours peaks and overnight dips, but the figure has no wall-clock reference; the stated peak value is also low.

**Kanıt:** Confirmed. The figure's only time reference is x-label 'Elapsed Time (days)', measured from trace start (job_df['arrival_sec']/86400 in cell cd12) — there are no time-of-day ticks, no wall-clock axis and no diurnal shading, so the reader cannot locate midnight. 3.dataset_and_workload.tex:164 nonetheless states 'It is clear that daily patterns exist within both plots, with peaks during business hours and significant dips overnight'. The same paragraph also acknowledges elsewhere (line ~82) that timestamps are normalized and hour-of-day is not the real submission time, which makes the business-h...

**Düzeltme:** Either add night-band shading plus a stated trace-start timestamp, or reword line 164 to 'a repeating ~24 h cycle is visible in both panels' and change 'near 800' to 'approaching 870'.

**Konum:** `notebooks/en/03_feature_engineering.ipynb cd12 (idx 15); thesis/latex/chapters/3.dataset_and_workload.tex:164`


### [MINOR] nb03-fig01-cluster-load.png — other
**Sorun:** Two-panel figure has no (a)/(b) panel tags although the caption references panels.

**Kanıt:** Confirmed as a factual matter. Cell cd12 creates plt.subplots(2,1) and adds no text/annotation panel tags — verified in the code and by inspecting the PNG (no letters anywhere). The caption at 3.dataset_and_workload.tex:169 does refer to '(upper panel)' and '(lower panel)'. Note the report's framing that 'most journals require lettered subpanels' is a style assertion I did not verify against any publisher guide; the verifiable part is simply that the panels are untagged while the caption addresses them positionally.

**Düzeltme:** Add ax.text(-0.06, 1.02, '(a)'/'(b)', transform=ax.transAxes, fontweight='bold') after creating the axes and switch the caption to (a)/(b).

**Konum:** `notebooks/en/03_feature_engineering.ipynb cell cd12 (idx 15)`


### [MINOR] nb03-fig02-correlation.png — other
**Sorun:** Seaborn whitegrid gridlines render through the masked upper triangle.

**Kanıt:** Confirmed visually. Zoomed the region (430,30)-(899,320) at 2x: a clear grey lattice of horizontal and vertical rules covers the entire empty upper-right half of the matrix. Code confirms the cause: sns.set_theme(style='whitegrid') is active from cell cd02 and cell cd18 never calls ax.grid(False) after sns.heatmap(). It is the most conspicuous cosmetic defect in this figure.

**Düzeltme:** Add ax.grid(False) immediately after the sns.heatmap(...) call in cell cd18, or wrap the plot in `with sns.axes_style("white"):`.

**Konum:** `notebooks/en/03_feature_engineering.ipynb cell cd18 (idx 21)`


### [MINOR] nb03-fig02-correlation.png — legend-error
**Sorun:** Colorbar is unlabelled and its range is pinned to +/-1 by the uninformative diagonal.

**Kanıt:** Confirmed. Cell cd18 calls sns.heatmap(..., vmin=-1, vmax=1, ...) with no cbar_kws, so the colorbar carries no label — the rendered PNG shows only bare tick numbers, nothing states the encoded quantity is a Pearson r. The diagonal is unmasked (mask uses k=1), so the constant 1.00 diagonal pins the top of the scale. The real off-diagonal range read from the figure is -0.47 to 0.89, so most informative cells fall in the washed-out band near zero — visually confirmed: the matrix reads as a pale grid with a dark red diagonal.

**Düzeltme:** Pass cbar_kws={'label': 'Pearson $r$'}; mask the diagonal too (k=0) so the ramp reflects real structure, or set vmin=-0.5, vmax=0.9.

**Konum:** `notebooks/en/03_feature_engineering.ipynb cell cd18 (idx 21)`


### [MINOR] nb03-fig02-correlation.png — other
**Sorun:** The arrival_sec x num_inst cell displays the meaningless value '-0.00'.

**Kanıt:** Confirmed by zoom. Cropping (275,405)-(355,432) at 10x shows the annotation rendered as '-0.00' with a leading hyphen. Cause is in code: cell cd18 uses annot=True with fmt='.2f', which formats a tiny negative float as -0.00.

**Düzeltme:** Build pre-formatted annotations (annot = corr.map(lambda v: f'{v:.2f}' if abs(v) >= 0.005 else '0.00')) and pass annot=annot, fmt=''.

**Konum:** `notebooks/en/03_feature_engineering.ipynb cell cd18 (idx 21)`


### [MINOR] nb03-fig02-correlation.png — other
**Sorun:** Two different minus glyphs (ASCII hyphen in cell annotations, Unicode minus in colorbar ticks) appear in one figure.

**Kanıt:** Confirmed by measurement and by side-by-side zoom. Measured the horizontal stroke: the colorbar's '-0.50' minus at y=483 spans x=846-854, a 9 px stroke; annotation minus strokes measure ~4 px at comparable digit height. Zoomed crops at 10x show the difference unmistakably — the colorbar glyph is visibly longer and sits higher. Cause verified structurally: fmt='.2f' produces Python's ASCII hyphen for annotations, while matplotlib's tick formatter uses U+2212 because axes.unicode_minus is True (confirmed = True when loading the project's exact theme in ./venv). This is genuinely cosmetic and ver...

**Düzeltme:** Set plt.rcParams['axes.unicode_minus'] = False in the setup cell (cd02), or format annotations with the Unicode minus.

**Konum:** `notebooks/en/03_feature_engineering.ipynb cd02 (idx 3) or cd18 (idx 21)`


### [MINOR] nb03-fig02-correlation.png — axis
**Sorun:** 90-degree-rotated snake_case x tick labels consume roughly a fifth of the figure height.

**Kanıt:** Confirmed by measurement. The x tick-label band (rows below the heatmap containing dark text) spans y=600 to y=773, about 174 px of the figure's 784 px height = 22.2% — very close to the reported 164 px / 21%. The labels are vertical in the rendered PNG (seaborn's default rotation for long tick strings; cell cd18 sets no rotation). Combined with the 0.679 print shrink these render at roughly 7.5 pt vertical text.

**Düzeltme:** After the heatmap call: plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor'), together with the shorter display labels from the naming fix.

**Konum:** `notebooks/en/03_feature_engineering.ipynb cell cd18 (idx 21)`


### [MINOR] nb03-fig02-correlation.png — caption-mismatch
**Sorun:** Title and caption describe a one-against-target comparison, but the figure is a full 10x10 inter-feature matrix carrying the multicollinearity result.

**Kanıt:** Confirmed. Cell cd18 sets ax.set_title('Pearson Correlation - Numeric Features vs job_runtime') and the caption at 3.dataset_and_workload.tex:219 reads 'Pearson correlation matrix between the numeric attributes and the target variable (job_runtime)'. The rendered figure is a masked lower-triangular 10x10 matrix of all feature-feature pairs. The body text at line 214 explicitly reasons about the feature-feature block ('Strong correlations exist among the sweep-line cluster utilization features ... up to a value of 0.89'), and I confirmed those cells in the figure: cluster_load_cpu/active_job_co...

**Düzeltme:** Retitle to 'Pearson correlation matrix of the numeric features and the target' and amend the caption at line 219 to mention both the first column and the multicollinearity block.

**Konum:** `notebooks/en/03_feature_engineering.ipynb cd18 (idx 21); thesis/latex/chapters/3.dataset_and_workload.tex:219`


---

## NB04 — 34 bulgu (6 kritik)

### [CRITICAL] NB04-Figure01..05.png (and thesis/latex/figures/nb04-fig01..05*.png) — caption-mismatch
**Sorun:** All five exported NB04 PNGs are stale relative to both notebooks; identical stale bytes were copied into thesis/latex/figures.

**Kanıt:** VERIFIED INDEPENDENTLY. stat: exports = 2026-08-31 17:33, notebooks/en/04_runtime_prediction_models.ipynb and notebooks/tr/04_calisma_zamani_tahmin_modelleri.ipynb = 2026-09-06 08:30. md5 of thesis/latex/figures/nb04-fig01-model-comparison.png = 349d9c93c1688ef613aae37c1e265503 = exactly the export (all 5 pairs match byte-for-byte). I extracted the 5 embedded output PNGs from BOTH notebooks; none matches any export md5 (EN cd32 vs export fig01: 111791 B vs 90492 B). Visual diff confirms every specific claim: export fig01 = 13 y-labels, MAE axis 0-15000, Random Forest at ~15,100 s and R2 = -0.6...

**Düzeltme:** Re-run scripts/export_thesis_results.py and its _sync_thesis_figures step. Add a build guard failing when any results/figures/thesis_export/png file is older than the notebook mtime.

**Konum:** `/Users/hasanugurcelebi/Thesis/alibaba-gpu-runtime-prediction-and-scheduling/scripts/export_thesis_results.py (extract_from_nb_dict, line 128; _sync_thesis_figures line 210) + notebooks/en/04_runtime_prediction_models.ipynb cells cd32/ecd01/ecd03/ecd05/ecd07 (indices 93/95/97/99/101)`


### [CRITICAL] NB04-Figure01.png (fig:model-comparison) vs tab:predresults — caption-mismatch
**Sorun:** Figure 1 numerically contradicts Table tab:predresults and the surrounding prose - three mutually inconsistent generations of results in one chapter.

**Kanıt:** VERIFIED. 6.results_and_discussion.tex line 22: 'A & Random Forest (Numeric) & 4,316 & 1,508 & 13,831 & 16.85 & 0.27'. Line 26: 'B & XGBoost (One-Hot) & 3,389 ... & 0.51'. Line 59 prose repeats 'MAE of 3,389 s; RMSE of 11,375 s; R2 of 0.51' and 'Random Forest, R2 = 0.27'. The shipped PNG plots Random Forest at MAE ~15,100 s with R2 = -0.67 and XGB (One-Hot) at ~6,642 s / 0.16. Cross-checked against the exported source table results/figures/thesis_export/html/NB04_Table07.html: row 2 'Random Forest 15236.70 ... -0.67', row 6 'XGB (One-Hot) 6642.39 ... 0.16'. The figure and the table on the faci...

**Düzeltme:** Export df_all to CSV in cd31 and generate tab:predresults with df_all.to_latex() from the same object that feeds cd32, then re-export the figure from that run.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell cd31 (index 91) vs thesis/latex/chapters/6.results_and_discussion.tex lines 20-44 and line 59`


### [CRITICAL] NB04-Figure03.png (fig:pred-vs-actual) — caption-mismatch
**Sorun:** The caption describes a log-log random-sample figure; the shipped PNG is the old linear-axis version.

**Kanıt:** VERIFIED. Caption at 6.results_and_discussion.tex line 99 (not ~104): 'on a random sample of 3,000 test-set jobs (fixed seed) shown on log-log axes'. Body text line 96 area: 'The figure now plots a genuinely random sample of the test set on log-log axes'. The shipped PNG has linear axes with ticks 0/50000/.../300000, suptitle 'Predicted vs Actual Runtime - Tree-Based Models (sample n=3,000)' and axis labels 'Actual Runtime (s)' / 'Predicted Runtime (s)' - no 'log scale'. Cell ecd03 source does contain ax.set_xscale('log'), ax.set_yscale('log'), labels '(s, log scale)' and suptitle '(random sam...

**Düzeltme:** Re-export Figure03 from the current ecd03 output.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell ecd03 (index 97); caption at thesis/latex/chapters/6.results_and_discussion.tex line 99`


### [CRITICAL] mae_spearman_vs_jct_gain_32gpu.png ve _256gpu.png <-> NB04 tablolari — naming-inconsistency
**Sorun:** Ayni model tezin farkli yerlerinde farkli isimlerle geciyor; 'LGBM (Categorical)' etiketi dogrudan YANILTICI (aslinda native-categorical model) ve ayni panelde 'Categorical' kelimesi XGBoost/RF icin one-hot, LGBM icin native anlamina geliyor.

**Kanıt:** MAE degerleriyle birebir esleme yaptim; NB04 hucre cd19 ciktisi ile NB05 rank_correlation_32gpu.csv tam tutuyor: RF (One-Hot) 6271.01 = sekildeki 'RF (Categorical)' 6271.008543; XGB (One-Hot) 6053.96 = 'XGBoost (Categorical)' 6053.956835; LGB (Native Categorical) 4973.16 = 'LGBM (Categorical)' 4973.163012; XGB (Native Categorical) 6600.07 = 'XGBoost (Native Cat)' 6600.065582; Per-User Median (baseline) 5191.50 = 'UserMedian (baseline)' 5191.497019. Kod provenansi da dogruladi: NB05 hucre 27d3cb84'te `preds_lgb_cat = lgb_cat_nat.predict(X_test_cat_base)` ve hucre cd04'te `lgb_cat_nat = joblib.l...

**Düzeltme:** Proje kokune tek kanonik isim sozlugu koy (orn. src/naming.py CANONICAL) ve hem NB04 tablolarinda hem NB05 annotate satirinda uygula; encoding tipini isim icinde acikca tut: 'XGBoost (One-Hot)', 'XGBoost (Native Cat.)', 'LightGBM (Native Cat.)', 'LightGBM (One-Hot)', 'Random Forest (One-Hot)', 'Per-User Median (baseline)'. Tek basina 'Categorical' kelimesini kullanma.

**Konum:** `NB05 hucre 7a843cee / 8d1a4e10 `_RAW_PREDS` sozlugunun anahtarlari; referans isimler /Users/hasanugurcelebi/Thesis/alibaba-gpu-runtime-prediction-and-scheduling/notebooks/en/04_runtime_prediction_models.ipynb hucre id cd19 ve cd31`


### [CRITICAL] NB04-Figure01.png (= thesis figures/nb04-fig01-model-comparison.png) vs Table \ref{tab:predresults} — caption-mismatch
**Sorun:** The figure and the facing table report mutually exclusive numbers for the same models.

**Kanıt:** Opened NB04-Figure01.png visually and NB04_Table07.html numerically; they agree with each other and both contradict the thesis. NB04_Table07.html row 2: 'A — ML Numeric | Numeric | Random Forest | 15236.70 | 13844.99 | 20130.93 | 10564.77 | -0.67'; row 3: 'B — ML Categorical | Categorical | LGB (Native) | 5697.39 | 2683.53 | 13542.68 | 1659.57 | 0.24' (lowest MAE and highest R² of all 19 rows); row 6: 'XGB (One-Hot) | 6642.39 | ... | 0.16'. The figure's R² axis runs -0.7 to ~0.25 and its top bar is LGB (Native) at ~0.24. thesis/latex/chapters/6.results_and_discussion.tex lines 22 and 26 hand-t...

**Düzeltme:** Write df_all from cell#91 to results/analysis/prediction_metrics.csv and generate the LaTeX body from it (`df_all.to_latex`), instead of hand-typing lines 22-44.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell#93 (id=cd32) generates the figure, cell#91 (id=cd31) builds df_all; thesis/latex/chapters/6.results_and_discussion.tex:22-44`


### [CRITICAL] NB04_Table01-09.html, NB05_*_Table02-05.html, NB04-Figure01.png, mae_spearman_vs_jct_gain_*.png, 6.results_and_discussion.tex, appendices.tex — naming-inconsistency
**Sorun:** The same four models carry up to five different display names across the deliverables.

**Kanıt:** Verified name-by-name in the actual files. LightGBM native-cat: 'LGB (Native Categorical)' (NB04_Table02.html, NB04_Table09.html) / 'LGB (Native)' (NB04_Table07.html and NB04-Figure01.png) / 'SJF-LGBM (Categorical)' (NB05_32GPU_Table02.html row 5) / 'LightGBM (Native Cat.)' (6.results_and_discussion.tex:27, appendices.tex:75). XGBoost one-hot: 'XGB (One-Hot)' (Table02/Table07/Figure01) / 'XGB (Categorical)' (Table09) / 'SJF-XGBoost (Categorical)' (NB05) / 'XGBoost (Categorical)' (mae_spearman figure, read off the rendered PNG) / 'XGBoost (One-Hot)' (tex:26). RF one-hot: 'RF (One-Hot)' (Table02...

**Düzeltme:** Define one canonical DISPLAY_NAME map in a single module and import it in notebook 04 cell#91/cell#103 and notebook 05 cell#29/cell#35; regenerate, then delete the tex:117 parenthetical.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell#91 (id=cd31) and cell#103 (id=ecd09); notebooks/en/05_scheduler_evaluation_32_gpu.ipynb cell#29 (id=7a843cee)`


### [MAJOR] NB04-Figure04.png (fig:residuals) — caption-mismatch
**Sorun:** Caption asserts residuals are centered about zero; the figure's own in-plot legend prints Mean = 10601s for the Random Forest panel.

**Kanıt:** VERIFIED. Caption at 6.results_and_discussion.tex line 109 (not ~114): 'Each model has residual values centered about zero.' The shipped PNG's RF panel legend literally reads 'Mean = 10601s' (nearly 3 hours) with the gold mean line visibly offset right of the crimson zero line; XGB = 357s, LGBM = -298s. The notebook's current ecd05 output reads Mean = -1100s / -1053s / -5445s, so the claim is still false for LightGBM (-5445 s = -1.5 h) after the retrain.

**Düzeltme:** Drop the 'centered about zero' claim or state per-model mean/median residuals; annotate the median in ecd05 as well.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell ecd05 (index 99); caption at thesis/latex/chapters/6.results_and_discussion.tex line 109`


### [MAJOR] NB04-Figure01..05 / NB04_Table01-09.html / tab:predresults — naming-inconsistency
**Sorun:** The same model carries up to four different names across NB04 figures and tables.

**Kanıt:** VERIFIED by dumping every exported table to text and reading cd31/ecd01/ecd07 source. Random Forest numeric: 'Random Forest' (Table01 row 2, Table07 row 2, Figure01 y-label) / 'Random Forest (Numeric)' (ecd01+ecd03+ecd05 subplot titles, tab:predresults line 22) / 'RF (Numeric)' (Table09 row 0). RF one-hot: 'RF (One-Hot)' (Table02 row 0, Table07 row 4) / 'RF (Categorical)' (Table09 row 3) / 'Random Forest (One-Hot)' (tab:predresults). XGB one-hot: 'XGB (One-Hot)' (Table02/07) / 'XGB (Categorical)' (Table09 row 4) / 'XGBoost (One-Hot)' (tab:predresults line 26). LightGBM native: 'LGB (Native)' (...

**Düzeltme:** Define one canonical MODEL_NAMES dict at the top of the notebook and route every figure label, tick label and table column through it; regenerate tab:predresults from the same frame.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cells cd31 (index 91), ecd01/ecd03/ecd05 label tuples, ecd07 _DL_RUNS (index 101)`


### [MAJOR] NB04-Figure06.png — other
**Sorun:** NB04-Figure06.png is a byte-identical orphan copy of NB04-Figure05.png that nothing in the current pipeline owns.

**Kanıt:** VERIFIED. md5 NB04-Figure05.png = md5 NB04-Figure06.png = bb71498dadb5758fbce5dda19a1e3955, both 54215 bytes. Figure06 mtime 2026-08-31 15:41 vs 17:33 for Figure01-05. scripts/export_thesis_results.py line 122: EXPECTED_FIGURE_COUNT['NB04'] = 5; THESIS_FIGURE_MAP lines 88-92 define only ('NB04',1)..('NB04',5). _clean_stale_exports (line 192) does unlink PNG_DIR.glob(f'{prefix}-Figure*.png'), so the file predates that guard or the guard has not run since.

**Düzeltme:** Delete results/figures/thesis_export/png/NB04-Figure06.png and re-run the exporter.

**Konum:** `/Users/hasanugurcelebi/Thesis/alibaba-gpu-runtime-prediction-and-scheduling/scripts/export_thesis_results.py lines 118-125 and 192-207`


### [MAJOR] NB04-Figure05.png (and the current ecd07 output) — label-overlap
**Sorun:** The 12 two-line x tick labels physically overlap at rotation=15.

**Kanıt:** VERIFIED by cropping and 3x magnifying the tick band (x 380-900, y 480-560) of the exported PNG. The second line of one label runs into the first line of the next: 'CNN-LSTM' (label 6) visibly touches 'E - Numeric (Sequence)' (label 7); 'LSTM' (label 5) and 'CNN' (label 4) each touch the 'D - Categorical' of their neighbour. Present identically in the notebook's current stored ecd07 output, so it is a code defect. Source confirmed: ax.set_xticklabels(dl_summary['Experiment'] + '\n' + dl_summary['Architecture'], rotation=15, ha='right'). Print math: figsize=(16,6) scaled to ~6.1 in textwidth = ...

**Düzeltme:** Put only the architecture on the tick at rotation=0 and encode Experiment as bar colour + legend; or rotation=45, ha='right', rotation_mode='anchor', fontsize=8 and abbreviate 'E - Numeric (Sequence)'.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell ecd07 (index 101)`


### [MAJOR] NB04-Figure04.png (centre and right panels only - NOT all three) — label-overlap
**Sorun:** The x tick labels of the XGBoost and LightGBM panels collide with zero gap.

**Kanıt:** VERIFIED by cropping and 3x magnifying the tick band of the exported PNG. Centre panel reads as one run-together string: the leading minus of each label is glued to the trailing 0 of the previous one - '-300000-250000-200000-150000-100000-50000  0  50000'. Same in the notebook's current ecd05 output, so it is a code defect not an export artifact. Differing limits confirmed: RF panel spans about -320k..+160k with y to ~4,700 while XGB/LGBM span -320k..+55k with y to ~6,700 and ~6,200. CORRECTION: the claim in the 'figure' header that this affects 'all three panels' is WRONG - I magnified the le...

**Düzeltme:** In ecd05 add ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5)) and a FuncFormatter rendering v/1000 as 'k', relabel the axis '[x1000 s]', and share one x-limit across the three panels.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell ecd05 (index 99)`


### [MAJOR] NB04-Figure02.png (fig:feature-importance) — units
**Sorun:** Three panels share the label 'Importance Score' but plot three different, non-comparable quantities; the LaTeX caption compounds the error by calling all three MDI-based.

**Kanıt:** VERIFIED. Code: ecd01 does importances = model_obj.feature_importances_ and ax.set_xlabel('Importance Score') for all three. grep -rn 'importance_type' over src/ and the notebook returns ZERO hits, so every model uses its library default. Installed versions: xgboost 3.1.1 (XGBRegressor.feature_importances_ default = gain), lightgbm 4.6.0 (LGBMRegressor default = split, a raw count), sklearn RF = MDI. src/tuning.py line 947 confirmed verbatim: lgb.LGBMRegressor(**{**best_params,...}, random_state=..., n_jobs=safe_n_jobs, objective='regression_l1') - no importance_type. The x axes prove it: RF 0...

**Düzeltme:** Normalise all three to gain: lgb_final.booster_.feature_importance(importance_type='gain') / sum, xgb via get_score(importance_type='gain'); relabel 'Normalised gain importance'; change the caption from 'MDI-based' to 'gain-based'; add ax.bar_label(..., fmt='%.3f').

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell ecd01 (index 95); src/tuning.py line 947; caption at thesis/latex/chapters/6.results_and_discussion.tex lines 83 and 88`


### [MAJOR] NB04-Figure05.png — color
**Sorun:** sns.color_palette('muted', 12) cycles a 10-colour palette, so two different experiment/architecture pairs share a colour, and the colour carries no information at all.

**Kanıt:** VERIFIED by pixel sampling the current ecd07 output at y=420. Bar 1 (C-Numeric CNN) fill = RGB(72,120,208) = #4878D0; bar 11 (F-Categorical LSTM) fill = RGB(72,120,208) - IDENTICAL. Bar 2 = RGB(238,133,74) = #EE854A; bar 12 = RGB(238,133,74) - IDENTICAL. #4878D0/#EE854A/#6ACC64 are exactly seaborn's 10-colour 'muted' palette, confirming the cycle at index 10. Source confirmed: color=sns.color_palette('muted', len(dl_summary)) with len = 12. There is no legend in the figure and every bar is already fully identified by its tick label, so the 12-colour ramp is decorative and simultaneously ambigu...

**Düzeltme:** Colour by Experiment (4 groups, 'colorblind' palette) with an explicit legend, which also lets the tick labels shrink to just the architecture name.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell ecd07 (index 101)`


### [MAJOR] NB04-Figure01.png (suptitle 'All Models & Features') — other
**Sorun:** The shipped Figure01 omits five model rows the notebook now produces, including the naive baselines that outperform every trained model, while the suptitle and caption claim completeness.

**Kanıt:** VERIFIED. cd31 source builds 3 + 9 + 3 + 3 + 3 + 3 = 24 rows including 'XGB (Native)', 'Per-User Median', 'ProfileMedian', 'Constant Median' and 'Constant Zero' (24 rows collapse to 18 unique y-labels because CNN/LSTM/CNN-LSTM and their (Sequence) variants appear on both tracks). The shipped PNG has 13 y-labels / 19 bar rows and contains none of those five; NB04_Table07.html likewise has 19 rows. In the notebook's current cd32 output ProfileMedian is the TOP bar at ~4,400 s MAE and Constant Median at ~5,800 s sits ahead of every DL row (all >= ~6,100 s) - i.e. naive baselines beat the trained ...

**Düzeltme:** Re-export Figure01 from the current cd31/cd32 run; add a 'Kind' column tagging baselines and draw axes[0].axvline at the best baseline MAE, or hatch the baseline bars.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cells cd31 (index 91) and cd32 (index 93); caption at thesis/latex/chapters/6.results_and_discussion.tex line 53`


### [MAJOR] NB04-Figure01..05.png (all) — dpi-quality
**Sorun:** Font sizes and resolution are far below journal minimums at final print size; no vector version is produced.

**Kanıt:** VERIFIED. thesis.cls line 39 \LoadClass[a4paper,12pt]{report}, line 42 \RequirePackage[a4paper]{geometry}; the \@dtype branch at line 97 sets left=3.5cm right=2cm => textwidth 15.5 cm = 6.10 in (the else branch at line 109 gives 16 cm = 6.30 in; either way the conclusion holds). All five figures are \includegraphics[width=\textwidth] (6.results_and_discussion.tex lines 52, 74, 87, 98, 108). Authored sizes confirmed in source: cd32 figsize=(18,8), ecd01 (18,6), ecd03 (18,6), ecd05 (18,5), ecd07 (16,6) => scale 0.34 (0.38 for fig05). cd02 rcParams set axes.titlesize 14, axes.labelsize 12 and NO ...

**Düzeltme:** Author at print size (figsize about (6.1,3.0) for the 1x3 panels), keep fonts at 8-9 pt, set savefig.dpi 600 in cd02, remove the hard-coded fontsize=8, and have the exporter emit a PDF alongside each PNG with \includegraphics switched to it.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell cd02 (index 5) and figsize calls in cd32/ecd01/ecd03/ecd05/ecd07; thesis/latex/thesis.cls lines 39-42 and 95-106`


### [MAJOR] NB04_Table01..09.html (all exported NB04 tables, and NB05 tables) — other
**Sorun:** Every exported HTML table is UTF-8 but carries no charset declaration.

**Kanıt:** VERIFIED at byte level. grep -c -i charset returns 0 for all nine NB04_Table0*.html. head of NB04_Table07.html: '<html><head><style>table{border-collapse:collapse;...}</style></head><body><div>' - no <meta charset>. xxd of the R-squared header: '52 c2 b2 3c 2f 74 68 3e' = 'R' + UTF-8 U+00B2 + '</th>'. xxd of the Experiment label: '41 20 e2 80 94 20 4d 4c' = 'A ' + UTF-8 em dash + ' ML'. Without a declaration a local file falls back to windows-1252, rendering 'RÂ²' and 'A â ML'. Source confirmed: scripts/export_thesis_results.py HTML_STYLE_HEADER at lines 38-43 emits the head with no meta tag...

**Düzeltme:** Change HTML_STYLE_HEADER to '<html><head><meta charset="utf-8"><style>' and re-run the exporter.

**Konum:** `/Users/hasanugurcelebi/Thesis/alibaba-gpu-runtime-prediction-and-scheduling/scripts/export_thesis_results.py lines 38-43`


### [MAJOR] NB04_Table07.html vs tab:predresults (thesis Table 6.1) — units
**Sorun:** The MAPE column is on two incompatible scales between the exported source table and the thesis table; the median-absolute-error abbreviation also differs.

**Kanıt:** VERIFIED. NB04_Table07.html: LightGBM (Numeric) MAPE = 2862.46, RF (One-Hot) = 2072.39, Random Forest (Numeric) = 10564.77 - header is bare 'MAPE' with no unit (cd31 uses the key 'MAPE'). tab:predresults header at line 20 is 'MAPE (\%)' with LightGBM (Numeric) 20.66 and Random Forest (One-Hot) 15.57 - roughly two orders of magnitude apart with no stated conversion. src/models/evaluation.py lines 79-84 confirm the helper already multiplies by 100.0 and its own docstring (lines 52-62) warns the LaTeX table 'is a separate, hand-transcribed artefact that can drift out of sync with this code', so t...

**Düzeltme:** Rename the df_all column to 'MAPE (%)' so cd31, the exported table and LaTeX agree; standardise on 'MdAE (s)'; regenerate tab:predresults from df_all instead of hand-typing it.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell cd31 (index 91) and src/models/evaluation.py lines 73-91 vs thesis/latex/chapters/6.results_and_discussion.tex line 20`


### [MAJOR] NB04_Table08.html — units
**Sorun:** Column headers are the raw internal metric keys, lowercase and unitless.

**Kanıt:** Parsed the <th> elements of NB04_Table08.html: ['Experiment', 'Architecture', 'mae', 'mdae', 'rmse', 'mape', 'r2']. By contrast NB04_Table07.html heads the identical quantities ['MAE (s)', 'MdAE (s)', 'RMSE (s)', 'MAPE', 'R²'] and the thesis uses MAE (s) / MedAE (s). Source confirmed in cell#101 (id=ecd07): `_CORE = ["mae", "mdae", "rmse", "mape", "r2"]` and `_row[_k] = round(float(_metrics[_k]), 2)` — the keys become the column labels verbatim, and `_cols` is assembled from the same list before `display(dl_summary[_cols])`.

**Düzeltme:** Rename before display: {'mae':'MAE (s)','mdae':'MedAE (s)','rmse':'RMSE (s)','mape':'MAPE (%)','r2':'R²'}, applied to the _std/_seed0 variants too.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell#101 (id=ecd07), `_CORE` list used as column labels`


### [MAJOR] NB04_Table01-06.html vs NB04_Table07.html vs Table \ref{tab:predresults} — units
**Sorun:** Three different header conventions for the same five metrics, including two spellings of median-absolute-error.

**Kanıt:** Parsed headers directly. NB04_Table01.html and NB04_Table02.html: ['Model', 'Test MAE', 'Test MdAE', 'Test RMSE', 'Test MAPE', 'Test R²'] — no units at all, while the values are seconds in the thousands (e.g. LightGBM 7076.86). NB04_Table07.html: ['MAE (s)', 'MdAE (s)', 'RMSE (s)', 'MAPE', 'R²'] — MAPE left unitless. 6.results_and_discussion.tex:20: '\\textbf{MAE (s)} & \\textbf{MedAE (s)} & \\textbf{RMSE (s)} & \\textbf{MAPE (\\%)} & \\textbf{R\\textsuperscript{2}}' — third spelling 'MedAE' vs the notebooks' 'MdAE'. appendices.tex:70 uses 'MedAE (s)' as well. Source confirmed: cell#91 (id=cd3...

**Düzeltme:** Standardise on MAE (s) / MedAE (s) / RMSE (s) / MAPE (%) / R² and apply the same rename to the per-experiment frames and to cell#91.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell#91 (id=cd31), the `"MdAE (s)"` keys`


### [MAJOR] NB04_Table09.html — truncation
**Sorun:** Hyperparameter cells are truncated mid-JSON by pandas' display width, though not in every row as claimed.

**Kanıt:** Counted directly in the file: 19 rows contain a `<td>{`-style Params cell and 11 of them end in `...</td>`, e.g. `{'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': 20, 'max_features': 0.7, 'max_sampl...` and `{'num_filters': 138, 'kernel_size': 1, 'lstm_hidden_size': 256, 'lstm_num_layers': 2, 'learning_rate': 0.0008, 'batc...`. The claim 'All 19 rows are cut' is WRONG — the 8 short DL rows fit in full, e.g. `{'num_filters': 256, 'kernel_size': 1, 'learning_rate': 0.001, 'batch_size': 246}`. The defect stands for the 11 tree-model and hybrid rows, which are preci...

**Düzeltme:** Explode Params into columns via pd.json_normalize, or set display.max_colwidth to None, or emit one table per model family.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell#103 (id=ecd09), `pd.set_option("display.max_colwidth", 120)``


### [MAJOR] NB04-Figure05.png (= thesis figures/nb04-fig05-dl-comparison.png) — label-overlap
**Sorun:** Two-line x tick labels rotated only 15 degrees run into each other across all twelve bars.

**Kanıt:** Opened the PNG. The tick labels are two-line strings ('D — Categorical' over 'CNN-LSTM') at a shallow rotation, and adjacent labels visibly interleave — the experiment line of one label sits under/against the architecture line of its neighbour across the whole axis, so bar-to-label attribution is ambiguous in the middle of the row. Source confirmed verbatim in cell#101 (id=ecd07): `ax.set_xticklabels(dl_summary["Experiment"] + "\\n" + dl_summary["Architecture"], rotation=15, ha="right")`. Aggravated by the downscaling issue below (authored 15.84 in wide, included at \\textwidth).

**Düzeltme:** Rotate to 90 with ha='center', or group the bars into four experiment clusters and put only the architecture on the tick label.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell#101 (id=ecd07)`


### [MAJOR] NB04-Figure05.png (= thesis figures/nb04-fig05-dl-comparison.png) — color
**Sorun:** Twelve bars get twelve colours that encode nothing, and the 10-colour palette wraps so two pairs repeat.

**Kanıt:** Confirmed numerically by sampling the rendered PNG at y=400 and extracting the 12 bar colour runs: [(72,120,208), (238,133,74), (106,204,100), (214,95,95), (149,108,180), (140,97,60), (220,126,192), (121,121,121), (213,187,103), (130,198,226), (72,120,208), (238,133,74)]. Bar 11 is byte-identical to bar 1 and bar 12 to bar 2 — the 'muted' palette's 10 colours wrap. There is no legend and no colour dimension in the data. Minor correction to the finding's attribution: the repeats are bar 11 = F/LSTM (matching bar 1 = C/CNN) and bar 12 = F/CNN-LSTM (matching bar 2 = C/LSTM), not 'bar 11 (F/CNN)'....

**Düzeltme:** Encode the feature track with two colours plus a legend, or use a single neutral fill.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell#101 (id=ecd07): `color=sns.color_palette("muted", len(dl_summary))``


### [MAJOR] nb05-fig01..05_{32,256}gpu.png and nb04-fig01 as placed in 6.results_and_discussion.tex — dpi-quality
**Sorun:** Figures are authored 16-18 inches wide and included at a fraction of that, shrinking label text below legibility.

**Kanıt:** Measured every file with PIL. thesis/latex/figures/nb05-fig05-wait-percentile_32gpu.png = 1589x590 px at 99.9998 dpi = 15.89 in authored width; nb05-fig01-scheduler-jct_32gpu.png = 1792x691 px = 17.92 in; nb04-fig05-dl-comparison.png = 1584x584 = 15.84 in; nb04-fig01-model-comparison.png = 1784x818 = 17.84 in; mae_spearman_vs_jct_gain_32gpu.png = 2378x1034 at 150 dpi = 15.85 in. Include widths verified by grep in 6.results_and_discussion.tex: lines 190, 192, 202, 204, 217, 219, 226, 228, 374, 376 all use `width=0.48\\textwidth`; nb04-fig01 at line 52 and nb04-fig05 at line 74 use `width=\\text...

**Düzeltme:** Author at final printed size: figsize=(6.5, 4.0) for the paired NB05 figures and (6.5, 3.2) for nb04-fig01, set rcParams font sizes to 8-9 pt, and save at dpi=300.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb cells #29, #35, #54; notebooks/en/04_runtime_prediction_models.ipynb cells #93, #101`


### [MAJOR] NB04-Figure01.png (= thesis figures/nb04-fig01-model-comparison.png) — axis
**Sorun:** A single outlier controls both panels' scales, and the two panels are sorted by different keys.

**Kanıt:** Opened the PNG. The MAE panel's x-axis runs 0 to ~15,000 because Random Forest (numeric) sits at 15,236.70 (NB04_Table07.html row 2), while the other twelve model bars fall between 5,697 and 7,456 — all compressed into the left ~half of the panel with visually indistinguishable lengths. The R² panel spans about -0.7 to 0.25 because of the same model at -0.67, so the 0.00-0.24 band where every other model lives occupies the right quarter. Different sorting confirmed both visually (left panel top row 'LGB (Native)' then 'CNN-LSTM'; right panel top row 'LGB (Native)' then 'XGB (One-Hot)') and in ...

**Düzeltme:** Use a log x-scale (or broken axis) for MAE, and pass the same `order=` to both sns.barplot calls so rows line up.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell#93 (id=cd32)`


### [MAJOR] NB04-Figure01.png, Fig. \ref{fig:model-comparison} prose — caption-mismatch
**Sorun:** The prose generalises Experiments E and F into a negative-R2 claim the plotted data contradicts.

**Kanıt:** 6.results_and_discussion.tex:11 reads 'Temporal or sequence-based models from Experiments E and F yielded negative R\\textsuperscript{2} values, which indicate their predictions were worse than a simple mean baseline.' NB04_Table07.html shows all three Experiment F models POSITIVE: row16 'F — DL Categorical (Sequence) | LSTM (Sequence) | 0.14', row17 'CNN (Sequence) | 0.05', row18 'CNN-LSTM (Sequence) | 0.04'. Only Experiment E is negative (rows 13-15: -0.07, -0.01, -0.01). The figure's R² panel confirms this visually (LSTM (Sequence) categorical is a positive orange bar at ~0.14). The thesis'...

**Düzeltme:** Rewrite line 11 to name Experiment E only and state the Experiment F split explicitly from NB04_Table07.html.

**Konum:** `thesis/latex/chapters/6.results_and_discussion.tex:11`


### [MINOR] NB04-Figure01.png — legend-error
**Sorun:** Duplicated 'Track' legend at inconsistent positions, hue that silently encodes the experiment for six duplicated y-labels, and two panels sorted differently.

**Kanıt:** VERIFIED on the shipped PNG and in cd32 source. (1) The identical two-entry 'Track' legend appears twice - upper right in the MAE panel, upper left in the R2 panel; cd32 makes two sns.barplot(..., hue='Track') calls and never removes either legend. (2) Six y-labels carry two bars each ('CNN', 'LSTM', 'CNN-LSTM', 'CNN (Sequence)', 'LSTM (Sequence)', 'CNN-LSTM (Sequence)'), so the orange/blue hue is the only cue separating Experiment C from D and E from F, and the legend says only 'Numeric'/'Categorical' - nothing tells the reader the orange 'CNN' bar is Experiment D. (3) cd32 sorts the left pan...

**Düzeltme:** Remove the per-axes legends and place one fig.legend; rename the hue to 'Feature set'; include the experiment letter in the Model string so every y label is unique; pass a single shared order= to both barplot calls.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell cd32 (index 93)`


### [MINOR] NB04-Figure01.png vs NB04-Figure02/03/04.png — color
**Sorun:** steelblue and darkorange carry different meanings in adjacent figures of the same chapter.

**Kanıt:** VERIFIED in source and in the images. cd32: palette = {'Numeric': 'steelblue', 'Categorical': 'darkorange'} - blue = numeric feature track, orange = categorical track. ecd01/ecd03/ecd05 all use the tuples ('Random Forest (Numeric)', 'steelblue'), ('XGBoost (Numeric)', 'darkorange'), ('LightGBM (Numeric)', 'seagreen') - so the same steelblue means Random Forest and the same darkorange means XGBoost. All three of those are numeric-track models, and Figure01 indeed draws Random Forest, XGBoost and LightGBM as blue bars. A reader who learns 'orange = categorical' from Figure01 (page 1) meets 'oran...

**Düzeltme:** Reserve one palette per semantic axis: keep steelblue/darkorange/seagreen for the three algorithms and give Figure01's track hue a distinct colour-blind-safe pair plus a redundant hatch.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell cd32 (index 93) palette dict vs cells ecd01 (95) / ecd03 (97) / ecd05 (99) colour tuples`


### [MINOR] NB04-Figure04.png — color
**Sorun:** The mean-residual marker is gold on a white/orange ground - very low contrast, and in the XGBoost panel it completely overplots the zero line.

**Kanıt:** PARTIALLY VERIFIED, with one sub-claim corrected. CONFIRMED: ecd05 draws ax.axvline(residuals.mean(), color='gold', linestyle='-', lw=1.5). Pixel sampling the XGB panel at x=1094-1095 gives exactly RGB(255,215,0) with RGB(255,255,255) white immediately left and RGB(255,140,0) darkorange histogram bars immediately right - gold-on-white is about 1.4:1 contrast and gold-on-darkorange is worse; at the 0.34 print scale the 1.5 pt line is ~0.5 pt. Also confirmed: scanning the whole centre plot area for crimson pixels finds them ONLY in the legend swatch (x 689-711), never as a vertical line - the cr...

**Düzeltme:** In ecd05 use color='black', linestyle='-', lw=2.0 for the mean and color='0.3', linestyle='--', lw=1.0 for zero, so weight plus dash pattern carry the distinction at high contrast.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell ecd05 (index 99)`


### [MINOR] NB04-Figure02.png (fig:feature-importance) — caption-mismatch
**Sorun:** The body text draws a three-way contrast between the panels that the figure does not support.

**Kanıt:** VERIFIED. 6.results_and_discussion.tex line 83 (not ~85) claims 'Random Forest's emphasis on arrival times, XGBoost's focus on resource requests, and LightGBM's reliance on global metrics regarding system-wide load'. In the shipped Figure02 the number-one feature is arrival_sec for Random Forest AND arrival_sec for LightGBM (only XGBoost differs, gpu_demand first) - the RF and LightGBM panels share their top feature, so no contrast exists to describe between them. In the notebook's current ecd01 output the ranking changes again: gpu_demand is first for both Random Forest and XGBoost, arrival_s...

**Düzeltme:** Regenerate the figure first, then rewrite the sentence from the actual top-3 of each panel; extend the ecd01 print loop from argmax to np.argsort(...)[::-1][:3].

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell ecd01 (index 95); text at thesis/latex/chapters/6.results_and_discussion.tex line 83`


### [MINOR] NB04-Figure01.png (baseline model labels) — naming-inconsistency
**Sorun:** The four naive baselines are named in inconsistent styles in cd31 and the inconsistency is already visible on the Figure01 y axis in the notebook's current output.

**Kanıt:** VERIFIED verbatim in cd31 source: "Model": "Per-User Median", "Model": "ProfileMedian", "Model": "Constant Median", "Model": "Constant Zero" - 'ProfileMedian' is CamelCase with no space while its three siblings are spaced Title Case. Both appear side by side on the y axis of the notebook's current cd32 output (ProfileMedian is the top bar, Per-User Median the third). Also confirmed: 'Per-User Median' is loaded from the checkpoint key 'exp_b_user_median', a third spelling.

**Düzeltme:** Rename all four to one style in cd31 (e.g. 'Profile Median (baseline)') and add a 'Kind' column set to 'Baseline' so cd32 can style them as a group rather than matching on the name string.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell cd31 (index 91)`


### [MINOR] NB04_Table07.html vs NB05_32GPU_Table02.html vs Table \ref{tab:predresults} — units
**Sorun:** MAPE is labelled inconsistently across sources; the thesis value also contradicts the pipeline value. But the 'scale is undocumented / one of the three must be wrong' reasoning is incorrect.

**Kanıt:** Value trace confirmed: NB04_Table07.html row 0 gives LightGBM MAPE = 2862.46 under a header of plain 'MAPE'; NB05_32GPU_Table02.html row 7 (SJF-LGBM (Numeric)) carries the identical 2862.46 under the header 'Model MAPE (%)'; 6.results_and_discussion.tex:24 states LightGBM (Numeric) MAPE = 20.66. So the missing '(%)' in the NB04 headers and the thesis/pipeline value gap are both real. HOWEVER the finding's core reasoning is wrong: the scale IS documented. src/models/evaluation.py:52-62 states MAPE is 'expressed on a 0-100 percentage scale (a value of 50.0 means 50%, not 0.5)' and warns it 'rout...

**Düzeltme:** Add '(%)' to the MAPE header in NB04_Table07/Table08 and re-derive the thesis MAPE column from the pipeline.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell#91 (id=cd31) `"MAPE"` keys; src/models/evaluation.py:52-93 (already correct)`


### [MINOR] NB04-Figure05.png / NB04-Figure06.png — other
**Sorun:** NB04-Figure06.png is a byte-identical stale leftover. But the accompanying claim that this blocks NB04's thesis sync is false.

**Kanıt:** md5 confirms NB04-Figure05.png == NB04-Figure06.png == thesis/latex/figures/nb04-fig05-dl-comparison.png (all bb71498dadb5758fbce5dda19a1e3955), with mtimes Aug 31 17:33 and Aug 31 15:41 respectively — so Figure06 is a leftover from an earlier run. THE CAUSAL CLAIM IS FALSE: I counted the notebook's actual outputs programmatically and 04_runtime_prediction_models.ipynb emits exactly 5 image/png outputs (cells 93, 95, 97, 99, 101), matching EXPECTED_FIGURE_COUNT['NB04'] = 5. And the sync demonstrably did NOT fail: thesis/latex/figures/nb04-fig01..05 all carry mtime Aug 31 17:33 (the export run)...

**Düzeltme:** Delete results/figures/thesis_export/png/NB04-Figure06.png. No count reconciliation is needed.

**Konum:** `results/figures/thesis_export/png/NB04-Figure06.png; scripts/export_thesis_results.py:192-207, :322`


### [MINOR] NB04_Table01-06.html — other
**Sorun:** Six per-experiment tables are strict subsets of Table07, carry the pandas index, and two of them are indistinguishable by content.

**Kanıt:** Parsed all three of Table01, Table03 and Table05. Table01 rows: 'LightGBM 7076.86 ...', 'XGBoost 7455.69 ...', 'Random Forest 15236.70 ...' — identical values to NB04_Table07.html rows 0-2, which additionally carry Experiment and Track columns. Table03 rows: 'CNN | 6480.78 | 3400.27 | 15628.44 | 2236.84 | -0.01', 'CNN-LSTM | 6697.56 | ...', 'LSTM | 7185.50 | ...'. Table05 rows: 'CNN | 6170.53 | ...', 'CNN-LSTM | 6791.17 | ...', 'LSTM | 6957.51 | ...'. Same three model names with no '(Sequence)' suffix and no experiment label in either — nothing in Table05 identifies it as Experiment E. All six...

**Düzeltme:** Export only the consolidated Table07, or add index=False, an Experiment column, and '(Sequence)' suffixes so Tables 03-06 are self-identifying.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb, the per-experiment `display(...)` calls exported as NB04_Table01-06`


### [MINOR] mae_spearman_vs_jct_gain_*.png vs NB04_Table07.html vs 6.results_and_discussion.tex — naming-inconsistency
**Sorun:** The non-learned baselines and the native-categorical XGBoost carry four different names, and XGBoost (Native) appears in no thesis table.

**Kanıt:** Verified in source and in the rendered figure. cell#91 (id=cd31) defines `"Model": "Per-User Median"` and `"Model": "ProfileMedian"` in adjacent lines — one spaced-and-hyphenated, one camel-case — plus 'Constant Median' and 'Constant Zero'. cell#29 (id=7a843cee) `_RAW_PREDS` keys are 'SJF-UserMedian (baseline)' and 'SJF-ProfileMedian (baseline)'. The rendered mae_spearman_vs_jct_gain_32gpu.png prints 'UserMedian (baseline)' and 'ProfileMedian (baseline)' (the code strips the 'SJF-' prefix). 6.results_and_discussion.tex:184 footnote says 'non-learned reference baselines (Per-User Median, Profil...

**Düzeltme:** Add the baselines and XGBoost-native to the single canonical name map, regenerate, and add the missing rows to tab:expb-full.

**Konum:** `notebooks/en/04_runtime_prediction_models.ipynb cell#91 (id=cd31); notebooks/en/05_scheduler_evaluation_32_gpu.ipynb cell#29 (id=7a843cee); thesis/latex/chapters/appendices.tex:70-78`


---

## NB05_32GPU — 26 bulgu (8 kritik)

### [CRITICAL] NB05_32GPU-Figure03.png / thesis: nb05-fig02-wait-cdf_32gpu.png — legend-error
**Sorun:** The grey-bundle legend count is a hardcoded string literal, not computed, and is wrong in the current notebook state.

**Kanıt:** Read the cell source: the literal `ax.plot([], [], lw=0.9, color="0.75", label="other ML policies (17)")` is verbatim in cell 38 - no variable, no f-string. Shipped PNG has 21 policies (counted 21 bars in nb05-fig01 and 21 boxes in nb05-fig03 from the same run), 4 highlighted, so 17 grey happens to be correct there. Cell 26 POLICIES list now contains 28 entries (counted: FIFO, SRF, SJF-Oracle + 25 SJF-Pred, with the comment 'Total: 27 or 28 policies'). I extracted the notebook's own stored render of cell 35 (same run as cell 38) and counted 28 bars, including SJF-AlibabaEstimate/-Group/-GroupG...

**Düzeltme:** Compute the count instead of hardcoding it, and drop 'ML': `_n_other = sum(1 for p in all_results['policy'].unique() if p not in _highlight)` then `label=f'other policies (n={_n_other})'`.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb, cell index 38 (id=e5cd01)`


### [CRITICAL] NB05_32GPU-Figure04.png / thesis: nb05-fig03-slowdown-box_32gpu.png — label-overlap
**Sorun:** X tick labels collide and are illegible.

**Kanıt:** Cropped the shipped PNG at 3x. Crop of x=580-1000: 'SJF-CNN-LSTM (Categorical Sequence)' is printed straight through 'SJF-LSTM (Numeric)' - the glyphs interleave into unreadable soup. Crop of x=1150-1584: 'FIFO' is overprinted directly on the 'Sequence)' portion of 'SJF-CNN-LSTM (Numeric Sequence)'. Both named collisions are exactly as reported. Cause confirmed in source: `ax.tick_params(axis='x', rotation=35)` rotates without setting ha, so labels stay centred on their tick. I also viewed the notebook's stored 28-policy render of the same cell: it is measurably worse - 'SJF-LGBM (No Cluster-L...

**Düzeltme:** Set `ax.set_xticklabels(policy_order, rotation=40, ha='right', rotation_mode='anchor', fontsize=8)`, or better, flip to a horizontal boxplot (y='policy', x='slowdown', log x) so 28 long names read left-to-right.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb, cell index 41 (id=e5cd03)`


### [CRITICAL] nb05-fig01-scheduler-jct_32gpu.png (= NB05_32GPU-Figure02.png) vs Table tab:schedresults — caption-mismatch
**Sorun:** The figure and the 32-GPU block of Table 6.2 report irreconcilable numbers for the same experiment, including a sign flip and a ranking reversal.

**Kanıt:** Read both. Every numeric sub-claim checks out. Table line 128: SJF-Oracle 92,064 s / 81.59 %; figure bar label: 252024s / 81.3 %. Table: FIFO 499,951 s; figure: 1344966s (2.7x). SIGN FLIP confirmed: table line 146 SJF-LSTM (Numeric Sequence) -45.52 %, figure right panel +10.3 %. SJF-LSTM (Categorical Sequence): table 17.87 %, figure 57.6 %. Ranking reversal confirmed: table puts SJF-XGBoost (Categorical) 56.25 % above SJF-LSTM (Categorical) 55.21 %; the figure puts SJF-LSTM (Categorical) 59.8 % above SJF-XGBoost (Categorical) 59.6 %; and the prose at line ~184 states 'the categorical LSTM poli...

**Düzeltme:** Emit Table 6.2's 32-GPU block from the same `eval_sorted` DataFrame that feeds the figure, in the same cell (`.to_latex(PROJECT_ROOT/'thesis/latex/tables/schedresults_32gpu.tex')`), and `\input{}` it, so the two cannot drift.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb cell 35 (id=cd21); table hand-written in thesis/latex/chapters/6.results_and_discussion.tex lines 118-148`


### [CRITICAL] mae_spearman_vs_jct_gain_32gpu.png (= NB05_32GPU-Figure01.png) — label-overlap
**Sorun:** All 21 points are annotated with a fixed (4,4) offset and nothing repels them, so dense clusters become unreadable stacks.

**Kanıt:** Confirmed in source: `ax.annotate(..., xytext=(4, 4), textcoords='offset points', fontsize=7)` inside an unconditional loop over every row, no repulsion. Cropped thesis/latex/figures/mae_spearman_vs_jct_gain_32gpu.png at 3x (region x=550-1000, y=90-300): 'XGBoost (Categorical)' and 'XGBoost (Native Cat)' are printed over each other, rendering as 'XGBoost (CategoricalXGBoost (Native Cat)'; 'CNN-LSTM (Categorical Sequence)', 'CNN (Numeric)' and 'LSTM (Numeric)' form a three-way pile. Right panel (from full-figure read): 'XGBoost (Native Cat)' overprints 'XGBoost (Categorical)', and 'CNN-LSTM (Ca...

**Düzeltme:** Repel with adjustText (`adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', lw=0.4, color='0.5'))`), or annotate only the extremes and reference baselines and colour-code the model families with a legend.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb, cell index 29 (id=7a843cee), the `for _, row in rank_df.iterrows(): ax.annotate(...)` loop`


### [CRITICAL] NB05_32GPU_Table01.html, NB05_256GPU_Table01.html — other
**Sorun:** Both files are a raw `sim_jobs.head(3)` data preview, not a publishable table.

**Kanıt:** Read both files in full. NB05_32GPU_Table01.html is a bare `.dataframe` HTML repr with 34 snake_case columns + unlabelled index: job_id, arrival_time, arrival_sec, job_runtime, gpu_demand, user, gpu_type, num_inst, num_cpu, hour_of_day, day_of_week, cluster_load_cpu, cluster_load_gpu, active_job_count, pred_rf_num, pred_xgb_num, pred_lgb_num, pred_rf_cat, pred_xgb_cat, pred_lgb_cat, pred_cnn, pred_lstm, pred_cnn_lstm, pred_cnn_cat, pred_lstm_cat, pred_cnn_lstm_cat, pred_cnn_num_seq, pred_lstm_num_seq, pred_hybrid_num_seq, pred_cnn_cat_seq, pred_lstm_cat_seq, pred_hybrid_cat_seq, submit_time, r...

**Düzeltme:** Delete the trailing `sim_jobs.head(3)` expression in cell#20 (use `print(sim_jobs.shape)`), or gate the exporter's `if "<table" in html_content:` branch on a curated cell-metadata tag.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb cell#20 (id=cd13), last line `sim_jobs.head(3)`; scripts/export_thesis_results.py:180-186`


### [CRITICAL] NB05_32GPU-Figure02.png / NB05_32GPU_Table04.html vs Table \ref{tab:wilcoxon} — caption-mismatch
**Sorun:** The thesis names the wrong two failing policies at both cluster scales.

**Kanıt:** Parsed NB05_32GPU_Table04.html: the only two rows flagged '❌ No' with p-value 1.000000 are row18 'SJF-CNN (Numeric Sequence)' (JCT vs FIFO -174445.5) and row19 'SJF-CNN-LSTM (Numeric Sequence)' (-201224.8); row17 'SJF-LSTM (Numeric Sequence)' is '✅ Yes (p<0.05)' at +138071.1. The CSS confirms only rows 18-19 get `color: red`. NB05_256GPU_Table04.html: row18 'SJF-CNN-LSTM (Numeric Sequence)' (-14252.2) and row19 'SJF-CNN (Numeric Sequence)' (-39732.3) fail; row17 SJF-LSTM (Numeric Sequence) is significant at +22952.9. NB05_32GPU_Table02.html gives -12.97% (CNN Num Seq) / -14.96% (CNN-LSTM Num S...

**Düzeltme:** Rewrite lines 243, 248, 174-175, 275-276 and 298-299 from NB05_*_Table04.html, and emit the failing-policy list from the notebook as a generated \\newcommand include so prose cannot drift.

**Konum:** `thesis/latex/chapters/6.results_and_discussion.tex:174-175,243,248,275-276,298-299`


### [CRITICAL] NB05_32GPU-Figure02.png, NB05_256GPU-Figure02.png (= thesis figures/nb05-fig01-scheduler-jct_{32,256}gpu.png) — label-overlap
**Sorun:** The two panels of the side-by-side figure have different row sets, so horizontal bands do not correspond.

**Kanıt:** Opened NB05_32GPU-Figure02.png visually. Left panel has 21 rows top-to-bottom: SJF-CNN-LSTM (Numeric Sequence), SJF-CNN (Numeric Sequence), FIFO, SJF-LSTM (Numeric Sequence), ... Right panel has 20 rows: SJF-CNN-LSTM (Numeric Sequence), SJF-CNN (Numeric Sequence), SJF-LSTM (Numeric Sequence), ... At the third row the left panel reads 'FIFO' and the right reads 'SJF-LSTM (Numeric Sequence)', and every row below is offset by one — exactly as reported. Source confirmed in cell#35 (id=cd21): `plot_df2 = plot_df[plot_df["Policy / Architecture"] != "FIFO"].copy()` filters FIFO out of the right panel...

**Düzeltme:** Keep FIFO in both panels (`plot_df2 = plot_df.copy()`, its improvement is 0.0% and is a meaningful reference), add `axes[1].sharey(axes[0])` and drop the duplicated tick labels.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb cell#35 (id=cd21)`


### [CRITICAL] mae_spearman_vs_jct_gain_32gpu.png, mae_spearman_vs_jct_gain_256gpu.png, NB05_32GPU-Figure01.png, NB05_256GPU-Figure01.png — label-overlap
**Sorun:** Point labels are drawn with a fixed offset and no collision handling; dense clusters are overprinted and illegible.

**Kanıt:** Opened thesis/latex/figures/mae_spearman_vs_jct_gain_32gpu.png and cropped/zoomed the top-left of the left panel. Verified overprints: 'XGBoost (Categorical)' and 'XGBoost (Native Cat)' render as the merged string 'XGBoost (CategoricalXGBoost (Native Cat)'; 'CNN-LSTM (Categorical Sequence)', 'CNN (Numeric)' and 'LSTM (Numeric)' collapse into an unreadable smear ('CNN-LSTM (Categorical SeqCNN…(Numeric)' overlaid with 'LSTM (Numeric)'); at the left edge 'ProfileMedian (baseline)' runs directly into 'LGBM (Categorical)'. In the right panel 'XGBoost (Native Cat)' sits inside 'XGBoost (Categorical)...

**Düzeltme:** Use adjustText, or label only the policies the prose discusses and give the rest a marker-shape/track legend.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb cell#29 (id=7a843cee), the annotate loop`


### [MAJOR] mae_spearman_vs_jct_gain_32gpu.png (= NB05_32GPU-Figure01.png) — axis
**Sorun:** The right panel carries the full y-axis label but has no y tick labels, only bare tick marks.

**Kanıt:** Cropped the right panel's y-axis strip (x=1210-1500) at 2x: the rotated label '32-GPU JCT Improvement over FIFO (%)' is fully drawn, and the left spine shows tick marks at every gridline with no numeral beside any of them. Cause confirmed in source: `sharey=True` suppresses tick labels on axes[1], while `ax.set_ylabel(f'{N_GPU}-GPU JCT Improvement over FIFO (%)')` sits inside the `for ax, x_col, xlabel, title in [...]` loop and so runs for both axes. Same defect present in the older NB05_32GPU-Figure01.png copy.

**Düzeltme:** Move the ylabel out of the loop and call `axes[0].set_ylabel(...)` once, or keep both labels and re-enable ticks with `axes[1].tick_params(labelleft=True)`.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb, cell index 29 (id=7a843cee): `plt.subplots(1, 2, figsize=(16, 7), sharey=True)` plus `ax.set_ylabel(...)` inside the loop`


### [MAJOR] NB05_32GPU-Figure02.png / thesis: nb05-fig01-scheduler-jct_32gpu.png — truncation
**Sorun:** The two longest bar value labels are drawn outside the left panel's axes frame, into the inter-panel gutter.

**Kanıt:** Detected the axes spines by column-wise dark-pixel density on the 1792x691 PNG: strong vertical lines at x = 285, 862, 1190, 1380, 1767, so axes[0] spans x=285-862. Cropped x=700-1260 at 3x: the right spine is clearly visible and '1546191s' straddles it ('1546' inside the axes, '191s' outside), as does '1519412s' ('1519' inside, '412s' outside). The third label '1344966s' stops short of the spine, so it is precisely the two longest labels, exactly as reported. Cause confirmed in source: `axes[0].text(bar.get_width() + 0.01 * plot_df['Mean JCT (s)'].max(), ...)` places text past the data limit ...

**Düzeltme:** `axes[0].set_xlim(0, plot_df['Mean JCT (s)'].max() * 1.18)`, or use `axes[0].bar_label(bars1, fmt='%.0fs', padding=3, fontsize=8)` followed by `axes[0].margins(x=0.15)`.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb, cell index 35 (id=cd21), the `for bar in bars1: axes[0].text(...)` loop`


### [MAJOR] NB05_32GPU-Figure02.png / thesis: nb05-fig01-scheduler-jct_32gpu.png — label-overlap
**Sorun:** In the right panel the two negative bars have their value labels printed on top of the bar fill while every positive label sits outside the bar end.

**Kanıt:** Cropped the right panel at 3x and 4x. '-15.0%' and '-13.0%' are unambiguously rendered inside the salmon rectangles, dark text on mid-salmon; the very next bar down, '10.3%', is drawn clear of the bar to its right. Cause confirmed in source: `axes[1].text(bar.get_width() + 0.3, ...)` with a fixed positive offset and `va='center'` only - for a negative width, width+0.3 lands inside the bar. The current 28-policy render shows the same thing with '-15.2%'.

**Düzeltme:** Mirror offset and alignment by sign: `axes[1].text(w + (0.6 if w >= 0 else -0.6), ..., ha='left' if w >= 0 else 'right')`, and widen xlim on the negative side.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb, cell index 35 (id=cd21), the `for bar in bars2: axes[1].text(...)` loop`


### [MAJOR] NB05_32GPU-Figure02.png / thesis: nb05-fig01-scheduler-jct_32gpu.png — color
**Sorun:** The two panels of the same figure give the same policy two different colours, and colour encodes nothing but rank.

**Kanıt:** Confirmed in source: left panel uses `sns.color_palette('mako', n_colors=len(plot_df))`, right uses `sns.color_palette('flare', n_colors=len(plot_df2))`, both indexed by row position with no policy->colour mapping. Confirmed visually in the shipped PNG: SJF-Oracle is the palest mint green in the left panel and the darkest purple in the right; the right panel's `plot_df2` explicitly drops FIFO (`plot_df[plot_df['Policy / Architecture'] != 'FIFO']`), so the ramp is also offset by one row between panels. Neither panel has a legend or colorbar, so the hue is pure decoration on top of bar length.

**Düzeltme:** Use one flat colour and reserve hue for the reference policies identically in both panels (e.g. a `_colour(p)` helper returning green for SJF-Oracle, red for FIFO, grey for SRF, one blue for everything else), and add a small legend naming those four roles.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb, cell index 35 (id=cd21): `palette = sns.color_palette('mako', ...)` and `palette2 = sns.color_palette('flare', ...)``


### [MAJOR] NB05_32GPU-Figure04.png / thesis: nb05-fig03-slowdown-box_32gpu.png — color
**Sorun:** `palette='muted'` has 10 colours but the plot has 21 boxes, so the palette cycles and the best and worst policies get identical fills.

**Kanıt:** Sampled box fill RGB along the scanline y=230 of the 1584x590 shipped PNG. 18 saturated runs found at 69 px spacing (the grey muted[7] boxes fall below the saturation threshold, producing the two gaps, and short boxes do not reach that row). Mapping run positions back to box indices: box 2 = (217,139,95) = #ee854a, box 3 = (117,191,113) = #6acc64, ... box 11 = (89,125,191) = #4878d0, box 12 = (217,139,95) again, box 21 = (89,125,191) again. So box 1 (SJF-Oracle, muted[0] = #4878d0), box 11 and box 21 (SJF-CNN (Numeric Sequence)) are rendered in the identical blue - the best policy and the wors...

**Düzeltme:** Either drop the palette (`color='#8FA8C8'`, recolouring only SJF-Oracle / FIFO / SRF), or make colour mean something by mapping policies to a family column and passing `hue='family', dodge=False, palette='colorblind'`.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb, cell index 41 (id=e5cd03): `sns.boxplot(..., palette='muted', ...)``


### [MAJOR] NB05_32GPU-Figure05.png / thesis: nb05-fig04-improvement-heatmap_32gpu.png — color
**Sorun:** Annotation text colour flips between white and black on visually identical backgrounds, producing low-contrast cells; and RdYlGn is not colourblind-safe.

**Kanıt:** Sampled cell fills and glyph luminance in column 1 of the shipped PNG. Text colour: 81.6 / 60.1 / 59.9 / 59.0 are white; 57.8 / 57.3 / 55.3 / 49.5 / 49.3 / 45.9 are black. Cell fills at the flip point: 59.0 sits on RGB (105,190,99) with WHITE text while 57.8 sits on RGB (107,191,100) with BLACK text - a 2/1/1 unit difference in background, opposite text colours. White on (107,191,100) computes to roughly 2.25:1 contrast (relative luminance ~0.417), well under the 4.5:1 threshold, and it degrades further after the 0.29x downscale in the PDF. `cmap='RdYlGn'` confirmed verbatim in source, and the...

**Düzeltme:** Bound the ramp and drive annotation colour explicitly: `cmap='RdYlBu', vmin=-25, vmax=100, center=0, annot_kws={'fontsize': 8}`, then `for t in ax.texts: t.set_color('white' if abs(float(t.get_text())) > 78 else 'black')`.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb, cell index 44 (id=e5cd05): `sns.heatmap(hm_df, ..., cmap='RdYlGn', center=0, ...)``


### [MAJOR] NB05_32GPU-Figure01.png vs thesis/latex/figures/mae_spearman_vs_jct_gain_32gpu.png — other
**Sorun:** Two contradicting copies of the same rank-correlation figure exist in the repo, disagreeing on the correlation the argument rests on.

**Kanıt:** md5 verified and matches the reported values exactly: results/figures/thesis_export/png/NB05_32GPU-Figure01.png = 3bbf0c65bbb2597b78277dc2886399ac (Aug 31), thesis/latex/figures/mae_spearman_vs_jct_gain_32gpu.png = 0653e273c940866c921311760e6a4dd4 (Sep 6). Opened both: the first is titled 'Predictor Quality vs Scheduling Gain Across 18 Runtime Models (32-GPU)' with Pearson r = -0.164 and 0.723; the second says 'Across 21 Runtime Models' with r = -0.065 and 0.323. The ranking-quality correlation drops from 0.723 to 0.323 between the two copies. Cause confirmed in source: cell 29 savefigs direct...

**Düzeltme:** Let one mechanism own the file: drop the direct `fig.savefig(...)` loop from cell 29 and rely on THESIS_FIGURE_MAP (which already maps ('NB05_32GPU',1) to mae_spearman_vs_jct_gain_32gpu.png), or re-run the export so both copies regenerate in one pass. Delete the stale positional copy.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb, cell index 29 (id=7a843cee), the `for _out_dir in [...]: fig.savefig(...)` block`


### [MAJOR] NB05_256GPU-Figure02.png vs NB05_32GPU-Figure02.png — axis
**Sorun:** Yan yana basilan iki seklin sol paneli x eksenini farkli bicimlendiriyor: biri duz tamsayi, digeri 1e6 ortak carpanli.

**Kanıt:** Her iki PNG'yi de actim. NB05_256GPU-Figure02.png sol panel tick'leri: 0, 25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000 (duz tamsayi). NB05_32GPU-Figure02.png sol panel tick'leri: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6 ve sag alt kosede kucuk puntolu '1e6' ofset metni. Tez ikisini yan yana 0.48\textwidth ile basiyor (6.results_and_discussion.tex satir 190/192), olcek 0.163 => '1e6' pratikte gorunmez. Gercek degerler: 32-GPU FIFO=1344966s, 256-GPU FIFO=153382s — yani 32-GPU degerleri ~8.8 kat DAHA BUYUK, ama eksen etiketleri tersini dusundurebiliyor. Kodda hicbir formatter/ti...

**Düzeltme:** Her iki deftere ayni FuncFormatter'i koy ve birimi eksen etiketine tasi (ornegin 'Mean JCT (thousand seconds)'), veya style='plain', useOffset=False ile zorla.

**Konum:** `notebooks/en/05_scheduler_evaluation_256_gpu.ipynb cell 35 (id cd21) ve 32-GPU defterindeki esdegeri`


### [MAJOR] NB05_256GPU-Figure04.png vs NB05_32GPU-Figure04.png — color
**Sorun:** palette='muted' 10 renkte donguye giriyor (21 kutu), ve renk siraya gore atandigi icin ayni politika iki sekilde farkli renkte cikiyor.

**Kanıt:** Kod dogrulandi: cell 41 `sns.boxplot(..., order=policy_order, palette='muted', ...)`, policy_order medyan slowdown'a gore. PNG gorsel kontrolu: 256-GPU seklinde 21 kutu var, renkler 11. kutudan itibaren tekrar basliyor (1=mavi, 11=mavi; 2=turuncu, 12=turuncu ...). Politika-renk kaymasi da dogrulandi: 256-GPU sirasi Oracle(mavi), LSTM(Cat)=TURUNCU, LSTM(CatSeq)=yesil, XGBoost(Cat)=KIRMIZI, CNN-LSTM(Cat)=mor; 32-GPU sirasi Oracle(mavi), XGBoost(Cat)=TURUNCU, LSTM(Cat)=YESIL, CNN-LSTM(Cat)=kirmizi, LSTM(CatSeq)=mor. Yani SJF-LSTM (Categorical) 256'da turuncu / 32'de yesil; SJF-XGBoost (Categorica...

**Düzeltme:** Politika ailesine gore sabit POLICY_COLORS sozlugu tanimla ve her iki deftere ver, veya renk bilgi tasimayacaksa tek renk kullan (color='#7fa8d1').

**Konum:** `notebooks/en/05_scheduler_evaluation_256_gpu.ipynb cell 41 (id e5cd03) ve 32-GPU esdegeri`


### [MAJOR] NB05_256GPU-Figure04.png vs NB05_32GPU-Figure04.png — axis
**Sorun:** Yan yana basilan boxplot ciftinin y eksenleri farkli araliklarda; ayni dusey yukseklik iki sekilde farkli slowdown degerine karsilik geliyor.

**Kanıt:** Iki PNG'yi de actim: NB05_256GPU-Figure04.png y ekseni 10^0'dan 10^4'e kadar etiketli (en ust major tick 10^4); NB05_32GPU-Figure04.png y ekseni 10^0'dan 10^5'e (en ust major tick 10^5). Kod dogrulandi: cell 41'de `ax.set_yscale('log')` var ama hicbir set_ylim cagrisi yok, ikisi de veriye gore otomatik olceklenıyor. Tez metni tam da bu iki paneli karsilastiriyor (satir 213), dolayisiyla eksen farki gorsel karsilastirmayi bozuyor.

**Düzeltme:** Iki deftere de ortak sabit y-limit ver (ornegin ax.set_ylim(1, 5e5)) ve ayni LogLocator tick setini zorla; ayni sabitleme CDF ve wait-percentile ciftleri icin de gerekli.

**Konum:** `notebooks/en/05_scheduler_evaluation_256_gpu.ipynb cell 41 (id e5cd03)`


### [MAJOR] NB05_32GPU_Table05.html, NB05_256GPU_Table05.html — units
**Sorun:** The percentile table gives the unit but never names the measurand.

**Kanıt:** Parsed the headers of NB05_32GPU_Table05.html: ['Policy', 'Median (s)', 'P75 (s)', 'P90 (s)', 'P95 (s)', 'P99 (s)', 'Max (s)'] — percentiles of an unnamed quantity. Source confirmed in cell#54 (id=e5cd09): `wt = df_p["waiting_time"]` then `pct_rows.append({"Policy": policy, "Median (s)": round(wt.median(), 1), "P75 (s)": ...})`, so they are wait times. The companion figure's axis label is likewise `ax.set_xlabel("Seconds (log scale)")` — confirmed both in the code and on the rendered nb05-fig05-wait-percentile_32gpu.png. The thesis caption tab:waitpercentile is indeed the only place the word '...

**Düzeltme:** Rename the keys to 'Median Wait (s)', 'P75 Wait (s)', … and change the axis label to 'Wait time (s, log scale)'.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb cell#54 (id=e5cd09), the `pct_rows.append({...})` dict and `ax.set_xlabel("Seconds (log scale)")``


### [MAJOR] NB05_32GPU-Figure02.png, NB05_256GPU-Figure02.png (= thesis figures/nb05-fig01-scheduler-jct_*.png) — color
**Sorun:** The two panels of one figure use two different sequential colour maps for the same policies, with no legend or colourbar.

**Kanıt:** Confirmed in the rendered PNG and in the code. In NB05_32GPU-Figure02.png the left panel runs from near-black at the top to pale mint at the bottom (mako) while the right panel runs from salmon at the top to dark purple at the bottom (flare). SJF-Oracle is the pale mint bar in the left panel and the dark purple bar in the right; SJF-CNN-LSTM (Numeric Sequence) is near-black on the left and salmon on the right. Neither panel has a legend or colourbar. Source verified verbatim in cell#35 (id=cd21): `palette = sns.color_palette("mako", n_colors=len(plot_df))` and `palette2 = sns.color_palette("fl...

**Düzeltme:** Drop the per-panel ramps and colour by feature track identically in both panels with one figure-level legend.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb cell#35 (id=cd21)`


### [MAJOR] NB05_32GPU-Figure06.png, NB05_256GPU-Figure06.png (= thesis figures/nb05-fig05-wait-percentile_*.png) — axis
**Sorun:** Log x-axes over less than one decade leave almost no readable ticks, and the two panels are sorted independently.

**Kanıt:** Opened thesis/latex/figures/nb05-fig05-wait-percentile_32gpu.png. The left panel shows exactly two labelled x ticks (10^5 and 10^6); the right panel shows exactly two (2x10^6 and 3x10^6). No individual policy value can be read off either panel. Independent sorting confirmed visually: left panel row 2 is 'SJF-LSTM (Categorical)' while right panel row 2 is 'SJF-CNN-LSTM (Categorical)', and the orders diverge throughout. Source confirmed in cell#54 (id=e5cd09): the per-axis loop does `df_sorted = pct_df.sort_values(col, ascending=True)` with col = 'Median (s)' for axes[0] and 'P95 (s)' for axes[1...

**Düzeltme:** Add minor log ticks with a ScalarFormatter (or drop the log scale on the right panel, which spans under half a decade), and sort both panels by the same key with the median shown as a second marker on the same row.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb cell#54 (id=e5cd09)`


### [MAJOR] NB05_32GPU_Table04.html, NB05_256GPU_Table04.html — other
**Sorun:** Emoji used as data values, p-value printed as exactly zero, six decimal places on seconds and rank sums, and no CI or effect size.

**Kanıt:** Parsed the rows directly. Row 0: ['SJF-Oracle', '252023.700000', '1092942.400000', '127580168.000000', '0.000000', '✅ Yes (p<0.05)']. Row 18: ['SJF-CNN (Numeric Sequence)', '1519411.600000', '-174445.500000', '55937416.500000', '1.000000', '❌ No']. So all four confirmed: emoji as data, p-value '0.000000', and six decimals on Mean JCT (s), JCT vs FIFO (s) and Wilcoxon W. Columns present are only [Policy, Mean JCT (s), JCT vs FIFO (s), Wilcoxon W, p-value, Significant?]. The staleness claim is also confirmed: cell#47 (id=e5cd07) now builds sig_rows with '95% CI low', '95% CI high', 'Effect size ...

**Düzeltme:** Re-export after re-running the notebook and format at display time with a Styler format map; ship the 95% CI and effect-size columns.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb cell#47 (id=e5cd07), the `display(sig_df.style.map(...))` call`


### [MINOR] NB05_32GPU-Figure05.png / thesis: nb05-fig04-improvement-heatmap_32gpu.png — caption-mismatch
**Sorun:** Column headers and title point in opposite directions.

**Kanıt:** Read the source and the PNG. The dict keys are literally 'Wait ↓ %', 'JCT ↓ %', 'Slowdown ↓ %', and they render as column headers in the shipped figure. Directly above them the title reads 'Scheduling Policy Improvement over FIFO Baseline\n(% reduction - higher = better)', and the colorbar is labelled 'Improvement over FIFO (%)'. The down arrow conventionally marks lower-is-better, so a reader trusting it would read SJF-Oracle's 99.5 in the Slowdown column as the worst cell rather than the best.

**Düzeltme:** Drop the arrow glyph and name the quantity: 'Wait time reduction (%)', 'JCT reduction (%)', 'Mean slowdown reduction (%)', and shorten the title to 'Improvement over FIFO baseline (higher is better)'.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb, cell index 44 (id=e5cd05), the `heatmap_rows.append({...})` dict keys`


### [MINOR] NB05_32GPU-Figure06.png / thesis: nb05-fig05-wait-percentile_32gpu.png — axis
**Sorun:** The P95 panel uses a log x-axis over less than one decade, yielding only two labelled ticks; and the two panels are sorted independently.

**Kanıt:** Opened the shipped PNG. The right (P95) panel shows exactly two labelled x ticks, '2 x 10^6' and '3 x 10^6'; the leftmost point (SJF-Oracle) sits well left of both with no labelled tick anywhere near it. Interpolating from the tick spacing the data range is roughly 1.3e6 to 3.3e6 s - about 0.4 of a decade - so the log scale buys nothing while costing the tick labels. Source confirms `ax.set_xscale('log')` is applied unconditionally to both panels, and the cell's own comment justifying log scale argues from bars having no zero, while the panel in fact plots dots (`ax.plot(..., 'o', ms=7)`), for...

**Düzeltme:** Use one shared row order derived from the median panel and apply log only when the ratio warrants it (`if d[col].max()/d[col].min() > 20: ax.set_xscale('log')`), plus a FuncFormatter rendering values as e.g. '2.1M'. Set y tick labels only on axes[0] so the shared order is visually obvious.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb, cell index 54 (id=e5cd09), the `for ax, col, title in [...]` loop`


### [MINOR] mae_spearman_vs_jct_gain_32gpu.png (= NB05_32GPU-Figure01.png) — truncation
**Sorun:** Point annotations for the rightmost markers are drawn past the axes frame, and the reported statistics are incomplete for a journal.

**Kanıt:** Cropped the right panel's right edge (x=2150-2378) at 4x: 'ProfileMedian (baseline)' begins exactly at the right spine and runs entirely outside the axes into the figure margin, and it also overprints the tail of 'LGBM (Categorical)'. Cause confirmed in source: `xytext=(4, 4)` in offset points with no xlim headroom and no clipping. The older exported copy does the same with 'LGBM (Categorical)'. Statistics claim confirmed too: the source discards the p-value with `r_value, _ = pearsonr(...)` and the annotation box prints only `f'Pearson r = {r_value:.3f}'` - no p, no n - even though `rho_p` an...

**Düzeltme:** Add `ax.margins(x=0.12)` for right-hand headroom, and report the test in full: `r_value, p_value = pearsonr(...)` then annotate `f'Pearson r = {r_value:.3f} (p = {p_value:.3f}, n = {len(rank_df)})'`.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb, cell index 29 (id=7a843cee): `r_value, _ = pearsonr(...)` and the annotate loop`


### [MINOR] NB05_32GPU-Figure02.png, NB05_256GPU-Figure02.png — other
**Sorun:** Bar value labels are raw integers with a glued unit while the same panel's axis uses scientific offset notation; negative bar labels sit on the wrong side of zero.

**Kanıt:** Read off the rendered PNG: left-panel labels are '1546191s', '1519412s', '1344966s', '252024s' — no thousands separator, unit glued — while the panel's x-axis is drawn 0.0 / 0.2 / … / 1.6 with a '1e6' offset in the corner. In the right panel the two negative labels '-15.0%' and '-13.0%' are drawn on top of their own orange bars, while every positive label sits outside its bar. Source confirmed in cell#35 (id=cd21): `f'{bar.get_width():.0f}s'` for the left panel, and `axes[1].text(bar.get_width() + 0.3, ...)` unconditionally for the right, which lands inside the bar when the width is negative.

**Düzeltme:** Use `f'{bar.get_width():,.0f} s'` with `ticklabel_format(axis='x', style='plain')`, and pick the offset/ha by sign in the right panel.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb cell#35 (id=cd21)`


---

## NB05_256GPU — 15 bulgu (4 kritik)

### [CRITICAL] NB05_256GPU-Figure01.png + thesis/latex/figures/mae_spearman_vs_jct_gain_256gpu.png — label-overlap
**Sorun:** Scatter panellerinde nokta etiketleri harf harf ic ice gecmis, okunaksiz yumaklar olusturuyor. Her iki panelde de var.

**Kanıt:** Iki PNG'yi de gorsel olarak actim ve kritik bolgeleri buyuttum (crop+2x). GUNCEL dosyada (mae_spearman_vs_jct_gain_256gpu.png, 2378x1034) sol panel ust bolgesinde 'LSTM (Categorical)' + 'XGBoost (Categorical)' + 'XGBoost (Native Cat)' harfleri ust uste binmis ('XGBoost (Categorical)' ile 'LSTM (Categorical)' tamamen ic ice); ayrica 'LSTM (Numeric)' / 'LSTM (Numeric Sequence)' / 'RF (Numeric)' / 'CNN (Categorical)' / 'CNN-LSTM (Categorical Sequence)' ayni bandda carpisiyor. Sag panelde 'LSTM (Categorical)'+'XGBoost (Native Cat)'+'XGBoost (Categorical)' ve 'LSTM (Categorical Sequence)'+'CNN-LSTM...

**Düzeltme:** adjustText kullan, ya da sadece uc noktalari etiketle / M1..M21 kisa kod + altta kod-model tablosu.

**Konum:** `notebooks/en/05_scheduler_evaluation_256_gpu.ipynb cell 29 (id 8d1a4e10), annotate dongusu`


### [CRITICAL] NB05_256GPU-Figure01..06.png ve thesis/latex/figures/nb05-fig01..05-*_256gpu.png — other
**Sorun:** Tezde kullanilan 5 NB05 PNG'si defterin guncel calismasiyla uyusmuyor; ustelik ayni tez icinde yan yana duran iki sekil birbiriyle celisen sayilar gosteriyor.

**Kanıt:** md5 karsilastirmasi: thesis/latex/figures/nb05-fig01..05-*_256gpu.png dosyalari byte-byte NB05_256GPU-Figure02..06.png ile ayni (5fd6748c…, 04e83421…, c52d8f20…, 7a8b05cf…, 9f32c25b…), hepsi 31 Agu 15:41 tarihli. Defter ise 6 Eyl 09:59. Defterin KAYITLI cell-35 ciktisini base64'ten cikarip actim: 28 politika var ve SJF-LSTM (Categorical)=61.3%, SJF-LGBM (Categorical)=29.3%, SJF-CNN-LSTM (Numeric)=5.8%, SJF-CNN (Categorical Sequence)=2.1%. Ihrac edilmis PNG ise 21 politika ve 63.2% / 61.4% / 31.9% / 17.5% gosteriyor. results/analysis/rank_correlation_256gpu.csv (6 Eyl 09:03) LGBM (Categorical) ...

**Düzeltme:** Defteri bastan calistir, scripts/export_thesis_results.py ile tum NB05 sekillerini yeniden uret, thesis/latex/figures'a kopyala; betige notebook mtime > PNG mtime tazelik kontrolu ekle.

**Konum:** `notebooks/en/05_scheduler_evaluation_256_gpu.ipynb (tum sekil hucreleri) + scripts/export_thesis_results.py`


### [CRITICAL] NB05_256GPU-Figure04.png (thesis: nb05-fig03-slowdown-box_256gpu.png) — label-overlap
**Sorun:** X ekseni kategori etiketleri ust uste biniyor; iki politika adi tamamen okunamaz hale geliyor.

**Kanıt:** PNG'yi gorsel olarak actim: 11. tick'te 'SJF-CNN-LSTM (Categorical Sequence)' ile 'SRF (Heuristic)' harfleri ic ice gecmis ('...cal SequenceRF)' seklinde bozuk); saga dogru 'FIFO' etiketi 'SJF-CNN-LSTM (Numeric Sequence)' uzerine binmis. NB05_32GPU-Figure04.png'de de ayni iki carpisma mevcut ('SJF-CNN-LSTM (Categorical Sequence)' x 'SJF-LSTM (Numeric)' ve 'FIFO' x 'SJF-CNN-LSTM (Numeric Sequence)'). Kod dogrulandi: cell 41 (id e5cd03) `ax.tick_params(axis="x", rotation=35)` — ha/rotation_mode verilmemis; ayni satir 32-GPU defterinin cell 41'inde de var.

**Düzeltme:** plt.setp(ax.get_xticklabels(), rotation=40, ha='right', rotation_mode='anchor') ya da boxplot'u yatay cevir (y='policy').

**Konum:** `notebooks/en/05_scheduler_evaluation_256_gpu.ipynb cell 41 (id e5cd03)`


### [CRITICAL] NB05_256GPU-Figure03.png (thesis: nb05-fig02-wait-cdf_256gpu.png) — legend-error
**Sorun:** Lejant sayisi sabit string; guncel calismada yanlis. Ayrica heuristik baseline'lar 'ML' olarak etiketleniyor ve 'best ML' iddiasi gorsel olarak celisiyor.

**Kanıt:** Kod dogrulandi: cell 38 (id e5cd01) `ax.plot([], [], lw=0.9, color="0.75", label="other ML policies (17)")` — sabit string, hesaplanmiyor. Cell 54'un kayitli pct_df ciktisi guncel calismada 28 politika listeliyor (0..27), _highlight 4 tane => 24 gri egri, lejant hala '(17)'. Ihrac edilmis PNG'de 21 politika oldugu icin sayi tesadufen dogru; yeniden uretilir uretilmez yanlis olacak. Baseline sorunu kodda dogrulandi: _anchors sadece FIFO/SRF/SJF-Oracle; SJF-UserMedian, SJF-ProfileMedian, SJF-AlibabaEstimate* politikalari _ml_policies'e dusuyor ve 'other ML policies' altinda toplaniyor. Ucuncu id...

**Düzeltme:** _n_other'i len(all_results['policy'].unique()) - len(_highlight) ile hesapla; etiketi 'other policies (N)' yap; label'i f'{policy} (best ML by mean wait)' olarak kritere bagla.

**Konum:** `notebooks/en/05_scheduler_evaluation_256_gpu.ipynb cell 38 (id e5cd01)`


### [MAJOR] NB05_256GPU-Figure02.png (thesis: nb05-fig01-scheduler-jct_256gpu.png) — label-overlap
**Sorun:** Negatif barlarin deger etiketleri barin ICINE dusuyor, pozitiflerinki disina — tutarsiz ve okunaksiz.

**Kanıt:** PNG gorsel kontrolu: sag panelde '-25.9%' ve '-9.3%' turuncu barin uzerinde siyah metin olarak duruyor; '15.0%' ve sonraki tum pozitifler barin sagina, beyaz zemine yaziliyor. NB05_32GPU-Figure02.png'de ayni sekilde '-15.0%' ve '-13.0%' bar icinde. Defterin GUNCEL cell-35 ciktisinda durum daha kotu: '-22.6%' bar icinde, ve 'SJF-AlibabaEstimate-Group (baseline)' icin '-0.3%' etiketi neredeyse sifir genislikte gorunmez bir barin ustunde. Kod dogrulandi: cell 35 (id cd21) `axes[1].text(bar.get_width() + 0.3, ...)` — isaret kontrolu yok, get_width()<0 icin ofset barin ic bolgesine dusuyor.

**Düzeltme:** off = 0.5 if w >= 0 else -0.5; ha='left' if w>=0 else 'right' ile isarete gore ofset ve hizalama ver.

**Konum:** `notebooks/en/05_scheduler_evaluation_256_gpu.ipynb cell 35 (id cd21), bars2 dongusu`


### [MAJOR] NB05_256GPU-Figure02.png (thesis: nb05-fig01-scheduler-jct_256gpu.png) — truncation
**Sorun:** Sol paneldeki en ust bar deger etiketi eksen cercevesini kesiyor.

**Kanıt:** PNG'yi 4x buyuterek actim: '193114s' etiketinin uzerinden eksen cerceve cizgisi geciyor — '193' cerceve icinde, '114s' cerceve disinda; cizgi '3' ile '1' arasindan gecerek metni ikiye boluyor. Defterin guncel cell-35 ciktisinda ayni tasma '174505s' icin mevcut. Kod dogrulandi: cell 35 (id cd21) `axes[0].text(bar.get_width() + 0.01 * plot_df['Mean JCT (s)'].max(), ...)`; hicbir set_xlim/margins cagrisi yok, xlim otomatik ~max*1.05'te kaliyor.

**Düzeltme:** axes[0].set_xlim(0, plot_df['Mean JCT (s)'].max() * 1.18) veya bar_label(padding=3) + margins(x=0.15).

**Konum:** `notebooks/en/05_scheduler_evaluation_256_gpu.ipynb cell 35 (id cd21), sol panel etiket dongusu`


### [MAJOR] NB05_256GPU-Figure01.png + thesis/latex/figures/mae_spearman_vs_jct_gain_256gpu.png — truncation
**Sorun:** Sag paneldeki en sagdaki nokta etiketi eksen cercevesinin uzerinden gecip disariya tasiyor.

**Kanıt:** Her iki dosyayi da sag kenardan 3x buyuterek actim. NB05_256GPU-Figure01.png: 'LGBM (Categorical)' etiketinin uzerinden sag spine cizgisi geciyor — 'LGBM' icerde, '(Categorical)' disarida, sekil kenar boslugunda. GUNCEL mae_spearman_vs_jct_gain_256gpu.png: ayni tasma bu kez 'ProfileMedian (baseline)' etiketinde — spine cizgisi 'ProfileMe' ile 'dian (baseline)' arasindan geciyor. Yani model degisse de hata devam ediyor. Kod dogrulandi: cell 29 annotate sabit (4,4) ofset, clip_on/margins ayari yok.

**Düzeltme:** ax.margins(x=0.18) ekle ve/veya en sag noktalarda ha='right', xytext=(-4,4) kullan.

**Konum:** `notebooks/en/05_scheduler_evaluation_256_gpu.ipynb cell 29 (id 8d1a4e10), annotate dongusu`


### [MAJOR] NB05_256GPU-Figure01.png + mae_spearman_vs_jct_gain_256gpu.png — axis
**Sorun:** sharey=True ile olusturulan panellerin ikisine de ylabel veriliyor; sag panelde y-etiketi var ama sayisal tick etiketleri yok.

**Kanıt:** Kod dogrulandi: cell 29 `fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)` ve dongu icinde her iki eksen icin `ax.set_ylabel(f'{N_GPU}-GPU JCT Improvement over FIFO (%)')`. Iki PNG'de de gorsel olarak dogrulandi: sag panelin sol kenarinda tick CIZGILERI var ama hicbir rakam yok, buna ragmen '256-GPU JCT Improvement over FIFO (%)' y-etiketi sag panelde de yaziliyor — yani ayni etiket sekilde iki kez tekrarlaniyor ve sag paneldeki bir noktanin degeri dogrudan okunamiyor.

**Düzeltme:** Dongu icindeki set_ylabel'i sil, dongu sonrasi sadece axes[0].set_ylabel(...) ver; okunabilirlik icin axes[1].tick_params(labelleft=True) ekle.

**Konum:** `notebooks/en/05_scheduler_evaluation_256_gpu.ipynb cell 29 (id 8d1a4e10)`


### [MAJOR] NB05_256GPU-Figure05.png (thesis: nb05-fig04-improvement-heatmap_256gpu.png) — color
**Sorun:** Heatmap kirmizi-yesil diverjan palet (RdYlGn) kullaniyor — renk korlugu erisilebilirligi acisindan problemli.

**Kanıt:** Kod dogrulandi: cell 44 (id e5cd05) `sns.heatmap(hm_df, ax=ax, annot=True, fmt='.1f', cmap='RdYlGn', center=0, linewidths=0.5, ...)`. 32-GPU defterinin cell 44'unde de ayni `cmap="RdYlGn"` var. PNG gorsel kontrolu: iyi degerler koyu/orta yesil, kotu degerler turuncu-kirmizi; renk disinda desen/sekil gibi ikinci bir ayirt edici kanal yok. Deuteranopide bu iki uc birbirine yaklasir. (Not: hucre ici sayisal annot oldugu icin bilgi tamamen kaybolmuyor — bu yuzden CRITICAL degil MAJOR olarak degerlendirmek dogru.)

**Düzeltme:** cmap='RdYlBu' / 'coolwarm' / 'PuOr' / 'BrBG' gibi renk-korlugune guvenli diverjan palete gec; ayni degisikligi 32-GPU defterinde de yap.

**Konum:** `notebooks/en/05_scheduler_evaluation_256_gpu.ipynb cell 44 (id e5cd05)`


### [MAJOR] NB05_256GPU-Figure05.png (thesis: nb05-fig04-improvement-heatmap_256gpu.png) — color
**Sorun:** Hucre ici annot metin rengi otomatik luminansa birakilmis; neredeyse ayni renkteki komsu hucrelerde siyah/beyaz arasinda ziplyor ve beyaz olanlar WCAG esiginin altinda.

**Kanıt:** PNG piksellerini olctum (PIL + WCAG kontrast formulu): '58.6' hucresi arka plan RGB(105,190,99), metin BEYAZ -> kontrast 2.30:1 (buyuk metin esigi 3:1'in altinda). Hemen sagindaki '56.3' hucresi arka plan RGB(112,193,100) — gozle ayirt edilemeyecek kadar ayni yesil — ama metin SIYAH -> 9.5:1. Bir alt satirdaki '58.2' hucresi RGB(107,191,100), metin yine SIYAH. Buyutulmus crop'ta bu sicrama cok net gorunuyor: ayni yesil tonda ust satir beyaz, alt satir siyah yazi. Kod dogrulandi: cell 44'te annot_kws hic verilmemis, seaborn kendi luminans esigine gore renk seciyor.

**Düzeltme:** annot_kws={'color': 'black', 'fontsize': 8} ile annot rengini sabitle; daha acik uclu bir palet (RdYlBu) ile birlikte tum hucrelerde >4.5:1 olur.

**Konum:** `notebooks/en/05_scheduler_evaluation_256_gpu.ipynb cell 44 (id e5cd05)`


### [MAJOR] NB05_256GPU-Figure06.png (thesis: nb05-fig05-wait-percentile_256gpu.png) — axis
**Sorun:** Sag panelin (P95) x ekseninde etiketli tek bir ana (decade) tick yok; log olcek 2.6 katlik dar bir aralikta fayda saglamiyor.

**Kanıt:** PNG'yi actim: sag panelin x ekseninde SADECE iki minor tick etiketi var — '2 x 10^5' ve '3 x 10^5'; 10^5 ve 10^6 ana tick'leri gorunur aralik disinda. Defterin cell-54 kayitli tablosu P95 araligini dogruluyor: min 142927.0 (SJF-Oracle), max 384569.1 (SJF-CNN (Numeric Sequence)) — yani 2.7 kat. Sekilde en sagdaki nokta sag cerceve cizgisine deger halde, marj yok. NB05_32GPU-Figure06.png'de ayni sorun ('2 x 10^6', '3 x 10^6' ve en sagdaki nokta cercevede). Kod dogrulandi: cell 54'te `ax.set_xscale('log')` iki panele de kosulsuz uygulaniyor, set_xticks/margins yok.

**Düzeltme:** P95 paneli icin lineer olcege gec, FuncFormatter ile bin saniyeye tasi, xlabel'i guncelle ve margins(x=0.08) ekle; log kalacaksa tick'leri acikca ver.

**Konum:** `notebooks/en/05_scheduler_evaluation_256_gpu.ipynb cell 54 (id e5cd09)`


### [MAJOR] NB05_256GPU-Figure01.png vs Figure02..06 — naming-inconsistency
**Sorun:** Ayni politika sekiller arasinda farkli isimlerle geciyor; ayrica baslikta sayilan model sayisi diger sekillerdeki satir sayisiyla uyusmuyor.

**Kanıt:** Kod dogrulandi: cell 29'da `row['Policy / Architecture'].replace('SJF-', '')`. Sekiller gorsel olarak karsilastirildi: Figure01/mae_spearman'da etiketler 'LSTM (Categorical)', 'XGBoost (Categorical)', 'RF (Numeric)'; Figure02, Figure04, Figure05, Figure06'da ayni satirlar 'SJF-LSTM (Categorical)', 'SJF-XGBoost (Categorical)', 'SJF-RF (Numeric)'. Baslik sayisi: NB05_256GPU-Figure01.png ust basligi 'Predictor Quality vs Scheduling Gain Across 18 Runtime Models (256-GPU)' iken NB05_256GPU-Figure02.png 21 politika listeliyor; guncel mae_spearman_vs_jct_gain_256gpu.png basligi '21 Runtime Models' i...

**Düzeltme:** Tek bir kanonik DISPLAY_NAME sozlugu kullan (ya da .replace('SJF-','') cagrisini kaldir) ve basligi len(rank_df)'e baglayip 'Learned Runtime Predictors' de.

**Konum:** `notebooks/en/05_scheduler_evaluation_256_gpu.ipynb cell 29 (id 8d1a4e10): replace + fig.suptitle`


### [MINOR] NB05_256GPU-Figure03.png (thesis: nb05-fig02-wait-cdf_256gpu.png) — axis
**Sorun:** X ekseninde verinin bittigi yerden sonra yaklasik bir tam dekad bos alan var; ayrica lejant eksen disinda genis yer kapliyor.

**Kanıt:** Piksel olcumu yaptim: eksen spine'lari x=61 ve x=1102 px; major decade tick'leri 61 (10^0), 221, 381, 541, 700, 860, 1020 (10^6) — dekad basina ~159.8 px. Sag sinir 1102 => 10^(6+0.51) ~ 3.3x10^6. Verinin maksimumu cell-54 tablosundan 425911.8 s = 10^5.63 => x~961 px. Yani 961-1102 arasi, ~0.88 dekad tamamen bos (cizim alaninin ~%13.5'i; bulgudaki '%20' bir miktar abartili). Ayrica eksen sag kenari (1102) ile sekil sag kenari (1389) arasi lejanta gidiyor = sekil genisliginin %20.7'si. Kod dogrulandi: cell 38'de sadece `ax.set_xlim(left=1)` var, sag sinir belirlenmemis; `ax.legend(bbox_to_ancho...

**Düzeltme:** ax.set_xlim(1, all_results['waiting_time'].max()*1.15) (veya 32/256 ciftinde ortak sabit sinir) ve lejanti ax.legend(loc='upper left', framealpha=0.9) ile eksen icine al.

**Konum:** `notebooks/en/05_scheduler_evaluation_256_gpu.ipynb cell 38 (id e5cd01)`


### [MINOR] NB05_256GPU-Figure06.png (thesis: nb05-fig05-wait-percentile_256gpu.png) — other
**Sorun:** Iki panel birbirinden bagimsiz siralanmis; ayni dusey konum iki panelde farkli politikaya karsilik geliyor.

**Kanıt:** Kod dogrulandi: cell 54'te dongu icinde `df_sorted = pct_df.sort_values(col, ascending=True)` — her panel kendi sutununa gore yeniden siraliyor. PNG gorsel kontrolu bunu dogruluyor: sol panelin 2. satiri SJF-LSTM (Categorical), sag panelin 2. satiri SJF-CNN-LSTM (Categorical); sol panelde SRF (Heuristic) 11. sirada, sag panelde 8. sirada. Her iki panelde de tam politika adlari y ekseninde tekrarlanmis. Iki panel ayni gorsel gramerde oldugu icin 'ayni satir = ayni politika' yanilsamasi olusuyor.

**Düzeltme:** Iki paneli tek ortak siralamaya sabitle (ornegin P95), sol panelin tick etiketlerini birak, sag panelinkini kaldir; alternatif olarak tek panelde dumbbell plot.

**Konum:** `notebooks/en/05_scheduler_evaluation_256_gpu.ipynb cell 54 (id e5cd09)`


### [MINOR] NB05_256GPU-Figure01..06.png (ihrac zinciri) + scripts/export_thesis_results.py — other
**Sorun:** Ihrac betigindeki konum-tabanli harita ile diskteki dosya seti arasinda bir konumluk kayma var; ayrica haritanin urettigi bir sekil tezde hic kullanilmiyor.

**Kanıt:** scripts/export_thesis_results.py satir ~102-110: ('NB05_256GPU', 1)->mae_spearman..., ( ,2)->nb05-fig06-load-backfill-sensitivity_256gpu.png, ( ,3)->nb05-fig01-scheduler-jct_256gpu.png ... ve EXPECTED_FIGURE_COUNT['NB05_256GPU'] = 7. Diskte ise sadece Figure01..06 (6 dosya) var. md5 ile eslestirdim: NB05_256GPU-Figure02.png (5fd6748c…) = thesis/latex/figures/nb05-fig01-scheduler-jct_256gpu.png, yani haritaya gore index 3'te olmasi gereken sekil diskte index 2'de duruyor — set, sensitivity sekli (cell 31, id d7a6e287) eklenmeden onceki bir revizyondan. Defterde su an gercekten 7 sekil hucresi v...

**Düzeltme:** Defteri bastan calistirip betigi yeniden kos (Figure01..07); nb05-fig06 icin tex'e figure blogu ekle; kaliciligi icin THESIS_FIGURE_MAP anahtarini (prefix, cell_id) yapip cell.get('id') ile esle.

**Konum:** `scripts/export_thesis_results.py satir 96-124 (THESIS_FIGURE_MAP ve EXPECTED_FIGURE_COUNT)`


---

## mae_spearman — 15 bulgu (3 kritik)

### [CRITICAL] mae_spearman_vs_jct_gain_32gpu.png ve _256gpu.png (sag panel) — axis
**Sorun:** Sag panelde y-ekseni tick SAYILARI yok ama y-ekseni BASLIGI var - okuyucu yuzde degerini sag panelden okuyamiyor.

**Kanıt:** Kodu offline yeniden calistirip olctum: panel1 icin gorunur y-tick etiketi sayisi = 0 (panel0 icin 32-GPU'da 10, 256-GPU'da 7), buna ragmen axes[1].get_ylabel() = '32-GPU JCT Improvement over FIFO (%)' dolu. PNG'yi kirpip (x=1130-1400) gorsel olarak da dogruladim: sag panelin sol kenarinda tick cizgileri var, hicbir sayi yok, yaninda dondurulmus '32-GPU JCT Improvement over FIFO (%)' yazisi duruyor. Kod nedeni birebir dogrulandi: `plt.subplots(1, 2, figsize=(16, 7), sharey=True)` + dongu icinde HER IKI eksene `ax.set_ylabel(f"{N_GPU}-GPU JCT Improvement over FIFO (%)")`.

**Düzeltme:** Dongudeki `ax.set_ylabel(...)` satirini sil; dongu sonrasi `axes[0].set_ylabel(f"{N_GPU}-GPU JCT Improvement over FIFO (%)")` ve `axes[1].tick_params(labelleft=True)` ekle.

**Konum:** `/Users/hasanugurcelebi/Thesis/alibaba-gpu-runtime-prediction-and-scheduling/notebooks/en/05_scheduler_evaluation_32_gpu.ipynb hucre index 29 (id 7a843cee); /Users/hasanugurcelebi/Thesis/alibaba-gpu-runtime-prediction-and-scheduling/notebooks/en/05_scheduler_evaluation_256_gpu.ipynb hucre id 8d1a4e10 (kod birebir ayni)`


### [CRITICAL] mae_spearman_vs_jct_gain_32gpu.png ve _256gpu.png (her iki panel) — label-overlap
**Sorun:** 21 nokta etiketi sabit (4,4) pt ofsetle, collision-avoidance olmadan basiliyor; cok sayida etiket ust uste binip okunamaz hale geliyor. NOT: raporlanan yuzde degerlerinin bir kismi abartili, ancak cakismalarin varligi ve ciddiyeti dogrulandi.

**Kanıt:** Kodu yeniden calistirip her etiketin renderer bbox'ini alarak cift-cift kesisim olctum. 256-GPU sol panel: 14 cakisan cift (rapordaki '14 cift' sayisi BIREBIR dogru); en kotuler LSTM (Categorical) x XGBoost (Categorical) %68.8, CNN-LSTM (Cat Seq) x CNN (Categorical) %67.2, CNN-LSTM (Cat Seq) x CNN (Numeric) %56.2, LSTM (Numeric Sequence) x LSTM (Numeric) %55.4. 256-GPU sag panel: 10 cakisan cift (rapordaki '10 cift' BIREBIR dogru); LSTM (Cat Seq) x CNN-LSTM (Categorical) %66.8, CNN-LSTM (Cat) x RF (Cat) %55.9, XGBoost (Native Cat) x XGBoost (Categorical) %50.7. 32-GPU sag panel: 5 cift; XGBoos...

**Düzeltme:** adjustText ile yerlesim (texts listesine topla, `adjust_text(texts, ax=ax, expand_points=(1.3,1.4), arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5))`); veya sadece anlatida gecen ~8 modeli etiketleyip kalanini renk/marker lejantina tasi; ek olarak isimleri kisalt (Categorical->Cat, Sequence->Seq) ve kisaltmayi TUM sekillerde ayni uygula.

**Konum:** `Ayni hucreler (7a843cee / 8d1a4e10) - `for _, row in rank_df.iterrows(): ax.annotate(row["Policy / Architecture"].replace("SJF-", ""), (row[x_col], row["JCT Improvement %"]), xytext=(4, 4), textcoords="offset points", fontsize=7)``


### [CRITICAL] mae_spearman_vs_jct_gain_32gpu.png ve _256gpu.png (kesikli trend cizgisi + 'Pearson r' kutusu) — other
**Sorun:** Dort adet istatistiksel olarak anlamsiz iliski icin kesikli OLS trend cizgisi ciziliyor ve yaninda p-degeri / n / guven araligi olmadan ciplak 'Pearson r' yaziliyor.

**Kanıt:** results/analysis/rank_correlation_{32,256}gpu.csv dosyalarindan kendim hesapladim (n=21), raporlanan degerler BIREBIR cikti: 32-GPU sag panel r=+0.323 p=0.153 %95 GA [-0.126,+0.662] (sifiri iceriyor); 32-GPU sol panel r=-0.065 p=0.779 GA [-0.483,+0.377]; 256-GPU sag panel r=+0.279 p=0.221 GA [-0.174,+0.634]; 256-GPU sol panel r=-0.032 p=0.891. Kodda p-degerinin bilerek atildigini dogruladim: `r_value, _ = pearsonr(rank_df[x_col], rank_df["JCT Improvement %"])` ve kutu metni `f"Pearson r = {r_value:.3f}"` - n, p, GA hicbiri yok. Trend cizgisi kosulsuz ciziliyor: `ax.plot(xs, slope * xs + interc...

**Düzeltme:** `r_value, p_value = pearsonr(...)` ile p'yi tut; Fisher-z GA hesapla (`z=np.arctanh(r); se=1/np.sqrt(len(rank_df)-3)`); kutu metnini `f"Pearson r = {r:+.2f} (95% CI [{lo:+.2f}, {hi:+.2f}])\nn = {n}, p = {p:.3f}"` yap; trend cizgisini anlamli degilse noktali ciz ve '(n.s.)' etiketi ver.

**Konum:** `Ayni hucreler - `slope, intercept = np.polyfit(rank_df[x_col], rank_df["JCT Improvement %"], deg=1)`, `ax.plot(xs, slope * xs + intercept, "k--", lw=1.5)`, `r_value, _ = pearsonr(...)`, `ax.text(0.03, 0.94, f"Pearson r = {r_value:.3f}", ...)``


### [MAJOR] mae_spearman_vs_jct_gain_32gpu.png ve _256gpu.png (panel sag kenarlari) — truncation
**Sorun:** Nokta etiketleri eksen cercevesinin (spine) disina tasiyor. Sorun gercek, ancak raporda listelenen etiketlerin bir kismi YANLIS - tasan etiket sayisi raporlanandan az.

**Kanıt:** Renderer bbox'lari ile ax.get_window_extent() karsilastirmasi yaptim (her iki figurde ayni sonuc). Sag panel: SADECE 2 etiket sag spine'i geciyor - 'ProfileMedian (baseline)' 94.1 px ve 'LGBM (Categorical)' 36.5 px. Sol panel: SADECE 1 etiket geciyor - 'CNN-LSTM (Numeric)' 79.2 px. Raporun iddia ettigi 'UserMedian (baseline)' sag panelde spine'i GECMIYOR; 'CNN-LSTM (Categorical)' sol panelde spine'i GECMIYOR - bu iki alt-iddia yanlis. PNG'yi 2x buyutup gorsel dogrulama yaptim: sag panelde dikey siyah spine cizgisi 'ProfileMedian (baseline)' metninin ortasindan geciyor ve metin figur marjina ta...

**Düzeltme:** Sag kenardaki noktalar icin etiketi sola hizala (`ha="right"`, `dx=-6`); `ax.margins(x=0.12)` veya acik `ax.set_xlim(...)` ile etiketlere yer ac; emniyet icin `annotation_clip=True` ekle.

**Konum:** `Ayni hucreler - `ax.annotate(..., xytext=(4, 4), textcoords="offset points")` (annotation_clip / clip_on ayari yok) ve hicbir `ax.set_xlim(...)` / `ax.margins(...)` cagrisi yok`


### [MAJOR] mae_spearman_vs_jct_gain_32gpu.png ve _256gpu.png (tum noktalar) — color
**Sorun:** 21 noktanin tamami tek renk/tek marker; model ailesi, ozellik modu veya ogrenilmis-model vs heuristik baseline ayrimi gorsel olarak hic kodlanmamis. Baseline'lar regresyonun kaldirac noktalari ama okuyucu bunlari ayirt edemiyor.

**Kanıt:** Kodda tek bir cagri var ve c/marker/label parametresi yok: `ax.scatter(rank_df[x_col], rank_df["JCT Improvement %"], s=70, alpha=0.85)` - matplotlib varsayilan C0 mavisi. PNG'de tum noktalar ayni mavi daire. Kaldirac iddiasini kendim hesapladim ve dogruladim: 32-GPU sag panel, iki baseline (ProfileMedian rho=0.720, UserMedian rho=0.597 - CSV'den teyit) cikarilinca r=+0.323 (p=0.153) yerine r=+0.416 (p=0.076) oluyor; 256-GPU'da r=+0.279 -> +0.344 (p=0.149). Yani egim gercekten bu iki noktaya duyarli. Model kimligi sadece fontsize=7 metin etiketiyle veriliyor, o etiketler de cakisiyor (yukaridak...

**Düzeltme:** Aile bazli renk+marker kodlamasi (Okabe-Ito paleti, renk korlugune ve gri basima dayanikli): Tree=(#0072B2,'o'), Deep=(#D55E00,'^'), Baseline=(#000000,'s'); her aile icin ayri scatter cagrisi + label. Baseline'lari trend fitinden ayirmayi (veya iki cizgi cizmeyi) dusun ve bu secimi altyazida belirt.

**Konum:** `Ayni hucreler - `ax.scatter(rank_df[x_col], rank_df["JCT Improvement %"], s=70, alpha=0.85)``


### [MAJOR] mae_spearman_vs_jct_gain_32gpu.png ve _256gpu.png (tum metin) — dpi-quality
**Sorun:** Figur 16 inc genisliginde uretiliyor ama tez metin genisligi 6.10 inc; olcek carpani ~0.38 oldugu icin basili tezde hicbir metin okunabilir punto degerinde kalmiyor.

**Kanıt:** thesis/latex/thesis.cls'te dogruladim: `\LoadClass[a4paper,12pt]{report}`, `\DeclareOption{msc}{\global\let\@dtype\@ne}` ve `\ifx\@dtype\@ne` dalinda left=3.5cm, right=2cm; main.tex `\documentclass[msc]{thesis}` kullaniyor. A4 21cm - 3.5 - 2 = 15.5 cm = 6.10 inc. Kodda `figsize=(16, 7)` -> olcek 6.10/16 = 0.381. Notebook'ta hicbir global rcParams/set_theme/plt.style ayari yok (tum hucrelerde arattim), yani eksen basligi ve tick etiketleri matplotlib varsayilani 10 pt -> baskida 3.8 pt; annotate `fontsize=7` -> baskida 2.7 pt. Karsilastirma: 6.results_and_discussion.tex'teki diger tum NB05 seki...

**Düzeltme:** Figuru basilacagi gercek olcude uret: `figsize=(7.2, 3.6)`; hucre basinda `plt.rcParams.update({"font.size":9,"axes.labelsize":9,"axes.titlesize":10,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8})`; annotate `fontsize=6.5`; `s=70` yerine `s=28`; sonra `width=\textwidth` ile ~1:1 basilir.

**Konum:** `Ayni hucreler - `plt.subplots(1, 2, figsize=(16, 7), sharey=True)`, `fontsize=7`, hic ayarlanmamis rcParams`


### [MAJOR] mae_spearman_vs_jct_gain_32gpu.png ve _256gpu.png (tez entegrasyonu) — caption-mismatch
**Sorun:** Sekil uretiliyor ve thesis/latex/figures/ altina yaziliyor ama tezin LaTeX kaynaginda HIC kullanilmiyor - altyazisi, numarasi ve metin referansi yok.

**Kanıt:** `grep -rn "mae_spearman" thesis/` -> hicbir eslesme yok (exit 1). thesis/latex/chapters/6.results_and_discussion.tex icinde 15 adet \includegraphics var, hepsini listeledim; hicbiri bu dosyayi cagirmiyor (nb04-fig01..05, nb05-fig01..05). Buna karsin ayni dosyada satir 396'da `\subsection{Evaluating Rank Correlation for Scheduling}` basligi ve satir 398'de tam olarak bu seklin argumanini metin olarak anlatan pasaj var ('...supports the DFL hypothesis: scheduling is primarily a sorting problem...'). Dosyalarin gercekten oraya yazildigini da dogruladim: thesis/latex/figures/mae_spearman_vs_jct_ga...

**Düzeltme:** 'Evaluating Rank Correlation for Scheduling' alt bolumune \begin{figure} bloguyla sekli ekle, \label{fig:rank-corr} ver ve metinden Figure~\ref{fig:rank-corr} ile referansla; altyaziya n=21, r ve p degerlerini ve baseline'larin fite dahil edildigini yaz; metindeki iddia gucunu altyaziyla hizala.

**Konum:** `Export: NB05 hucre 7a843cee sonundaki `for _out_dir in [...]: fig.savefig(...)`; eksik referans: /Users/hasanugurcelebi/Thesis/alibaba-gpu-runtime-prediction-and-scheduling/thesis/latex/chapters/6.results_and_discussion.tex satir 396-398`


### [MAJOR] mae_spearman_vs_jct_gain_32gpu.png ve _256gpu.png (kesikli siyah cizgi) — legend-error
**Sorun:** Sekilde hicbir lejant yok; kalin kesikli siyah cizginin ne oldugu (OLS uyum mu, referans cizgisi mi) hicbir yerde belirtilmiyor.

**Kanıt:** Kodda dogruladim: `ax.plot(xs, slope * xs + intercept, "k--", lw=1.5)` cagrisinda `label=` parametresi YOK ve hucrenin tamaminda `ax.legend(` / `fig.legend(` cagrisi HIC gecmiyor (hucre kaynagini bastan sona okudum). PNG'de her iki panelde kesikli siyah cizgi gorunuyor, hicbir lejant kutusu yok; tek aciklayici metin sol ustteki 'Pearson r = ...' kutusu ve okuyucunun cizgiyle bagini kendi kurmasi bekleniyor. Karsilastirma: ayni notebook'ta hucre 7cc8bbfc `axes[1].legend(loc="best", fontsize=8, frameon=True)` cagirıyor, yani lejant eklemek proje standardi disinda degil - bu hucrede unutulmus.

**Düzeltme:** Cizgiye label ver (`label=f"OLS fit (p={p_value:.3f})"` veya anlamsizsa '(n.s.)'), nokta gruplarina da label ekledikten sonra `axes[0].legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=8)` ya da `fig.legend(..., loc="lower center", ncol=4)` ile tek lejant koy.

**Konum:** `Ayni hucreler - `ax.plot(xs, slope * xs + intercept, "k--", lw=1.5)` (label yok), hucrede hic ax.legend() yok`


### [MAJOR] mae_spearman_vs_jct_gain_32gpu.png, mae_spearman_vs_jct_gain_256gpu.png — axis
**Sorun:** The right panel carries a full y-axis label but has no y-tick numbers.

**Kanıt:** Verified in the rendered PNG: the right panel shows the rotated label '32-GPU JCT Improvement over FIFO (%)' but its y-axis carries only bare tick marks, no numerals; the left panel has both. Source confirmed in cell#29: `fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)` and, inside the `for ax, x_col, xlabel, title in [...]` loop, `ax.set_ylabel(f"{N_GPU}-GPU JCT Improvement over FIFO (%)")` is called for both axes while sharey suppresses the right panel's tick labels. The same defect is visible in NB05_32GPU-Figure01.png.

**Düzeltme:** Move the ylabel out of the loop: call `axes[0].set_ylabel(...)` once after it.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb cell#29 (id=7a843cee), `ax.set_ylabel(...)` inside the per-axis loop`


### [MAJOR] mae_spearman_vs_jct_gain_32gpu.png, mae_spearman_vs_jct_gain_256gpu.png — truncation
**Sorun:** Point labels overflow past the axes frame into the inter-panel gutter.

**Kanıt:** Cropped the gutter region (px 950-1300 x 620-760 of the 2378x1034 file) and magnified it. The label 'CNN-LSTM (Numeric)' begins inside the left panel, crosses the left panel's right spine (clearly visible passing through the vertical black spine line in the crop) and is drawn in the empty gutter, ending immediately beside the right panel's rotated y-axis label '32-GPU JCT…'. The same happens on the right panel where 'ProfileMedian (baseline)' runs to the right edge. `bbox_inches="tight"` in cell#29's savefig only expands the canvas, it does not reflow.

**Düzeltme:** Add `ax.margins(x=0.12)` before the annotate loop and flip the offset/ha for extreme points.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb cell#29 (id=7a843cee), annotate loop with fixed `xytext=(4, 4)``


### [MINOR] mae_spearman_vs_jct_gain_32gpu.png ve _256gpu.png (sag panel x-ekseni + Pearson kutusu) — units
**Sorun:** Ayni panelde iki farkli seviyedeki iki korelasyon katsayisi ayirt edici aciklama olmadan gosteriliyor; x-ekseni basligi neyin neyle korele edildigini soylemiyor ve 'rho' Yunan harfi yerine duz metin. NOT: gercek bir sorun ama raporun verdigi CRITICAL/MAJOR agirligindan cok netlik/okunabilirlik seviyesinde - ben MINOR olarak derecelendiriyorum.

**Kanıt:** Kodda dogruladim: dongu tanimi `(axes[1], "Spearman rho", "Test-set Spearman rho", "Ranking Quality vs JCT Gain")` ve ayni eksende `ax.text(0.03, 0.94, f"Pearson r = {r_value:.3f}", ...)`. PNG'de sag panelin x-ekseni basligi duz metin 'Test-set Spearman rho' ($\rho$ degil) ve sol ustte 'Pearson r = 0.323' kutusu var. Birincisi model tahminleri ile gercek runtime arasindaki Spearman (rank_rows dongusunde `spearmanr(y_test_num, y_pred)` ile hesaplaniyor - dogruladim), ikincisi ise 21 modelin rho degerleri ile JCT kazanci arasindaki Pearson. Iki ayri seviye, ikisi de 'korelasyon', hicbir ayirt ed...

**Düzeltme:** x-ekseni basligini `r"Prediction-vs-actual Spearman $\rho$ (test set)"` yap; kutu metnini `f"Across-model Pearson r = {r:+.2f}\n(n={n}, p={p:.3f})"` seklinde seviye belirterek yaz; `ax.axvline(0, color="0.8", lw=0.8, zorder=0)` ile sifir referansini goster.

**Konum:** `Ayni hucreler - dongu tanimindaki xlabel dizesi ve `ax.text(0.03, 0.94, f"Pearson r = {r_value:.3f}", transform=ax.transAxes, ...)``


### [MINOR] mae_spearman_vs_jct_gain_32gpu.png vs mae_spearman_vs_jct_gain_256gpu.png (y-ekseni) — axis
**Sorun:** Yan yana okunmasi gereken iki olcek sekli farkli y-limit ve farkli tick araligi kullaniyor; ayni JCT yuzdesi iki sekilde farkli dikey yuksekliğe denk geliyor.

**Kanıt:** Kodu yeniden calistirip degerleri okudum: 32-GPU ylim = (-19.06, 67.05), yticks = [-20,-10,0,10,20,30,40,50,60,70] (10'ar birim); 256-GPU ylim = (-26.82, 65.51), yticks = [-40,-20,0,20,40,60,80] (20'ser birim). Raporun tarif ettigi fark birebir dogru. PNG'lerde de gorunuyor: 32-GPU sekli -10/0/10/... etiketleri, 256-GPU sekli -20/0/20/... etiketleri tasiyor. Kod nedeni: hucrelerde hicbir `ax.set_ylim(...)` veya `yaxis.set_major_locator(...)` cagrisi yok; sadece `sharey=True` var ve o yalnizca ayni figurdeki iki paneli baglar, iki ayri figuru degil.

**Düzeltme:** Her iki notebook'ta ortak sabit limit ve tick araligi: `axes[0].set_ylim(-30, 70)` + `axes[0].yaxis.set_major_locator(mticker.MultipleLocator(10))`; ayrica `ax.axhline(0, color="0.6", lw=0.9, zorder=0)` ile 'FIFO'dan kotu' bolgesini gorunur kil.

**Konum:** `Ayni hucreler (7a843cee / 8d1a4e10) - set_ylim / major_locator cagrisi yok`


### [MINOR] mae_spearman_vs_jct_gain_32gpu.png ve _256gpu.png (dosya formati) — dpi-quality
**Sorun:** Sekil yalnizca raster PNG olarak export ediliyor; tamamen cizgi-sanati olmasina ragmen vektorel (PDF/EPS) surum hic uretilmemis. NOT: raporun 'RGBA seffaflik artefakti' alt-iddiasi zayif - dosya RGBA ama alfa kanali tamamen opak.

**Kanıt:** Kodda tek export var: `fig.savefig(_out_dir / f"mae_spearman_vs_jct_gain_{N_GPU}gpu.png", dpi=150, bbox_inches="tight")` - baska format yok, facecolor belirtilmemis. `ls results/figures/thesis_export/` -> sadece 'html' ve 'png' klasorleri, 'pdf' yok; `find results/figures -name "*.pdf"` -> hicbir sonuc yok. PIL ile dosyalari inceledim: 2378x1034 px, mode=RGBA, dpi metadata 150. Efektif baski cozunurlugu 2378/6.10 in = 390 dpi (kabul edilebilir, cogu dergi line-art icin >=600 dpi veya vektor ister). Alfa kanali ekstremumlari (255, 255) - yani gercek bir seffaflik yok, bu alt-iddia abartili.

**Düzeltme:** Hem vektor hem yuksek-dpi raster uret: PNG icin dpi=600, ayrica ayni ada PDF kaydet ve `facecolor="white", transparent=False` ver; `plt.rcParams["pdf.fonttype"] = 42` ile fontu gom; LaTeX tarafinda uzantisiz cagir ki pdflatex PDF surumu secsin.

**Konum:** `Ayni hucreler - export dongusu `for _out_dir in [PROJECT_ROOT/"results"/"figures"/"thesis_export"/"png", PROJECT_ROOT/"thesis"/"latex"/"figures"]: fig.savefig(..., dpi=150, bbox_inches="tight")``


### [MINOR] mae_spearman_vs_jct_gain_32gpu.png ve _256gpu.png (suptitle) — caption-mismatch
**Sorun:** Ust baslik '21 Runtime Models' diyor ama sayilan 21 ogenin ikisi ogrenilmis model degil, ogrenilmemis heuristik baseline.

**Kanıt:** Kodda dogruladim: `fig.suptitle(f"Predictor Quality vs Scheduling Gain Across {len(_RAW_PREDS)} Runtime Models ({N_GPU}-GPU)", fontweight="bold")`. `_RAW_PREDS` sozlugunu saydim: tam 21 anahtar; bunlardan ikisi 'SJF-UserMedian (baseline)' (preds_user_median) ve 'SJF-ProfileMedian (baseline)' (preds_profile_median) - yani 19 ogrenilmis model + 2 heuristik. PNG'de baslik 'Predictor Quality vs Scheduling Gain Across 21 Runtime Models (32-GPU)' seklinde basili. Bu iki nokta ayni zamanda kaldirac noktasi (baseline'lar cikarilinca 32-GPU sag panelde r=+0.323 -> +0.416, kendim hesapladim), yani hepsi...

**Düzeltme:** Sayimi ayir: `_n_base = sum("baseline" in k for k in _RAW_PREDS); _n_learn = len(_RAW_PREDS) - _n_base` ve basligi `f"...: {_n_learn} Learned Models + {_n_base} Heuristic Baselines ({N_GPU}-GPU)"` yap; ya da model sayisini basliktan cikarip altyaziya tasi (dergi konvansiyonu).

**Konum:** `Ayni hucreler - `fig.suptitle(f"Predictor Quality vs Scheduling Gain Across {len(_RAW_PREDS)} Runtime Models ({N_GPU}-GPU)", fontweight="bold")``


### [MINOR] thesis/latex/figures/mae_spearman_vs_jct_gain_{32,256}gpu.png — other
**Sorun:** The two newest figures in the project sit in the thesis figure directory but are never included by any chapter.

**Kanıt:** grep -rn 'mae_spearman' over thesis/latex returns zero hits in any .tex file (only the two .png files exist on disk, dated Sep 6 09:03 and 09:05 — the newest figures in thesis/latex/figures). Counted the \\includegraphics per chapter: 3.dataset_and_workload.tex 9, 5.simulation_framework.tex 2, 6.results_and_discussion.tex 15, all others 0 — 26 total, none of them mae_spearman, exactly as claimed. Confirmed the orphaned argument: 6.results_and_discussion.tex:396 opens \\subsection{Evaluating Rank Correlation for Scheduling} and line 398 argues the rank-correlation-vs-scheduling-gain point entir...

**Düzeltme:** Add a two-panel figure environment in that subsection referencing both files with the Pearson r per panel — after fixing the label-overlap and missing-y-tick issues.

**Konum:** `thesis/latex/chapters/6.results_and_discussion.tex:396-398`


---

## Table — 2 bulgu (0 kritik)

### [MINOR] nb05-fig01..05-*_32gpu.png vs Table tab:predresults / chapter prose — naming-inconsistency
**Sorun:** The same model carries different names across figure, table and prose.

**Kanıt:** (a) CONFIRMED: every figure writes 'LGBM'; grep counts 14 occurrences of 'LightGBM' in the chapter prose, and Table 6.1 line 27 reads 'LightGBM (Native Cat.)'. (b) CONFIRMED: Table 6.1 uses 'One-Hot' (lines 26, 28, 34, 35, 36) while every figure uses '(Categorical)'; the papering-over parenthetical is verbatim at line 117: "(Models labeled ``One-Hot'' in Table~\ref{tab:predresults} appear as ``Categorical'' below; LGBM (Native Cat.) = SJF-LGBM (Categorical).)". (d) CONFIRMED: the current cell-35 render carries both 'SJF-XGBoost (Categorical)' and 'SJF-XGBoost (Native Cat)' on the same y axis, ...

**Düzeltme:** Define one canonical display name per model in the notebook (a POLICY_DISPLAY dict applied via `.replace()` to every plotted frame) and rewrite Table 6.1 with the same strings, so the explanatory parenthetical at line 117 can be deleted.

**Konum:** `thesis/latex/chapters/6.results_and_discussion.tex lines 22-45 and 117; notebooks/en/05_scheduler_evaluation_32_gpu.ipynb cells 35/41/44/54`


### [MINOR] Table \ref{tab:wilcoxon} / RQ1 text in 6.results_and_discussion.tex — other
**Sorun:** The same p-values are reported two incompatible ways, including a p-value of exactly zero.

**Kanıt:** 6.results_and_discussion.tex:405 reads 'The improvements were all statistically significant ($p = 0.000$).' The tab:wilcoxon body (lines 253-299) prints '$<0.001$' for every significant policy, and the prose at line 243 says '$p<0.001$'. All three verified by reading the lines directly. The exported HTML compounds it: NB05_32GPU_Table04.html prints the p-value column as literal '0.000000' for all 18 significant rows. The notebook itself already computes Holm-adjusted p-values and a rank-biserial effect size in cell#47 with an explicit comment that 'A p-value on 16k paired jobs is significant f...

**Düzeltme:** Change line 405 to '($p < 0.001$, Holm-adjusted)' and add the Holm column and effect sizes the notebook already computes to tab:wilcoxon.

**Konum:** `thesis/latex/chapters/6.results_and_discussion.tex:405 vs :253-299`


---

## diger — 8 bulgu (3 kritik)

### [CRITICAL] all five: nb05-fig01..05-*_32gpu.png — dpi-quality
**Sorun:** Font sizes are 3-5x below print legibility as the figures are actually placed in the document.

**Kanıt:** Every input to the arithmetic verified. thesis/latex/main.tex line 5 is `\documentclass[msc]{thesis}`; thesis.cls line 31 maps `msc` to `\@dtype=\@ne`, and lines 96-107 give that branch left=3.5cm, right=2cm on a4paper, so \textwidth = 21-5.5 = 15.5 cm = 6.10 in. Grepped the chapter: all ten NB05 includegraphics are at width=0.48\textwidth (lines 190, 192, 202, 204, 217, 219, 226, 228, 374, 376) = 2.93 in. Figsizes read from source: cell 35 (18,7), cell 38 (14,7), cell 41 (16,6), cell 44 (10,8), cell 54 (16,6). Scale factors 0.163 / 0.209 / 0.183 / 0.293 / 0.183 - so the 9 pt bar-value labels ...

**Düzeltme:** Stop shrinking wide figures: give each its own \includegraphics[width=\textwidth] with the 32-/256-GPU panels stacked as subfigures (a)/(b), or re-render at column geometry via rcParams (figure.figsize (6.1,4.0), font.size 8, savefig.dpi 300) and drop the per-call figsize overrides. Also save a PDF alongside each PNG so LaTeX embeds vector text.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb cells 35/38/41/44/54 (figsize); thesis/latex/chapters/6.results_and_discussion.tex lines 190-192, 202-204, 217-219, 226-228, 374-376`


### [CRITICAL] nb05-fig01..05-*_256gpu.png (LaTeX yerlestirmesi) — dpi-quality
**Sorun:** Cok genis uretilip cok kucuk yerlestirilen sekillerde tipografik olcek 0.16-0.29'a dusuyor; basimda yazi tipleri 1.5-2.9 pt.

**Kanıt:** thesis/latex/main.tex satir 5: \documentclass[msc]{thesis}. thesis.cls satir 31: msc -> \@dtype=\@ne; satir 96-106: a4paper, left=3.5cm, right=2cm => textwidth = 21-3.5-2 = 15.5 cm = 6.10 in. 6.results_and_discussion.tex satirlari 190/192, 202/204, 217/219, 226/228, 374/376 hepsi width=0.48\textwidth => 2.93 in. Defterdeki figsize'lari kodda dogruladim: cell 35 (18,7), cell 38 (14,7), cell 41 (16,6), cell 44 (10,8), cell 54 (16,6). Olcek faktorleri 2.93/18=0.163, 2.93/14=0.209, 2.93/16=0.183, 2.93/10=0.293 — bulgudaki degerlerle birebir. Cell 6'da hicbir rcParams/sns.set_theme override yok, ya...

**Düzeltme:** Ciftleri alt alta width=\textwidth ile ver ve figsize'lari (7.0,4.5) civarina indir; hucre basina yayin rcParams'i (font.size 8, tick 7, dpi 300) koy.

**Konum:** `thesis/latex/chapters/6.results_and_discussion.tex satir 190-192, 202-204, 217-219, 226-228, 374-376 + notebook cell 35/38/41/44/54 figsize`


### [CRITICAL] results/figures/thesis_export/png/* vs thesis/latex/figures/* — other
**Sorun:** The thesis figure directory holds figures from two different pipeline runs whose values disagree.

**Kanıt:** Timestamps and contents both confirm it. thesis/latex/figures/mae_spearman_vs_jct_gain_{32,256}gpu.png are dated Sep 6 09:03/09:05 and their title reads 'Predictor Quality vs Scheduling Gain Across 21 Runtime Models (32-GPU)' with UserMedian/ProfileMedian baselines present; thesis/latex/figures/nb05-fig01..05_*.png are dated Aug 31 15:41 and NB05_32GPU-Figure01.png (same pipeline position, extracted from the notebook JSON) is titled 'Across 18 Runtime Models' with no baselines. Values disagree as claimed: NB05_32GPU-Figure02.png / NB05_32GPU_Table02.html give SJF-LSTM (Categorical) 59.8% and S...

**Düzeltme:** Re-run both NB05 notebooks then scripts/export_thesis_results.py end to end; make _sync_thesis_figures raise instead of warn on a count mismatch; remove the direct fig.savefig into thesis/latex/figures from cell#29.

**Konum:** `scripts/export_thesis_results.py:78-110 (THESIS_FIGURE_MAP), :118-125 (EXPECTED_FIGURE_COUNT), :209-241 (_sync_thesis_figures); notebooks/en/05_scheduler_evaluation_32_gpu.ipynb cell#29 direct savefig`


### [MAJOR] nb05-fig01..05-*_32gpu.png (all five shipped thesis figures) — naming-inconsistency
**Sorun:** The shipped thesis figures are from a 21-policy run; the notebook now produces 28. Seven policies exist in the figures-to-be that the chapter never names, and one exported figure position is missing entirely.

**Kanıt:** Cell 26's POLICIES list contains 28 entries. I extracted the notebook's own stored PNG output for cell 35 and counted 28 bars, including SJF-AlibabaEstimate (baseline), SJF-AlibabaEstimate-Group (baseline), SJF-AlibabaEstimate-GroupGPU (baseline), SJF-LGBM (No Cluster-Load Features), SJF-XGBoost (Native Cat), SJF-UserMedian (baseline), SJF-ProfileMedian (baseline) - none of which appear in the shipped 21-bar nb05-fig01-scheduler-jct_32gpu.png. Values moved too: SJF-LSTM (Categorical) 59.8 % shipped vs 63.1 % current; SJF-CNN (Numeric Sequence) -13.0 % vs -15.2 %; FIFO 1344966s vs 1376710s. `gr...

**Düzeltme:** Re-run `python scripts/export_thesis_results.py` so all seven positions are emitted and the thesis figures refresh together, then name the new baselines in the chapter and Table 6.2 - or exclude them from the plotted set with an explicit POLICIES_FOR_THESIS filter if they are internal diagnostics. Do not ship the current mixed state.

**Konum:** `scripts/export_thesis_results.py (THESIS_FIGURE_MAP / EXPECTED_FIGURE_COUNT); notebooks/en/05_scheduler_evaluation_32_gpu.ipynb cells 26/31/35/38/41/44/54`


### [MAJOR] nb05-fig03-slowdown-box_*.png ile tez metni — caption-mismatch
**Sorun:** Tez metnindeki iki sayisal iddia da sekilden dogrulanmiyor.

**Kanıt:** 6.results_and_discussion.tex satir 213 aynen: 'Median slowdowns of categorical models are close to 100 in small-scale experiments using 32 GPUs (left); in large scale experiments using 256 GPUs (right), they drop below 2.0.' Sekilleri actim: NB05_256GPU-Figure04.png'de kategorik modellerin medyan cizgileri log eksende ~9-21 bandinda (SJF-LSTM (Categorical) ~9, SJF-LSTM (Cat Seq) ~11, SJF-XGBoost (Cat) ~12-13, SJF-CNN-LSTM (Cat) ~16, SJF-LGBM (Cat) ~20, SJF-CNN (Cat) ~21); 2.0'in altina inen TEK kutu SJF-Oracle (~1.5-1.7). NB05_32GPU-Figure04.png'de kategorik medyanlar ~200-370 bandinda (XGBoos...

**Düzeltme:** Metni sekilden okunan degerlerle degistir (ornegin '~2x10^2 -> ~1x10^1, yaklasik yirmi kat azalma; sadece SJF-Oracle 2'ye yaklasiyor') ve rakamlari cell 41 sonrasina eklenecek all_results.groupby('policy')['slowdown'].median() ciktisindan al.

**Konum:** `thesis/latex/chapters/6.results_and_discussion.tex satir 213; sekil kaynagi notebook cell 41`


### [MAJOR] nb05-fig02-wait-cdf_256gpu.png ile tez metni — caption-mismatch
**Sorun:** Tez metnindeki '%25' orani sekilde okunan degerin ~2 kati.

**Kanıt:** 6.results_and_discussion.tex satir 211 aynen: 'On the right side of Figure~\ref{fig:wait-cdf}, SJF-LSTM (Categorical) is found to schedule about 25\% of jobs as soon as it starts compared with the Oracle...'. Tezde kullanilan nb05-fig02-wait-cdf_256gpu.png (= NB05_256GPU-Figure03.png, md5 04e83421…) sekline baktim: x=1 s'de mavi SJF-LSTM (Categorical) egrisi CDF ~0.12'den basliyor, yesil SJF-Oracle ~0.31'den. ~0.25 seviyesine mavi egri ancak x~10^3 s civarinda ulasiyor. Defterin guncel cell-38 ciktisinda ise mavi ~0.10, yesil ~0.29. Her iki durumda da '%25' iddiasi sekille uyumsuz.

**Düzeltme:** Rakami defterden turet (cell 38'e (wait<=1).mean() hesabi ekle) ve metne '~12% vs Oracle 31%' gibi olculen degeri yaz; olcum esigini (<=1 s) metinde belirt.

**Konum:** `thesis/latex/chapters/6.results_and_discussion.tex satir 211; sekil kaynagi notebook cell 38 (id e5cd01)`


### [MAJOR] all 19 files in results/figures/thesis_export/html/ — other
**Sorun:** The generated HTML wrapper has no character-set declaration, so UTF-8 content mojibakes in Latin-1-defaulting consumers.

**Kanıt:** Scripted a grep -qi charset over every file in results/figures/thesis_export/html/: 19 of 19 have no charset declaration (19 total files). The header is confirmed at scripts/export_thesis_results.py:38-44 as `"<html><head><style>"` with no meta tag. Reproduced the consequence: `pandas.read_html` on NB05_32GPU_Table04.html returns the Significant? column as 'aˆš Yes (p<0.05)' — the ✅ decoded as mojibake. Affected characters verified present: ✅/❌ in NB05_*_Table04, 'R²' in every metric header, and 19 em dashes in NB04_Table07.html ('A — ML Numeric', 'B — ML Categorical'). NOTE: the second half o...

**Düzeltme:** Change HTML_STYLE_HEADER at line 38 to start `"<!DOCTYPE html><html><head><meta charset=\"utf-8\"><style>"`.

**Konum:** `scripts/export_thesis_results.py:38-44 (HTML_STYLE_HEADER)`


### [MINOR] notebook figure position 2 (load/backfill sensitivity) -> nb05-fig06-load-backfill-sensitivity_32gpu.png — legend-error
**Sorun:** The legend box is placed over the plotted data in the right panel, and it lists a series that duplicates the reference line.

**Kanıt:** Extracted and viewed the notebook's stored render of cell 31. In the right ('Backfilling enabled') panel the legend frame occupies the top-right quadrant and the black dashed SJF-Oracle curve runs underneath it at that end; a grey marker is visible protruding from behind the 'SJF-LGBM (Categorical)' legend row. `loc='best'` had no free quadrant because the Oracle trace runs across the top. The legend also lists FIFO, which the same render shows as a flat line at exactly 0 %, drawn directly on top of the `ax.axhline(0, color='grey', linewidth=1)` reference. Source confirms both the loc='best' c...

**Düzeltme:** Move the legend outside the axes (`fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8)` with `fig.tight_layout(rect=[0, 0, 0.82, 1])`) and drop the redundant FIFO series.

**Konum:** `notebooks/en/05_scheduler_evaluation_32_gpu.ipynb, cell index 31 (id=7cc8bbfc): `axes[1].legend(loc='best', fontsize=8, frameon=True)``


---
