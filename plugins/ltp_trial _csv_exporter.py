"""
ltp_cc_averager.py
──────────────────
Kaboom plugin to analyze and bin Current-Clamp EPSP amplitudes for H5 LTP files.
Averages blocks of 6 sweeps, normalizes to baseline, scales the X-axis dynamically,
allows explicit exclusion of artifact sweeps, and AUTOMATICALLY saves a clean CSV.
"""

NAME        = "LTP CC H5 Averager (With CSV Export - gemini)"
DESCRIPTION = "Bins 6 sweeps, normalizes, handles exclusions, and exports a standalone CSV for group data assembly."

# ── Parameter dialog ─────────────────────────────────────────────────────────
PARAMS = {
    "baseline_start_ms" : 200,   
    "baseline_end_ms"   : 400,   
    "epsp_start_ms"     : 538,   
    "epsp_end_ms"       : 580,   
    "iti_seconds"       : 14,    
    "baseline_first_sw" : 82,    # Shifting forward to skip run-up/polarization by default
    "baseline_last_sw"  : 102,    
    "post_first_sw"     : 108,    
    "post_last_sw"      : 250,    
    "exclude_sweeps"    : "134, 136, 143, 145-146",     # e.g., "142, 150-152"
}

# ── Entry point ───────────────────────────────────────────────────────────────

def run(ctx, **p):
    import numpy as np
    import traceback
    import h5py
    import os

    ctx.set_results("Starting H5 plugin execution...")

    try:
        if not ctx.file_path:
            ctx.set_results("Error: No data file is currently loaded in Kaboom.")
            return

        # ── 1. Read parameters ───────────────────────────────────────────────
        bl_start   = float(p["baseline_start_ms"])
        bl_end     = float(p["baseline_end_ms"])
        epsp_start = float(p["epsp_start_ms"])
        epsp_end   = float(p["epsp_end_ms"])
        iti_sec    = float(p["iti_seconds"])
        
        b_start_user = int(p["baseline_first_sw"])
        b_end_user   = int(p["baseline_last_sw"])
        p_start_user = int(p["post_first_sw"])
        p_end_user   = int(p["post_last_sw"])
        exclude_str  = str(p["exclude_sweeps"]).strip()

        bin_duration_min = (6.0 * iti_sec) / 60.0

        # ── 2. Parse Exclusion List ─────────────────────────────────────────
        excluded_sweeps = set()
        if exclude_str:
            parts = [part.strip() for part in exclude_str.split(",") if part.strip()]
            for part in parts:
                if "-" in part:
                    try:
                        start_range, end_range = part.split("-")
                        for sw in range(int(start_range), int(end_range) + 1):
                            excluded_sweeps.add(sw)
                    except:
                        ctx.log(f"Warning: Could not parse range '{part}'")
                else:
                    try:
                        excluded_sweeps.add(int(part))
                    except:
                        ctx.log(f"Warning: Could not parse sweep '{part}'")

        # ── 3. H5 Native Loading Loop ────────────────────────────────────────
        app = ctx._app
        keys = app.h5_keys
        
        all_raw = []
        dt_ms = None
        
        with h5py.File(app.file_list[0], 'r') as f:
            for key in keys:
                d = f[key][:]
                if dt_ms is None:
                    dt_ms = float(f[key].attrs.get('dx', 0.02))  
                all_raw.append(np.asarray(d, dtype=float))
        
        n_sweeps = len(all_raw)
        
        # Pad variable length sweeps safely if needed
        max_len = max(len(d) for d in all_raw)
        matrix = np.array([
            np.pad(d, (0, max_len - len(d)), constant_values=np.nan)
            if len(d) < max_len else d for d in all_raw
        ], dtype=float)

        bl_first_idx   = max(0, b_start_user - 1)
        bl_last_idx    = min(b_end_user, n_sweeps)
        post_first_idx = max(0, p_start_user - 1)
        post_last_idx  = min(p_end_user, n_sweeps)

        def ms_to_idx(ms):
            return int(max(0, min(round(ms / dt_ms), matrix.shape[1] - 1)))

        idx_bl_s, idx_bl_e = ms_to_idx(bl_start), ms_to_idx(bl_end)
        idx_epsp_s, idx_epsp_e = ms_to_idx(epsp_start), ms_to_idx(epsp_end)

        # ── 4. Calculate Individual Sweep Amplitudes ────────────────────
        raw_amplitudes = []
        for i in range(n_sweeps):
            sweep_num = i + 1  
            if sweep_num in excluded_sweeps:
                raw_amplitudes.append(np.nan)
                continue
                
            trace = matrix[i]
            bl_seg = trace[idx_bl_s:idx_bl_e]
            bl_val = np.mean(bl_seg[~np.isnan(bl_seg)]) if len(bl_seg[~np.isnan(bl_seg)]) > 0 else 0.0
            
            epsp_seg = trace[idx_epsp_s:idx_epsp_e]
            valid_epsp = epsp_seg[~np.isnan(epsp_seg)]
            
            if len(valid_epsp) == 0:
                raw_amplitudes.append(0.0)
            else:
                raw_amplitudes.append(float(np.max(valid_epsp) - bl_val))

        bl_raw = raw_amplitudes[bl_first_idx:bl_last_idx]
        post_raw = raw_amplitudes[post_first_idx:post_last_idx]

        # ── 5. Averaging Chunks (Bin size = 6 sweeps) ────────────────────────
        def bin_sweeps(raw_list):
            binned = []
            for i in range(0, len(raw_list), 6):
                chunk = raw_list[i:i+6]
                valid_chunk = [v for v in chunk if not np.isnan(v)]
                binned.append(np.mean(valid_chunk) if len(valid_chunk) > 0 else np.nan)
            return binned

        bl_binned = bin_sweeps(bl_raw)
        post_binned = bin_sweeps(post_raw)

        valid_bl_raw = [v for v in bl_raw if not np.isnan(v)]
        if len(valid_bl_raw) == 0:
            ctx.set_results("Error: No valid baseline data remains to normalize.")
            return
        baseline_mean_mv = np.mean(valid_bl_raw)

        bl_normalized = [(v / baseline_mean_mv) * 100.0 if not np.isnan(v) else np.nan for v in bl_binned]
        post_normalized = [(v / baseline_mean_mv) * 100.0 if not np.isnan(v) else np.nan for v in post_binned]
        all_normalized = bl_normalized + post_normalized

        # Construct time axes
        bl_times = [(i - len(bl_binned)) * bin_duration_min for i in range(len(bl_binned))]
        post_times = [i * bin_duration_min for i in range(len(post_binned))]
        all_times = bl_times + post_times

        # Clean NaN points for Pyqtgraph mapping
        plot_bl_t = [t for t, v in zip(bl_times, bl_normalized) if not np.isnan(v)]
        plot_bl_v = [v for v in bl_normalized if not np.isnan(v)]
        plot_post_t = [t for t, v in zip(post_times, post_normalized) if not np.isnan(v)]
        plot_post_v = [v for v in post_normalized if not np.isnan(v)]

        # ── 6. Generate Window Plot ───────────────────────────────────────────
        win = ctx.new_plot_window(title=f"LTP Plot: {ctx.file_name}", rows=1, cols=1, size=(850, 450))
        win.plot(plot_bl_t, plot_bl_v, color=(100, 150, 255), width=2, label="Baseline")
        win.scatter(plot_bl_t, plot_bl_v, color=(100, 150, 255), size=8)
        win.plot(plot_post_t, plot_post_v, color=(255, 90, 90), width=2, label="Post-Induction")
        win.scatter(plot_post_t, plot_post_v, color=(255, 90, 90), size=8)
        win.hline(100.0, color=(128, 128, 128), style='dash')
        win.vline(0.0, color=(255, 255, 100), style='solid')
        win.set_labels(title=f"Normalized Cell Profile: {ctx.file_name}", xlabel="Time (minutes)", ylabel="Amplitude (%)")
        win.auto_range()

        # ── 7. AUTOMATIC CSV EXPORT ──────────────────────────────────────────
        # Targets the same directory where your active loaded H5 file lives
        source_dir = os.path.dirname(ctx.file_path)
        csv_name = ctx.file_name.replace(".h5", "_analyzed.csv")
        csv_out_path = os.path.join(source_dir, csv_name)

        with open(csv_out_path, 'w') as csv_file:
            csv_file.write("Time (min),Normalized Amplitude (%)\n")
            for t, v in zip(all_times, all_normalized):
                v_str = f"{v:.4f}" if not np.isnan(v) else "" # Empty string if excluded
                csv_file.write(f"{t:.2f},{v_str}\n")

        # ── 8. Print Results Summary ─────────────────────────────────────────
        lines = [
            "=== CELL ANALYSIS COMPLETE ===",
            f"Analyzed File     : {ctx.file_name}",
            f"Baseline Mean     : {baseline_mean_mv:.4f} mV",
            f"CSV Auto-Exported : {csv_out_path}",
            "",
            "This CSV contains a clean single-column profile of this cell.",
            "You can easily open it in Excel or copy it into your group summary!",
            "",
            f"{'Time (min)':<12} {'Amp (% Base)':>14} {'Zone':>10}",
            "-" * 40
        ]
        for t, v in zip(all_times[::2], all_normalized[::2]): # Show every second point to keep panel brief
            zone = "Baseline" if t < 0 else "Post-Ind"
            v_str = f"{v:.2f}%" if not np.isnan(v) else "EXCLUDED"
            lines.append(f"{t:<12.2f} {v_str:>14} {zone:>10}")

        ctx.set_results("\n".join(lines))

    except Exception as e:
        ctx.set_results(f"CRITICAL SCRIPT CRASH:\n{traceback.format_exc()}")