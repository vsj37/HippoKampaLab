import streamlit as st
import numpy as np
import h5py
import re
import io
import os
import tempfile

import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

# Optional imports — only used if the user has these installed
try:
    import igor2.binarywave as bw
    HAS_IBW = True
except ImportError:
    HAS_IBW = False

try:
    import heka_reader
    HAS_HEKA = True
except ImportError as _heka_err:
    HAS_HEKA = False
    _heka_import_error = str(_heka_err)
else:
    _heka_import_error = None

# ─── Helper ────────────────────────────────────────────────────────────────────

def exp_growth(t, tau, v_steady, v_start):
    return v_steady + (v_start - v_steady) * np.exp(-t / tau)

def nat_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', s)]

# ─── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Mayalaran", layout="wide")
st.title("🧠 Mayalaran — Patch Clamp Browser")

# ─── Session state init ────────────────────────────────────────────────────────

defaults = dict(
    sweep_data=None,
    time_vec=None,
    x_scale=0.02,
    current_index=0,
    total_sweeps=0,
    mode=None,           # 'ibw' | 'dat' | 'h5' | 'heka_tree'
    file_bytes=None,
    file_name=None,
    h5_keys=[],
    heka_obj=None,
    series_start_sweep=0,
    series_total=0,
    fig_extra_traces=[],  # list of (x, y, color, name) for overlays
    key_nav=0,             # incremented by keyboard JS to trigger rerun
    passive_results=None,  # stores RMP, Rin, Tau between reruns
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Data Source")

    file_type = st.radio("Format", ["Igor (.ibw)", "HEKA (.dat)", "HDF5 Master (.h5)"])

    uploaded = st.file_uploader(
        "Upload file",
        type=["ibw", "dat", "h5", "hdf5"],
        key="uploader",
    )

    if uploaded and st.button("Load File"):
        st.session_state.file_bytes = uploaded.read()
        st.session_state.file_name = uploaded.name
        st.session_state.current_index = 0
        st.session_state.fig_extra_traces = []

        if file_type == "Igor (.ibw)":
            if not HAS_IBW:
                st.error("igor2 not installed. Run: pip install igor2")
            else:
                try:
                    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.ibw')
                    os.close(tmp_fd)
                    with open(tmp_path, 'wb') as f:
                        f.write(st.session_state.file_bytes)
                    data = bw.load(tmp_path)
                    raw = data['wave']['wData']
                    st.session_state.x_scale = data['wave']['wave_header']['sfA'][0]
                    sweep = raw[:, 0] if raw.ndim == 2 else raw
                    st.session_state.sweep_data = sweep
                    st.session_state.time_vec = np.arange(len(sweep)) * st.session_state.x_scale
                    st.session_state.total_sweeps = 1
                    st.session_state.mode = 'ibw'
                except Exception as e:
                    st.error(f"IBW load error: {e}")

        elif file_type == "HDF5 Master (.h5)":
            suffix = os.path.splitext(uploaded.name)[1]
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.close(tmp_fd)  # Close the fd first, then write via open()
            with open(tmp_path, 'wb') as f:
                f.write(st.session_state.file_bytes)
            with h5py.File(tmp_path, 'r') as f:
                keys = sorted(list(f.keys()), key=nat_sort_key)
            st.session_state.h5_keys = keys
            st.session_state.total_sweeps = len(keys)
            st.session_state.mode = 'h5'
            st.session_state.h5_path = tmp_path

        elif file_type == "HEKA (.dat)":
            if not HAS_HEKA:
                st.error(f"heka_reader not found: {_heka_import_error}. Place heka_reader.py in the same folder as app_2.py")
            else:
                try:
                    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.dat')
                    os.close(tmp_fd)  # Close fd before writing
                    with open(tmp_path, 'wb') as f:
                        f.write(st.session_state.file_bytes)
                    st.session_state.heka_path = tmp_path
                    st.session_state.heka_obj = heka_reader.Bundle(tmp_path)
                    st.session_state.mode = 'heka_tree'
                    st.success(f"Loaded: {uploaded.name} — expand groups in the tree below to select a series.")
                except Exception as e:
                    st.error(f"DAT load error: {e}")

        st.rerun()

    # ── HEKA Tree ──────────────────────────────────────────────────────────────
    if st.session_state.mode == 'heka_tree' and st.session_state.heka_obj is not None:
        st.subheader("HEKA Hierarchy")
        heka = st.session_state.heka_obj
        cumulative = 0
        for g_idx, group in enumerate(heka.pul):
            with st.expander(f"Group {g_idx}: {group.Label}"):
                for s_idx, series in enumerate(group):
                    n = len(series)
                    label = f"{series.Label.strip()} ({n} swp)"
                    if st.button(label, key=f"series_{g_idx}_{s_idx}"):
                        st.session_state.series_start_sweep = cumulative
                        st.session_state.total_sweeps = n
                        st.session_state.series_total = n
                        st.session_state.current_index = 0
                        st.session_state.fig_extra_traces = []
                        st.rerun()
                    cumulative += n

    st.divider()

    # ── Navigation ─────────────────────────────────────────────────────────────
    st.subheader("🔀 Navigation")
    total = st.session_state.total_sweeps
    if total > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("◀ Prev", key="btn_prev") and st.session_state.current_index > 0:
                st.session_state.current_index -= 1
                st.session_state.fig_extra_traces = []
                st.rerun()
        with col2:
            st.write(f"{st.session_state.current_index + 1} / {total}")
        with col3:
            if st.button("Next ▶", key="btn_next") and st.session_state.current_index < total - 1:
                st.session_state.current_index += 1
                st.session_state.fig_extra_traces = []
                st.rerun()

        jump = st.number_input("Jump to sweep", min_value=1, max_value=max(total, 1), step=1)
        if st.button("🎯 Jump"):
            st.session_state.current_index = int(jump) - 1
            st.session_state.fig_extra_traces = []
            st.rerun()

    st.divider()

    # ── Passive Analysis ───────────────────────────────────────────────────────
    st.subheader("📐 Passive Analysis")
    i_inj = st.number_input("Current Injection (pA)", value=-100.0)
    base_start = st.number_input("Baseline Start (ms)", value=1750.0)
    base_end = st.number_input("Baseline End (ms)", value=2250.0)
    rc_start = st.number_input("RC Fit Start (ms)", value=85.0)
    rc_end = st.number_input("RC Fit End (ms)", value=145.0)
    run_passive = st.button("🚀 Run Passive Analysis")

    # Persistent results box — survives reruns
    if st.session_state.passive_results:
        r = st.session_state.passive_results
        st.markdown(
            f"""
            <div style="background:#1a1a2e;border:1px solid #3498db;border-radius:6px;padding:10px;font-family:monospace;font-size:13px;color:#3498db;">
            <b>RMP:</b> {r['rmp']:.2f} mV<br>
            <b>R_in:</b> {r['rin']:.1f} MΩ<br>
            <b>Tau:</b> {r['tau']:.1f} ms
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Active Analysis ────────────────────────────────────────────────────────
    st.subheader("⚡ Active (AP) Analysis")
    dvdt_thresh = st.number_input("dV/dt Threshold (V/s)", value=50.0)
    run_active = st.button("⚡ Run Active Analysis")

# ─── Load current sweep data ───────────────────────────────────────────────────

def load_current_sweep():
    mode = st.session_state.mode
    idx = st.session_state.current_index

    if mode == 'ibw':
        return  # Already loaded at file-upload time

    elif mode == 'h5':
        key = st.session_state.h5_keys[idx]
        with h5py.File(st.session_state.h5_path, 'r') as f:
            st.session_state.sweep_data = f[key][:]
            st.session_state.x_scale = float(f[key].attrs.get('dx', 0.02))
        st.session_state.time_vec = (
            np.arange(len(st.session_state.sweep_data)) * st.session_state.x_scale
        )

    elif mode in ('heka_tree', 'dat'):
        fs = 20000
        offset = 1024
        samples = int((2500 / 1000) * fs)
        global_idx = st.session_state.series_start_sweep + idx
        skip = offset + global_idx * samples * 4
        with open(st.session_state.heka_path, 'rb') as f:
            f.seek(skip)
            trace = np.fromfile(f, dtype=np.float32, count=samples)
        if len(trace) == samples:
            st.session_state.sweep_data = trace * 1000  # V → mV
            st.session_state.x_scale = 1000 / fs
            st.session_state.time_vec = (
                np.arange(len(st.session_state.sweep_data)) * st.session_state.x_scale
            )


if st.session_state.mode in ('h5', 'heka_tree', 'dat'):
    load_current_sweep()

# ─── Build base plot ───────────────────────────────────────────────────────────

sweep = st.session_state.sweep_data
time = st.session_state.time_vec

if sweep is not None and time is not None:
    color = '#2ecc71' if st.session_state.mode == 'h5' else 'white'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time, y=sweep,
        mode='lines',
        line=dict(color=color, width=1),
        name='Sweep',
    ))

    # Overlay traces (from analyses)
    for (ox, oy, oc, oname) in st.session_state.fig_extra_traces:
        fig.add_trace(go.Scatter(
            x=ox, y=oy,
            mode='lines+markers' if len(ox) <= 2 else 'lines',
            line=dict(color=oc, width=2),
            marker=dict(size=10, color=oc),
            name=oname,
        ))

    fig.update_layout(
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(title='Time (ms)', color='white', gridcolor='#333'),
        yaxis=dict(title='Voltage (mV)', color='white', gridcolor='#333'),
        margin=dict(l=60, r=20, t=30, b=50),
        height=500,
        showlegend=True,
    )

    st.plotly_chart(fig, width='stretch')

    # ── Export ──────────────────────────────────────────────────────────────────
    fname = (
        st.session_state.h5_keys[st.session_state.current_index]
        if st.session_state.mode == 'h5'
        else f"Sweep_{st.session_state.current_index + 1}"
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        try:
            svg_bytes = fig.to_image(format='svg')
            st.download_button("🎨 Download SVG", data=svg_bytes,
                               file_name=f"{fname}.svg", mime="image/svg+xml")
        except Exception:
            st.warning("SVG export needs kaleido: `pip install kaleido`")
    with col_b:
        try:
            jpg_bytes = fig.to_image(format='jpg', width=1600, height=800, scale=2)
            st.download_button("🖼 Download JPG", data=jpg_bytes,
                               file_name=f"{fname}.jpg", mime="image/jpeg")
        except Exception:
            st.warning("JPG export needs kaleido: `pip install kaleido`")
    with col_c:
        # Always available fallback — no kaleido needed
        html_bytes = fig.to_html(include_plotlyjs='cdn').encode('utf-8')
        st.download_button("🌐 Download HTML", data=html_bytes,
                           file_name=f"{fname}.html", mime="text/html")

    # ── Passive Analysis ────────────────────────────────────────────────────────
    if run_passive:
        try:
            xs = st.session_state.x_scale
            fs_ms = 1.0 / xs

            b0 = int(base_start * fs_ms)
            b1 = int(base_end * fs_ms)
            rmp = np.mean(sweep[b0:b1])

            f0 = int(rc_start * fs_ms)
            f1 = int(rc_end * fs_ms)
            y_data = sweep[f0:f1]
            x_data = np.linspace(0, rc_end - rc_start, len(y_data))

            popt, _ = curve_fit(
                exp_growth, x_data, y_data,
                p0=[20, np.mean(y_data[-10:]), rmp],
                maxfev=10000,
            )

            r_in = abs((popt[1] - popt[2]) / i_inj) * 1000
            tau = popt[0]

            # Store results persistently so they survive rerun
            st.session_state.passive_results = {'rmp': rmp, 'rin': r_in, 'tau': tau}

            # Add fit overlay
            x_fit = x_data + rc_start
            y_fit = exp_growth(x_data, *popt)
            st.session_state.fig_extra_traces.append(
                (x_fit.tolist(), y_fit.tolist(), 'red', 'RC Fit')
            )
            st.rerun()

        except Exception as e:
            st.error(f"Passive analysis error: {e}")

    # ── Active Analysis ─────────────────────────────────────────────────────────
    if run_active:
        try:
            xs = st.session_state.x_scale
            fs_khz = 1.0 / xs

            wl = 11 if fs_khz >= 40 else 31
            v_smooth = savgol_filter(sweep, window_length=wl, polyorder=3)
            dvdt = np.diff(v_smooth) * fs_khz

            peaks, _ = find_peaks(sweep, height=0, distance=int(2.0 * fs_khz))

            rows = []
            thresh_xs, thresh_ys = [], []
            peak_xs, peak_ys = [], []

            for p in peaks:
                lookback = int(2.0 * fs_khz)
                ws = max(0, p - lookback)
                window_dvdt = dvdt[ws:p]
                idx_over = np.where(window_dvdt >= dvdt_thresh)[0]

                if len(idx_over) > 0:
                    t_idx = ws + idx_over[0]
                    v_thresh = sweep[t_idx]
                    v_peak = sweep[p]
                    amp = v_peak - v_thresh
                    v_half = v_thresh + amp / 2

                    hw_start = max(0, p - int(1.0 * fs_khz))
                    hw_end = min(len(sweep), p + int(4.0 * fs_khz))
                    over_half = np.where(sweep[hw_start:hw_end] >= v_half)[0]
                    hw = (over_half[-1] - over_half[0]) * xs if len(over_half) > 1 else 0

                    rows.append({
                        "Peak (mV)": round(float(v_peak), 2),
                        "Threshold (mV)": round(float(v_thresh), 2),
                        "Upstroke (V/s)": round(float(np.max(window_dvdt)), 1),
                        "Half-Width (ms)": round(float(hw), 3),
                    })
                    thresh_xs.append(t_idx * xs)
                    thresh_ys.append(float(v_thresh))
                    peak_xs.append(p * xs)
                    peak_ys.append(float(v_peak))

            if rows:
                import pandas as pd
                st.dataframe(pd.DataFrame(rows), width='stretch')

                # Add markers as scatter overlays
                st.session_state.fig_extra_traces.append(
                    (thresh_xs, thresh_ys, 'blue', 'AP Threshold')
                )
                st.session_state.fig_extra_traces.append(
                    (peak_xs, peak_ys, 'red', 'AP Peak')
                )
                st.rerun()
            else:
                st.warning("No spikes detected with current threshold.")

        except Exception as e:
            st.error(f"Active analysis error: {e}")

    # ─── Keyboard shortcut JS (clicks the Prev/Next sidebar buttons) ──────────────
    import streamlit.components.v1 as components
    components.html("""
    <script>
    (function() {
        // Remove stale listener from previous render
        if (window._patchKeyHandler) {
            window.parent.document.removeEventListener('keydown', window._patchKeyHandler);
        }

        function clickButton(label) {
            // Search all buttons in the parent document for matching text
            const btns = window.parent.document.querySelectorAll('button[kind="secondary"], button');
            for (const btn of btns) {
                if (btn.innerText.trim() === label) {
                    btn.click();
                    return true;
                }
            }
            return false;
        }

        window._patchKeyHandler = function(e) {
            // Ignore keypresses when typing in inputs
            const active = window.parent.document.activeElement;
            const tag = active ? active.tagName.toLowerCase() : '';
            if (tag === 'input' || tag === 'textarea' || tag === 'select') return;

            if (e.key === 'ArrowRight') {
                e.preventDefault();
                clickButton('Next ▶');
            } else if (e.key === 'ArrowLeft') {
                e.preventDefault();
                clickButton('◀ Prev');
            }
        };

        window.parent.document.addEventListener('keydown', window._patchKeyHandler);
    })();
    </script>
    """, height=0)

else:
    st.info("👆 Upload a file using the sidebar to get started.")
    st.markdown("""
    **Supported formats:**
    - **Igor Binary Wave (.ibw)** — requires `igor2`
    - **HEKA (.dat)** — requires `heka_reader`
    - **HDF5 Master (.h5 / .hdf5)** — requires `h5py`

    **Keyboard shortcuts:** `←` / `→` to navigate sweeps.
    """)
