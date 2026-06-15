import sys
import glob
import importlib.util
import os
import re
import math
import tempfile
import numpy as np
import h5py
import igor2.binarywave as bw
import heka_reader
import neo

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QLabel,
                             QLineEdit, QGroupBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QTreeWidget, QTreeWidgetItem, QScrollArea,
                             QTextEdit, QCheckBox, QDialog, QDialogButtonBox,
                             QSpinBox, QDoubleSpinBox, QFormLayout, QSplitter)
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QKeySequence, QShortcut, QMovie
from PyQt6.QtCore import Qt, QUrl, QTimer, QByteArray, QBuffer
import pyqtgraph as pg
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter, iirnotch, sosfiltfilt, tf2sos
import pyqtgraph.exporters
import warnings as _warnings


# ══════════════════════════════════════════════════════════════════════════════
# EPHYS ANALYSIS ENGINE  (converted from MATLAB)
# ══════════════════════════════════════════════════════════════════════════════

def _ephys_mono_exp(x, a, b, c):
    return c + a * np.exp(-b * x)

def _ephys_lowpass(signal, cutoff, fs, order=5):
    from scipy.signal import butter, sosfiltfilt as _sf
    sos = butter(order, cutoff / (fs / 2), btype='low', output='sos')
    return _sf(sos, signal)

def _ephys_build_notch(sf):
    sos_list = []
    for freq in [50.0, 100.0, 150.0]:
        if freq < sf / 2:
            b, a = iirnotch(freq, Q=30, fs=sf)
            sos_list.append(tf2sos(b, a))
    return np.vstack(sos_list)

def run_ephys_analysis(v, t, csteps,
                       stepstart_t=0.25, stepend_t=0.75, safety_t=0.001,
                       steadystate_factor=0.25, ctau=-50,
                       AP_dv_dt_threshold=40, AP_thresh_std_factor=3,
                       AP_dur_t=0.003, fAHP_dur_t=0.01):
    """
    Core ephys analysis. v: (samples, sweeps) mV array, t: seconds 1-D array.
    csteps: 1-D array of current steps in pA, matching v columns.
    Returns a dict with all computed quantities + internal arrays for plotting.
    """
    sf        = 1.0 / (t[1] - t[0])
    stepstart = round(stepstart_t * sf)
    stepend   = round(stepend_t   * sf)
    safety    = round(safety_t    * sf)
    AP_dur    = round(AP_dur_t    * sf)
    fAHP_dur  = round(fAHP_dur_t  * sf)
    steplength        = stepend - stepstart
    steadystate_start = round(stepend - steplength * steadystate_factor)

    sos_notch = _ephys_build_notch(sf)
    zero_idx  = int(np.searchsorted(csteps, 0))
    for fi in range(min(zero_idx + 1, v.shape[1])):
        v[:, fi] = sosfiltfilt(sos_notch, v[:, fi])

    RMP              = np.mean(v[:stepstart - safety, :], axis=0)
    mean_RMP         = float(np.mean(RMP))
    steadystate_mean = np.mean(v[steadystate_start:stepend - safety, :], axis=0)
    dv_steadystate   = steadystate_mean - RMP

    idx_m100  = int(np.searchsorted(csteps, -100))
    idx_p100  = int(np.searchsorted(csteps,  100))
    fit_steps = csteps[idx_m100:idx_p100 + 1]
    fit_dv    = dv_steadystate[idx_m100:idx_p100 + 1]
    P_R_in    = np.polyfit(fit_steps, fit_dv, 1)
    R_in      = float(P_R_in[0] * 1000)

    taustep = int(np.searchsorted(csteps, ctau))
    t_seg   = t[stepstart:stepend] - t[stepstart]
    v_seg   = v[stepstart:stepend, taustep]
    try:
        dV_g = v_seg[0] - v_seg[-1]
        popt, _ = curve_fit(_ephys_mono_exp, t_seg, v_seg,
                            p0=[dV_g, 1.0 / 0.02, v_seg[-1]], maxfev=10000)
        tau_m    = float(1.0 / popt[1])
        fit_vals = _ephys_mono_exp(t_seg, *popt)
    except RuntimeError:
        tau_m    = float('nan')
        fit_vals = np.full_like(t_seg, float('nan'))

    v_seg_safe = v[stepstart + safety:stepend - safety, taustep]
    half       = round(len(v_seg_safe) / 2)
    smoothed   = np.convolve(v_seg_safe[:half], np.ones(5) / 5, mode='valid')
    min_peak_x = int(np.argmin(smoothed))
    min_peak_y = float(smoothed[min_peak_x])
    sag_ratio  = float((min_peak_y - RMP[taustep]) / (steadystate_mean[taustep] - RMP[taustep]))

    AP_found    = False
    rheobase    = None
    AP_thresh_x = np.full((200, v.shape[1]), np.nan)
    for s in range(v.shape[1]):
        vs      = v[stepstart + safety:stepend - safety, s]
        vs_lp   = _ephys_lowpass(vs, 5000, sf)
        dv_dt   = np.diff(vs_lp) * sf / 1000
        d2v_dt2 = np.diff(vs_lp, n=2)
        hits    = np.where(dv_dt >= AP_dv_dt_threshold)[0]
        if len(hits) == 0:
            continue
        AP_detect = hits[0]
        if not AP_found:
            AP_found = True
            rheobase = csteps[s]
        sf_ms = round(sf / 1000)
        start = max(0, AP_detect - sf_ms)
        crit  = d2v_dt2[start:] >= np.mean(d2v_dt2) + AP_thresh_std_factor * np.std(d2v_dt2)
        AP_thresh_x[0, s] = (np.argmax(crit) + start) if np.any(crit) else AP_detect
        AP_peak_x = np.argmax(vs[AP_detect:min(AP_detect + AP_dur * 2, len(vs))]) + AP_detect
        AP_count  = 1
        currpoint = AP_peak_x
        next_hits = np.where(dv_dt[currpoint:] >= AP_dv_dt_threshold)[0]
        while len(next_hits) > 0:
            AP_detect2 = next_hits[0] + currpoint
            if AP_detect2 > len(d2v_dt2) - AP_dur:
                break
            AP_peak_x2 = np.argmax(vs[AP_detect2:min(AP_detect2 + AP_dur * 2, len(vs))]) + AP_detect2
            start2 = max(0, AP_detect2 - sf_ms)
            seg    = d2v_dt2[start2:AP_peak_x2]
            if len(seg) == 0:
                break
            crit2 = seg >= np.mean(d2v_dt2) + AP_thresh_std_factor * np.std(d2v_dt2)
            AP_thresh_x[AP_count, s] = (np.argmax(crit2) + start2) if np.any(crit2) else (np.argmax(seg) + start2)
            AP_count  += 1
            currpoint  = AP_peak_x2
            next_hits  = np.where(dv_dt[currpoint:] >= AP_dv_dt_threshold)[0]

    if rheobase is None:
        raise ValueError('No APs detected in any sweep.')

    rheo_idx     = int(np.searchsorted(csteps, rheobase))
    vs           = v[stepstart + safety:stepend - safety, rheo_idx]
    AP1_thresh_x = int(AP_thresh_x[0, rheo_idx])
    AP1_thresh_y = float(vs[AP1_thresh_x])
    end_ap       = min(AP1_thresh_x + AP_dur, len(vs))
    AP_amp       = float(np.max(vs[AP1_thresh_x:end_ap]) - AP1_thresh_y)
    AP_max_x     = int(np.argmax(vs[AP1_thresh_x:end_ap])) + AP1_thresh_x
    AP_half_amp  = AP1_thresh_y + AP_amp / 2
    hw_s         = np.where(vs[AP1_thresh_x:] >= AP_half_amp)[0]
    hw_e         = np.where(vs[AP_max_x:]     <= AP_half_amp)[0]
    AP_HW_start  = hw_s[0] + AP1_thresh_x if len(hw_s) else AP_max_x
    AP_HW_end    = hw_e[0] + AP_max_x     if len(hw_e) else AP_max_x + 1
    AP_HW        = float((AP_HW_end - AP_HW_start) / (sf / 1000))
    AP_latency   = float((AP1_thresh_x + safety) / (sf / 1000))

    valid_thresh  = AP_thresh_x[~np.isnan(AP_thresh_x[:, rheo_idx]), rheo_idx]
    fAHP_thresh_x = AP1_thresh_x
    mAHP_end      = len(vs)
    if len(valid_thresh) > 1:
        for APi in range(1, len(valid_thresh)):
            AP2_thresh_x = int(valid_thresh[APi])
            mAHP_end     = AP2_thresh_x if not np.isnan(AP2_thresh_x) else len(vs)
            if mAHP_end <= fAHP_thresh_x + fAHP_dur + 1:
                fAHP_thresh_x = AP2_thresh_x
            elif APi > 1:
                break
            else:
                break

    fAHP_thresh_y = float(vs[fAHP_thresh_x])
    fAHP_end      = min(fAHP_thresh_x + fAHP_dur, len(vs))
    fAHP_amp      = float(np.min(vs[fAHP_thresh_x:fAHP_end]) - AP1_thresh_y)
    fAHP_x        = int(np.argmin(vs[fAHP_thresh_x:fAHP_end])) + fAHP_thresh_x
    mAHP_seg      = vs[fAHP_thresh_x + fAHP_dur:mAHP_end]
    mAHP_amp      = float(np.min(mAHP_seg) - fAHP_thresh_y) if len(mAHP_seg) > 0 else float('nan')
    mAHP_x        = int(np.argmin(mAHP_seg)) + fAHP_thresh_x + fAHP_dur if len(mAHP_seg) > 0 else fAHP_x
    dap_seg       = vs[fAHP_x:mAHP_x]
    DAP_amp       = float(np.max(dap_seg) - (fAHP_amp + fAHP_thresh_y)) if len(dap_seg) > 0 else float('nan')
    DAP_x         = int(np.argmax(dap_seg)) + fAHP_x if len(dap_seg) > 0 else fAHP_x

    mean_ISIs     = np.full(v.shape[1], np.nan)
    cv_ISIs       = np.full(v.shape[1], np.nan)
    SF_adaptation = np.full(v.shape[1], np.nan)
    AP_freq       = np.zeros(v.shape[1])
    for s in range(rheo_idx, v.shape[1]):
        AP_times   = AP_thresh_x[~np.isnan(AP_thresh_x[:, s]), s].astype(int)
        AP_freq[s] = len(AP_times) / (stepend_t - stepstart_t)
        if len(AP_times) < 2:
            continue
        ISIs = np.diff(t[AP_times + stepstart + safety - 1])
        mean_ISIs[s]     = float(np.mean(ISIs))
        cv_ISIs[s]       = float(np.std(ISIs) / np.mean(ISIs))
        SF_adaptation[s] = float(ISIs[0] / ISIs[-1]) if len(ISIs) >= 2 else 0.0

    idx_0    = int(np.searchsorted(csteps, 0))
    idx_300  = int(np.searchsorted(csteps, 300))
    P_FI     = np.polyfit(csteps[idx_0:idx_300 + 1], AP_freq[idx_0:idx_300 + 1], 1)
    FI_slope = float(P_FI[0])

    return dict(
        mean_RMP=mean_RMP, R_in=R_in, tau_m=tau_m, sag_ratio=sag_ratio,
        rheobase=int(rheobase), AP1_thresh_y=AP1_thresh_y, AP_amp=AP_amp,
        AP_HW=AP_HW, AP_latency=AP_latency, fAHP_amp=fAHP_amp,
        mAHP_amp=mAHP_amp, DAP_amp=DAP_amp,
        mean_ISI_last=float(mean_ISIs[rheo_idx]),
        cv_ISI_last=float(cv_ISIs[rheo_idx]),
        SF_adaptation_last=float(SF_adaptation[rheo_idx]),
        FI_slope=FI_slope, max_AP_freq=float(np.max(AP_freq)),
        # internal arrays for plotting
        _t=t, _v=v, _csteps=csteps, _sf=sf,
        _stepstart=stepstart, _stepend=stepend, _safety=safety,
        _taustep=taustep, _t_seg=t_seg, _fit_vals=fit_vals,
        _steadystate_start=steadystate_start,
        _RMP=RMP, _steadystate_mean=steadystate_mean,
        _dv_steadystate=dv_steadystate, _fit_steps=fit_steps, _fit_dv=fit_dv,
        _P_R_in=P_R_in, _P_FI=P_FI, _AP_freq=AP_freq,
        _AP_thresh_x=AP_thresh_x, _rheo_idx=rheo_idx,
        _AP1_thresh_x=AP1_thresh_x, _AP1_thresh_y=AP1_thresh_y,
        _AP_max_x=AP_max_x, _AP_half_amp=AP_half_amp,
        _AP_HW_start=AP_HW_start, _AP_HW_end=AP_HW_end,
        _fAHP_thresh_x=fAHP_thresh_x, _fAHP_thresh_y=fAHP_thresh_y,
        _fAHP_x=fAHP_x, _fAHP_amp=fAHP_amp,
        _mAHP_x=mAHP_x, _mAHP_amp=mAHP_amp,
        _DAP_x=DAP_x, _DAP_amp=DAP_amp,
        _min_peak_x=min_peak_x, _min_peak_y=min_peak_y,
        _stepstart_t=stepstart_t, _stepend_t=stepend_t,
    )


def _ensure_cat_gif(gif_path: str):
    """Generate cat_yarn.gif next to kakashi.py if it doesn't exist."""
    if os.path.exists(gif_path):
        return
    try:
        from PIL import Image, ImageDraw
        W, H, FRAMES = 96, 72, 16
        BG = (15, 15, 25)
        CAT_BODY  = (230, 120,  30); CAT_LIGHT = (245, 160,  70)
        CAT_DARK  = (180,  80,  10); CAT_BELLY = (245, 200, 140)
        CAT_EAR   = (220,  90,  90); CAT_NOSE  = (220,  80, 100)
        CAT_EYE   = (255, 200,   0); EYE_PUPIL = ( 10,  10,  10)
        WHISKER   = (240, 240, 240)
        YARN_1    = (220,  50,  80); YARN_2    = (255,  90, 110)
        YARN_DARK = (140,  20,  40)

        def dc(d, cx, cy, r, fill):
            d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=fill)

        def draw_paw(d, rx, ry, px, py, ta):
            d.line([rx, ry, px, py], fill=CAT_BODY, width=3)
            dc(d, px, py, 4, CAT_BODY)
            ang = math.atan2(py-ry, px-rx) + math.pi/2
            for s in (-1, 1):
                dc(d, int(px+s*2*math.cos(ang)), int(py+s*2*math.sin(ang)), 1, CAT_DARK)

        def draw_yarn(d, cx, cy, r, t):
            dc(d, cx, cy, r, YARN_1)
            dc(d, int(cx+r*0.3*math.cos(t*1.5)), int(cy+r*0.3*math.sin(t*1.5)), r//3, YARN_2)
            for i in range(5):
                a1=t+i*math.pi*0.4; a2=a1+math.pi*0.6
                d.line([int(cx+(r-1)*math.cos(a1)), int(cy+(r-1)*math.sin(a1)),
                        int(cx+(r-1)*math.cos(a2)), int(cy+(r-1)*math.sin(a2))],
                       fill=YARN_DARK, width=1)

        def draw_cat(d, frame, n):
            t=frame/n; ta=t*2*math.pi
            ts=math.sin(ta*1.5)*8
            for i in range(12):
                f=i/11
                dc(d, int(62+f*18+math.sin(f*math.pi+ta)*5),
                   int(54-f*12+ts*f), max(1,5-i//3),
                   (245,160,70) if i>8 else CAT_BODY)
            dc(d, 50, 56, 14, CAT_BODY)
            d.ellipse([32,36,64,62], fill=CAT_BODY)
            d.ellipse([36,44,60,62], fill=CAT_BELLY)
            for sx,sy,sw,sh in [(38,38,6,2),(46,37,6,2),(40,43,4,2)]:
                d.ellipse([sx,sy,sx+sw,sy+sh], fill=CAT_DARK)
            dc(d, 44, 26, 15, CAT_BODY); dc(d, 42, 22, 6, CAT_LIGHT)
            d.polygon([(30,16),(34,6),(39,16)], fill=CAT_BODY)
            d.polygon([(49,16),(53,6),(58,16)], fill=CAT_BODY)
            d.polygon([(32,15),(35,9),(38,15)], fill=CAT_EAR)
            d.polygon([(51,15),(54,9),(57,15)], fill=CAT_EAR)
            blink = frame in (6,7)
            for ex,ey in [(38,24),(50,24)]:
                if blink: d.line([ex-3,ey,ex+3,ey], fill=CAT_DARK, width=2)
                else:
                    dc(d,ex,ey,4,CAT_EYE); dc(d,ex,ey,2,EYE_PUPIL)
                    d.point((ex+1,ey-1), fill=(255,255,255))
            d.polygon([(43,30),(45,30),(44,32)], fill=CAT_NOSE)
            d.arc([41,30,44,34],0,180,fill=CAT_DARK,width=1)
            d.arc([44,30,47,34],0,180,fill=CAT_DARK,width=1)
            for wy in [28,31]:
                d.line([22,wy,38,wy+1],fill=WHISKER,width=1)
                d.line([50,wy+1,66,wy],fill=WHISKER,width=1)
            draw_paw(d,38,52,int(30+math.sin(ta)*14),int(57+math.cos(ta)*5),ta)
            draw_paw(d,54,52,int(58+math.sin(ta+math.pi)*12),int(58+math.cos(ta+math.pi)*4),ta)
            draw_paw(d,40,60,int(34+math.sin(ta+math.pi*.5)*9),int(66+math.cos(ta+math.pi*.5)*9),ta)
            draw_paw(d,56,60,int(60+math.sin(ta+math.pi*1.5)*11),int(65+math.cos(ta+math.pi*1.5)*10),ta)

        def make_frame(frame, n):
            img=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(img)
            t=frame/n; ta=t*2*math.pi
            cx=int(14+math.sin(ta+math.pi)*5); cy=int(58+abs(math.sin(ta))*3)
            draw_yarn(d,cx,cy,9,ta)
            d.line([cx+9,cy-2,int(30+math.sin(ta)*14),57],fill=YARN_1,width=1)
            draw_cat(d,frame,n)
            return img

        frames = [make_frame(i, FRAMES) for i in range(FRAMES)]
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       loop=0, duration=90, optimize=False)
    except Exception as e:
        print(f"Cat GIF generation failed: {e}")


def exp_growth(t, tau, v_steady, v_start):
    return v_steady + (v_start - v_steady) * np.exp(-t / tau)


def read_trace(file_path, trace):
    offset  = trace.fields['Data']
    samples = trace.fields['DataPoints']
    sr      = 1.0 / trace.fields['XInterval']
    yunit   = trace.fields['YUnit']
    label   = trace.fields['Label'].strip()

    with open(file_path, 'rb') as f:
        f.seek(offset)
        raw = np.fromfile(f, dtype='<f4', count=samples)

    if raw.size > 0 and np.abs(raw).max() > 1e6:
        raw = raw.byteswap().view(dtype='<f4')

    duration_ms = (samples / sr) * 1000.0
    time_ms     = np.linspace(0, duration_ms, samples)
    return raw, time_ms, sr, yunit, label


# ── Plugin Window ─────────────────────────────────────────────────────────────

class PluginWindow:
    def __init__(self, title="Plugin Plot", rows=1, cols=1, size=(800, 500)):
        self._win = pg.GraphicsLayoutWidget(title=title, show=False)
        self._win.resize(*size)
        self._win.setWindowTitle(title)
        self._rows = rows
        self._cols = cols

        self._plots = {}
        for r in range(rows):
            for c in range(cols):
                p = self._win.addPlot(row=r, col=c)
                self._plots[(r, c)] = p

        self.win    = self._win
        self.layout = self._win.ci

    def get_plot(self, row: int = 0, col: int = 0) -> pg.PlotItem:
        if (row, col) not in self._plots:
            raise IndexError(f"No plot at ({row}, {col}) — window has {self._rows}×{self._cols} grid.")
        return self._plots[(row, col)]

    def add_plot(self, row: int = None, col: int = None,
                 title: str = None, xlabel: str = "Time (ms)",
                 ylabel: str = "Signal") -> pg.PlotItem:
        kw = {}
        if row is not None: kw['row'] = row
        if col is not None: kw['col'] = col
        p = self._win.addPlot(**kw)
        if title:
            p.setTitle(title)
        p.setLabel('bottom', xlabel)
        p.setLabel('left',   ylabel)
        if row is not None and col is not None:
            self._plots[(row, col)] = p
        return p

    def plot(self, x, y, row=0, col=0, color='w', width=1, label=None,
             symbol=None, symbol_size=8, symbol_brush=None):
        p = self.get_plot(row, col)
        pen = pg.mkPen(color, width=width)
        kw  = dict(pen=pen, name=label)
        if symbol is not None:
            kw['symbol']      = symbol
            kw['symbolSize']  = symbol_size
            kw['symbolBrush'] = pg.mkBrush(symbol_brush or color)
            kw['symbolPen']   = pg.mkPen(None)
        if label:
            p.addLegend()
        return p.plot(x, y, **kw)

    def scatter(self, x, y, row=0, col=0, color='w', size=8, label=None):
        return self.plot(x, y, row=row, col=col, color=color, width=0,
                         symbol='o', symbol_size=size, label=label)

    def hline(self, y, row=0, col=0, color='y', width=1, style='dash', label=None):
        styles = {
            'solid': pg.QtCore.Qt.PenStyle.SolidLine,
            'dash':  pg.QtCore.Qt.PenStyle.DashLine,
            'dot':   pg.QtCore.Qt.PenStyle.DotLine,
        }
        pen  = pg.mkPen(color, width=width, style=styles.get(style, styles['dash']))
        line = pg.InfiniteLine(pos=y, angle=0, pen=pen,
                               label=label or "",
                               labelOpts={"color": color, "position": 0.95})
        self.get_plot(row, col).addItem(line)
        return line

    def vline(self, x, row=0, col=0, color='y', width=1, style='dash'):
        styles = {
            'solid': pg.QtCore.Qt.PenStyle.SolidLine,
            'dash':  pg.QtCore.Qt.PenStyle.DashLine,
            'dot':   pg.QtCore.Qt.PenStyle.DotLine,
        }
        pen  = pg.mkPen(color, width=width, style=styles.get(style, styles['dash']))
        line = pg.InfiniteLine(pos=x, angle=90, pen=pen)
        self.get_plot(row, col).addItem(line)
        return line

    def shade_region(self, t_start, t_end, row=0, col=0,
                     color=(255, 200, 0), alpha=40):
        r, g, b = color
        region  = pg.LinearRegionItem(
            values=(t_start, t_end),
            brush=pg.mkBrush(r, g, b, alpha),
            pen=pg.mkPen(r, g, b, 120),
            movable=False
        )
        self.get_plot(row, col).addItem(region)
        return region

    def set_labels(self, row=0, col=0,
                   title=None, xlabel=None, ylabel=None, yunits=None):
        p = self.get_plot(row, col)
        if title  is not None: p.setTitle(title)
        if xlabel is not None: p.setLabel('bottom', xlabel)
        if ylabel is not None: p.setLabel('left',   ylabel, units=yunits)

    def set_range(self, row=0, col=0, xrange=None, yrange=None):
        p = self.get_plot(row, col)
        if xrange: p.setXRange(*xrange)
        if yrange: p.setYRange(*yrange)

    def auto_range(self, row=None, col=None):
        if row is None:
            for p in self._plots.values():
                p.autoRange()
        else:
            self.get_plot(row, col).autoRange()

    def clear(self, row=None, col=None):
        if row is None:
            for p in self._plots.values():
                p.clear()
        else:
            self.get_plot(row, col).clear()

    def set_title(self, title: str):
        self._win.setWindowTitle(title)

    def set_size(self, width: int, height: int):
        self._win.resize(width, height)

    def show(self):
        self._win.show()
        self._win.raise_()

    def close(self):
        self._win.close()

    def save_image(self, path: str):
        exp = pg.exporters.ImageExporter(self._win.scene())
        exp.export(path)


# ── Plugin Context ────────────────────────────────────────────────────────────

class PluginContext:
    def __init__(self, app):
        self._app = app

    @property
    def file_path(self) -> str:
        return self._app.file_list[0] if self._app.file_list else ""

    @property
    def file_name(self) -> str:
        import os
        return os.path.basename(self.file_path)

    @property
    def is_heka(self) -> bool:
        return self._app.is_dat_mode

    @property
    def is_h5(self) -> bool:
        return self._app.is_h5_mode

    @property
    def is_igor(self) -> bool:
        return not self._app.is_dat_mode and not self._app.is_h5_mode

    @property
    def n_sweeps(self) -> int:
        return self._app.total_sweeps

    @property
    def current_index(self) -> int:
        return self._app.current_index

    @property
    def sample_rate_khz(self) -> float:
        return 1.0 / self._app.x_scale if self._app.x_scale else 0.0

    @property
    def x_scale(self) -> float:
        return self._app.x_scale

    @property
    def time_ms(self) -> "np.ndarray":
        return self._app.time_vec.copy() if self._app.time_vec is not None else np.array([])

    @property
    def current_group_series(self) -> tuple:
        if not self.is_heka or not self._app.flat_traces:
            return (0, 0)
        g, s, sw, t = self._app.flat_traces[self._app.current_index]
        return (g, s)

    @property
    def series_sweep_count(self) -> int:
        if not self.is_heka or not self._app.flat_traces:
            return self.n_sweeps
        g_cur, s_cur = self.current_group_series
        return sum(1 for (g, s, sw, t) in self._app.flat_traces
                   if g == g_cur and s == s_cur)

    @property
    def series_start_index(self) -> int:
        if not self.is_heka or not self._app.flat_traces:
            return 0
        g_cur, s_cur = self.current_group_series
        for i, (g, s, sw, t) in enumerate(self._app.flat_traces):
            if g == g_cur and s == s_cur:
                return i
        return 0

    def current_sweep(self) -> tuple:
        app = self._app
        if app.sweep_data is None:
            return np.array([]), np.array([])
        return app.sweep_data.copy(), app.time_vec.copy()

    def load_sweep(self, flat_index: int) -> tuple:
        return self._app._load_sweep(flat_index)

    def load_series(self, n: int = None) -> tuple:
        app = self._app
        g_cur, s_cur = self.current_group_series
        indices = [i for i, (g, s, sw, t) in enumerate(app.flat_traces)
                   if g == g_cur and s == s_cur]
        if n is not None:
            indices = indices[:n]

        all_data, time_ref = [], None
        for idx in indices:
            d, t = self.load_sweep(idx)
            if time_ref is None:
                time_ref = t
            all_data.append(d)

        if not all_data:
            return np.array([[]]), np.array([])

        max_len = max(len(d) for d in all_data)
        padded  = [np.pad(d, (0, max_len - len(d)), constant_values=np.nan)
                   if len(d) < max_len else d for d in all_data]
        return np.array(padded), time_ref

    def load_sweep_range(self, start: int, end: int) -> tuple:
        all_data, time_ref = [], None
        for idx in range(start, end):
            d, t = self.load_sweep(idx)
            if time_ref is None:
                time_ref = t
            all_data.append(d)
        if not all_data:
            return np.array([[]]), np.array([])
        max_len = max(len(d) for d in all_data)
        padded  = [np.pad(d, (0, max_len - len(d)), constant_values=np.nan)
                   if len(d) < max_len else d for d in all_data]
        return np.array(padded), time_ref

    @property
    def plot_widget(self):
        return self._app.pw

    def plot(self, x, y, color='w', width=1, label=None):
        import pyqtgraph as pg
        pen = pg.mkPen(color, width=width)
        self._app.pw.plot(x, y, pen=pen, name=label)

    def hline(self, y, color='y', width=1, style='dash', label=None):
        import pyqtgraph as pg
        styles = {
            'solid': pg.QtCore.Qt.PenStyle.SolidLine,
            'dash':  pg.QtCore.Qt.PenStyle.DashLine,
            'dot':   pg.QtCore.Qt.PenStyle.DotLine,
        }
        pen = pg.mkPen(color, width=width, style=styles.get(style, styles['dash']))
        line = pg.InfiniteLine(pos=y, angle=0, pen=pen,
                               label=label or "",
                               labelOpts={"color": color, "position": 0.95})
        self._app.pw.addItem(line)

    def vline(self, x, color='y', width=1, style='dash'):
        import pyqtgraph as pg
        styles = {
            'solid': pg.QtCore.Qt.PenStyle.SolidLine,
            'dash':  pg.QtCore.Qt.PenStyle.DashLine,
            'dot':   pg.QtCore.Qt.PenStyle.DotLine,
        }
        pen = pg.mkPen(color, width=width, style=styles.get(style, styles['dash']))
        self._app.pw.addItem(pg.InfiniteLine(pos=x, angle=90, pen=pen))

    def shade_region(self, t_start, t_end, color=(255, 200, 0), alpha=40):
        import pyqtgraph as pg
        r, g, b = color
        region = pg.LinearRegionItem(
            values=(t_start, t_end),
            brush=pg.mkBrush(r, g, b, alpha),
            pen=pg.mkPen(r, g, b, 120),
            movable=False
        )
        self._app.pw.addItem(region)

    def set_results(self, text: str):
        self._app.results_text.setPlainText(text)

    def set_last_results(self, d: dict):
        self._app._last_results = d

    @property
    def excluded_sweeps(self) -> set:
        """Set of 0-based flat sweep indices the user has excluded via Ctrl+E."""
        return set(self._app._excluded_sweeps)

    def log(self, msg: str):
        current = self._app.info_label.text()
        self._app.info_label.setText(current + "\n" + msg)

    def new_plot_window(self, title="Plugin Plot", rows=1, cols=1, size=(800, 500)):
        app = self._app
        if not hasattr(app, '_plugin_windows'):
            app._plugin_windows = []
        pw = PluginWindow(title=title, rows=rows, cols=cols, size=size)
        app._plugin_windows.append(pw)
        pw.show()
        return pw

    def close_plugin_windows(self):
        app = self._app
        for w in getattr(app, '_plugin_windows', []):
            try:
                w.close()
            except Exception:
                pass
        app._plugin_windows = []


class DoubleClickViewBox(pg.ViewBox):
    def mouseDoubleClickEvent(self, ev):
        self.autoRange()
        ev.accept()


class PatchBrowser(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MEOW")

        self.file_list     = []
        self.h5_keys       = []
        self.current_index = 0
        self.is_dat_mode   = False
        self.is_h5_mode    = False
        self.total_sweeps  = 0
        self.sweep_data    = None
        self.time_vec      = None
        self.x_scale       = 0.05

        self.bundle        = None
        self.flat_traces   = []
        self.plugins       = {}
        self._last_results  = {}
        self._active_spikes = []
        self._plugin_windows = []
        self._pick_mode     = None
        self._pick_clicks   = []
        self._pick_region   = None
        self._show_dvdt          = False   # whether the dV/dt split view is active
        self._excluded_sweeps    = set()   # flat indices of user-excluded sweeps
        self._plugin_params_cache = {}     # {plugin_name: {param_key: last_value}}
        self._show_split         = False   # whether the side-by-side split view is active
        self._split_zoom         = None    # saved (xrange, yrange) for right panel

        self.init_ui()
        self.load_plugins()
        self.setAcceptDrops(True)

        # ── cat GIF ───────────────────────────────────────────────────────
        self._cat_gif_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'cat_yarn.gif'
        )
        _ensure_cat_gif(self._cat_gif_path)
        self._setup_cat_overlay()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self._root_splitter = QSplitter(Qt.Orientation.Horizontal)

        mb = self.menuBar()

        main_menu = mb.addMenu("Main")
        main_menu.addAction("📂 Open…",            self.open_any_file)
        main_menu.addSeparator()

        ex_menu = main_menu.addMenu("💾 Export")
        ex_menu.addAction("Save JPG",              lambda: self.export_plot('jpg'))
        ex_menu.addAction("Save SVG",              lambda: self.export_plot('svg'))
        ex_menu.addSeparator()
        ex_menu.addAction("Export Sweep CSV",      self.export_sweep_csv)
        ex_menu.addAction("Export All Sweeps CSV", self.export_all_sweeps_csv)
        ex_menu.addSeparator()
        ex_menu.addAction("🎞 Export Traces to GIF…",  self.export_traces_gif)
        ex_menu.addAction("🌊 Waterfall Plot…",         self.show_waterfall)
        ex_menu.addSeparator()
        ex_menu.addAction("💾 Save Results CSV",   self.save_results_csv)

        an_menu = mb.addMenu("Analysis")
        an_menu.addAction("⚡ Run Active Analysis",  self.run_active_analysis)
        an_menu.addAction("🚀 Run Passive Analysis", self.run_passive_analysis)
        an_menu.addAction("🔬 Ephys Full Analysis…", self.run_ephys_analysis_gui)
        an_menu.addSeparator()
        an_menu.addAction("📐 Pick Baseline Window",
                          lambda: self._start_pick("baseline"))
        an_menu.addAction("📐 Pick RC Fit Window",
                          lambda: self._start_pick("rc"))

        view_menu = mb.addMenu("View")
        view_menu.addAction("Auto Range\tEsc",            self.pw_auto_range)
        view_menu.addAction("Close Plugin Windows",       self._close_plugin_windows)

        # ── Plugins — own top-level menu ──────────────────────────────────
        self.plugins_menu = mb.addMenu("🔌 Plugins")
        self.plugins_menu.addAction("🔄 Reload Plugins\tCtrl+R", self.load_plugins)
        self.plugins_menu.addSeparator()

        # ── Shortcuts help menu ───────────────────────────────────────────
        help_menu = mb.addMenu("⌨ Shortcuts")
        help_menu.addAction("Show All Shortcuts…", self._show_shortcuts_dialog)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumWidth(200)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        container = QWidget()
        self.controls = QVBoxLayout(container)
        self.controls.setSpacing(6)
        self.scroll.setWidget(container)
        self._root_splitter.addWidget(self.scroll)

        self.info_label = QLabel("No Files Loaded")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(
            "background:#1a1a2e; color:#aaa; padding:6px; "
            "border-radius:4px; font-size:11px;"
        )
        self.controls.addWidget(self.info_label)

        tg = QGroupBox("HEKA Tree")
        tl = QVBoxLayout()
        self.data_tree = QTreeWidget()
        self.data_tree.setHeaderLabel("Group / Series  (click to jump)")
        self.data_tree.setMinimumHeight(200)
        self.data_tree.itemClicked.connect(self.on_tree_item_clicked)
        tl.addWidget(self.data_tree)
        tg.setLayout(tl)
        self.controls.addWidget(tg)

        jg = QGroupBox("Navigation")
        jl = QVBoxLayout()
        self.sweep_count_label = QLabel("Total Sweeps: --")
        jl.addWidget(self.sweep_count_label)
        nav_row = QHBoxLayout()
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setFixedWidth(40)
        self.prev_btn.clicked.connect(self.go_prev)
        self.next_btn = QPushButton("▶")
        self.next_btn.setFixedWidth(40)
        self.next_btn.clicked.connect(self.go_next)
        self.jump_input = QLineEdit()
        self.jump_input.setPlaceholderText("Jump to #")
        self.jump_input.returnPressed.connect(self.jump_to_sweep)
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.jump_input)
        nav_row.addWidget(self.next_btn)
        jl.addLayout(nav_row)
        jg.setLayout(jl)
        self.controls.addWidget(jg)

        # ── Ephys Full Analysis ───────────────────────────────────────────
        ea_box = QGroupBox("Ephys Full Analysis")
        ea_box.setStyleSheet(
            "QGroupBox { border:1px solid #8e44ad; border-radius:4px; margin-top:8px; }"
            "QGroupBox::title { color:#9b59b6; font-weight:bold; subcontrol-origin:margin; left:8px; }"
        )
        ea_lay = QVBoxLayout()

        ea_info = QLabel("Sweep range to analyse (comma or dash, e.g. 1-11):")
        ea_info.setWordWrap(True)
        ea_info.setStyleSheet("font-size:11px; color:#aaa;")
        ea_lay.addWidget(ea_info)

        self.ea_range_edit = QLineEdit("1-11")
        self.ea_range_edit.setStyleSheet("font-size:12px; padding:3px;")
        self.ea_range_edit.setPlaceholderText("e.g.  1-11  or  1,3,5-9")
        ea_lay.addWidget(self.ea_range_edit)

        self.ea_csteps_edit = QLineEdit("-100:50:400")
        self.ea_csteps_edit.setStyleSheet("font-size:12px; padding:3px;")
        ea_lay.addWidget(QLabel("Current steps (start:step:stop pA):"))
        ea_lay.addWidget(self.ea_csteps_edit)

        run_ea_btn = QPushButton("🔬 Run Ephys Analysis")
        run_ea_btn.setStyleSheet(
            "background-color:#8e44ad; color:white; font-weight:bold; padding:6px;"
        )
        run_ea_btn.clicked.connect(self.run_ephys_analysis_gui)
        ea_lay.addWidget(run_ea_btn)

        save_ea_btn = QPushButton("💾 Save Ephys Results CSV")
        save_ea_btn.clicked.connect(self.save_ephys_results_csv)
        ea_lay.addWidget(save_ea_btn)

        ea_box.setLayout(ea_lay)
        self.controls.addWidget(ea_box)
        self._ephys_results = None   # last analysis result dict
        # Wrap plot area in a container so we can add sweep label above it
        plot_container = QWidget()
        plot_vbox = QVBoxLayout(plot_container)
        plot_vbox.setContentsMargins(0, 0, 0, 0)
        plot_vbox.setSpacing(2)

        self.sweep_label = QLabel("")
        self.sweep_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sweep_label.setStyleSheet(
            "color:#aaa; font-size:12px; padding:2px 0px;"
        )
        plot_vbox.addWidget(self.sweep_label)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground('k')
        plot_vbox.addWidget(self.glw)

        # Install application-level event filter so Ctrl+E (and other shortcuts)
        # fire regardless of which child widget currently holds keyboard focus.
        QApplication.instance().installEventFilter(self)

        self._root_splitter.addWidget(plot_container)
        self._root_splitter.setStretchFactor(0, 0)
        self._root_splitter.setStretchFactor(1, 1)
        self._root_splitter.setSizes([300, 900])
        root.addWidget(self._root_splitter)

        # Top plot — voltage / current trace (always visible)
        vb = DoubleClickViewBox()
        vb.setMouseMode(pg.ViewBox.RectMode)
        self.pw = self.glw.addPlot(row=0, col=0, viewBox=vb)
        self.pw.setLabel('left', 'Signal')
        self.pw.setLabel('bottom', 'Time', units='ms')
        self.pw.getAxis('bottom').enableAutoSIPrefix(False)

        # Right split panel — pinned reference sweep (Ctrl+N)
        vb_split = DoubleClickViewBox()
        vb_split.setMouseMode(pg.ViewBox.RectMode)
        self.pw_split = self.glw.addPlot(row=0, col=1, viewBox=vb_split)
        self.pw_split.setLabel('left', 'Signal')
        self.pw_split.setLabel('bottom', 'Time', units='ms')
        self.pw_split.getAxis('bottom').enableAutoSIPrefix(False)
        self.pw_split.setVisible(False)
        self.glw.ci.layout.setColumnStretchFactor(0, 1)
        self.glw.ci.layout.setColumnStretchFactor(1, 1)

        # Bottom plot — dV/dt (hidden until checkbox is ticked)
        vb2 = DoubleClickViewBox()
        vb2.setMouseMode(pg.ViewBox.RectMode)
        self.pw_dvdt = self.glw.addPlot(row=1, col=0, viewBox=vb2)
        self.pw_dvdt.setLabel('left', 'dV/dt', units='V/s')
        self.pw_dvdt.setLabel('bottom', 'Time', units='ms')
        self.pw_dvdt.getAxis('bottom').enableAutoSIPrefix(False)
        self.pw_dvdt.setXLink(self.pw)   # x-axes always in sync
        self.pw_dvdt.setVisible(False)
        self.pw_dvdt.setMaximumHeight(0)  # truly collapse when hidden
        self.glw.ci.layout.setRowStretchFactor(0, 1)
        self.glw.ci.layout.setRowStretchFactor(1, 0)

        # route mouse clicks for pick-mode through the top plot's scene
        self.glw.scene().sigMouseClicked.connect(self._on_plot_clicked)

        # ── dV/dt controls (kept for use via menu/shortcut) ──────────────
        self.dvdt_thresh_input = QLineEdit("50")
        self.dvdt_check = QCheckBox("Show dV/dt trace")
        self.dvdt_check.setChecked(False)
        self.dvdt_check.stateChanged.connect(self._on_dvdt_toggle)
        self.i_inj_input = QLineEdit("-100")
        self.base_start  = QLineEdit("1750")
        self.base_end    = QLineEdit("2250")
        self.rc_start    = QLineEdit("85")
        self.rc_end      = QLineEdit("145")

        res_box = QGroupBox("Results")
        res_lay = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFixedHeight(140)
        self.results_text.setStyleSheet(
            "background:#0d0d1a; color:#3498db; "
            "font-family: monospace; font-size:11px;"
        )
        self.results_text.setPlaceholderText("Results appear here after analysis…")
        res_lay.addWidget(self.results_text)
        self.spike_table = QTableWidget(0, 4)
        self.spike_table.setHorizontalHeaderLabels(["Peak mV", "Thresh mV", "Upstroke V/s", "HW ms"])
        self.spike_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.spike_table.setFixedHeight(120)
        res_lay.addWidget(self.spike_table)
        save_res_btn = QPushButton("💾 Save Results CSV")
        save_res_btn.clicked.connect(self.save_results_csv)
        res_lay.addWidget(save_res_btn)
        res_box.setLayout(res_lay)
        self.controls.addWidget(res_box)

        self.controls.addStretch()

        # ── Keyboard shortcuts ────────────────────────────────────────────
        QShortcut(QKeySequence("Ctrl+O"),     self).activated.connect(self.open_any_file)
        QShortcut(QKeySequence("Ctrl+R"),     self).activated.connect(self.load_plugins)
        QShortcut(QKeySequence("Ctrl+E"),     self).activated.connect(self._toggle_exclude_current_sweep)
        QShortcut(QKeySequence("Ctrl+S"),     self).activated.connect(self.save_results_csv)
        QShortcut(QKeySequence("Ctrl+A"),     self).activated.connect(self.run_active_analysis)
        QShortcut(QKeySequence("Ctrl+P"),     self).activated.connect(self.run_passive_analysis)
        QShortcut(QKeySequence("Ctrl+B"),     self).activated.connect(lambda: self._start_pick("baseline"))
        QShortcut(QKeySequence("Ctrl+F"),     self).activated.connect(lambda: self._start_pick("rc"))
        QShortcut(QKeySequence("Ctrl+D"),     self).activated.connect(self._toggle_dvdt_shortcut)
        QShortcut(QKeySequence("Ctrl+W"),     self).activated.connect(self._close_plugin_windows)
        QShortcut(QKeySequence("Ctrl+/"),     self).activated.connect(self._show_shortcuts_dialog)
        QShortcut(QKeySequence("Escape"),     self).activated.connect(self.pw_auto_range)

    # ── file loading ───────────────────────────────────────────────────────

    def open_any_file(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open Data Files", "",
            "All Supported (*.ibw *.dat *.h5 *.hdf5);;"
            "Igor Binary Wave (*.ibw);;"
            "HEKA Data (*.dat);;"
            "HDF5 Master (*.h5 *.hdf5)"
        )
        if not files:
            return
        self._dispatch_files(files)

    def _dispatch_files(self, files):
        def nat_sort_key(s):
            return [int(t) if t.isdigit() else t.lower()
                    for t in re.split('([0-9]+)', os.path.basename(s))]

        dat_files = [f for f in files if f.lower().endswith('.dat')]
        h5_files  = [f for f in files if f.lower().endswith(('.h5', '.hdf5'))]
        ibw_files = [f for f in files if f.lower().endswith('.ibw')]

        if dat_files:
            self.is_dat_mode = True
            self.is_h5_mode  = False
            self.file_list   = sorted(dat_files, key=nat_sort_key)
            self.current_index = 0
            self._load_heka_file(self.file_list[0])
        elif h5_files:
            self.is_dat_mode = False
            self.is_h5_mode  = True
            self.file_list   = [sorted(h5_files, key=nat_sort_key)[0]]
            self._load_h5_file(self.file_list[0])
        elif ibw_files:
            self.is_dat_mode = False
            self.is_h5_mode  = False
            self.file_list   = sorted(ibw_files, key=nat_sort_key)
            self.current_index = 0
            self.total_sweeps  = len(self.file_list)
            self.sweep_count_label.setText(f"Total Sweeps: {self.total_sweeps}")
            self.display_current_data()
        else:
            self.info_label.setText("Unsupported file type.\nSupported: .ibw  .dat  .h5")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            supported = ('.ibw', '.dat', '.h5', '.hdf5')
            if any(u.toLocalFile().lower().endswith(supported) for u in urls):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()
                 if u.toLocalFile()]
        if files:
            self._dispatch_files(files)
            event.acceptProposedAction()

    def _load_heka_file(self, file_path):
        self.info_label.setText("Parsing HEKA headers…")
        QApplication.processEvents()
        try:
            self.bundle      = heka_reader.Bundle(file_path)
            self.flat_traces = []

            for g in range(len(self.bundle.pul)):
                for s in range(len(self.bundle.pul[g])):
                    for sw in range(len(self.bundle.pul[g][s])):
                        self.flat_traces.append((g, s, sw, 0))

            self.total_sweeps = len(self.flat_traces)
            self.sweep_count_label.setText(f"Total Sweeps: {self.total_sweeps}")
            self._populate_tree()
            self.info_label.setText(
                f"Loaded: {os.path.basename(file_path)}\n"
                f"{self.total_sweeps} sweeps total\n"
                f"← → keys or Jump to navigate"
            )
            self.display_current_data()

        except Exception as e:
            self.info_label.setText(f"HEKA load error:\n{e}")
            import traceback; traceback.print_exc()

    def _populate_tree(self):
        self.data_tree.clear()
        flat_idx = 0

        for g in range(len(self.bundle.pul)):
            n_series = len(self.bundle.pul[g])
            if n_series == 0:
                continue

            g_item = QTreeWidgetItem(self.data_tree, [f"Group {g}  ({n_series} series)"])

            for s in range(n_series):
                series  = self.bundle.pul[g][s]
                n_sw    = len(series)
                tr      = series[0][0]
                sr      = 1.0 / tr.fields['XInterval']
                label   = tr.fields['Label'].strip()
                samples = tr.fields['DataPoints']
                dur_ms  = (samples / sr) * 1000.0

                s_text = (f"Series {s}  ·  {n_sw} sw  ·  "
                          f"{sr/1000:.0f} kHz  ·  {dur_ms:.0f} ms  ·  {label}")
                s_item = QTreeWidgetItem(g_item, [s_text])
                s_item.setData(0, Qt.ItemDataRole.UserRole, (flat_idx, n_sw))
                flat_idx += n_sw

        self.data_tree.expandAll()

    def on_tree_item_clicked(self, item, _col):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(data, tuple):
            start_idx, n_sw = data
            self.current_index = start_idx
            self.sweep_count_label.setText(
                f"Series: {n_sw} sweeps  (#{start_idx+1} – #{start_idx+n_sw})"
            )
            self.display_current_data()

    def _load_h5_file(self, path):
        self.is_h5_mode  = True
        self.is_dat_mode = False
        self.file_list   = [path]
        with h5py.File(path, 'r') as f:
            def nat_sort(s):
                return [int(t) if t.isdigit() else t.lower()
                        for t in re.split('([0-9]+)', s)]
            self.h5_keys      = sorted(f.keys(), key=nat_sort)
            self.total_sweeps = len(self.h5_keys)
        self.sweep_count_label.setText(f"Total Sweeps: {self.total_sweeps}")
        self.current_index = 0
        self.info_label.setText(
            f"Loaded: {os.path.basename(path)}\n"
            f"{self.total_sweeps} sweeps (HDF5)"
        )
        self.display_current_data()

    # ── navigation ─────────────────────────────────────────────────────────

    def jump_to_sweep(self):
        val = self.jump_input.text()
        if val.isdigit():
            target = int(val) - 1
            if 0 <= target < self.total_sweeps:
                self.current_index = target
                self.display_current_data()
                self.jump_input.clear()
                self.pw.setFocus()
            else:
                self.info_label.setText(f"Out of range (max {self.total_sweeps})")

    def go_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_data()

    def go_next(self):
        if self.current_index < self.total_sweeps - 1:
            self.current_index += 1
            self.display_current_data()

    def pw_auto_range(self):
        self.pw.autoRange()
        if self._show_dvdt:
            self.pw_dvdt.autoRange()

    def _on_dvdt_toggle(self, state):
        self._show_dvdt = bool(state)
        self.pw_dvdt.setVisible(self._show_dvdt)
        if self._show_dvdt:
            self.pw_dvdt.setMaximumHeight(16777215)    # restore Qt default max
            self.glw.ci.layout.setRowStretchFactor(0, 3)
            self.glw.ci.layout.setRowStretchFactor(1, 1)
        else:
            self.pw_dvdt.setMaximumHeight(0)           # collapse to nothing
            self.glw.ci.layout.setRowStretchFactor(0, 1)
            self.glw.ci.layout.setRowStretchFactor(1, 0)
        self.display_current_data()

    def _close_plugin_windows(self):
        for w in self._plugin_windows:
            try:
                w.close()
            except Exception:
                pass
        self._plugin_windows = []

    def _toggle_dvdt_shortcut(self):
        """Toggle the dV/dt checkbox via keyboard shortcut."""
        self.dvdt_check.setChecked(not self.dvdt_check.isChecked())

    def _toggle_exclude_current_sweep(self):
        """Ctrl+E: toggle the current sweep in/out of the global exclusion set."""
        idx = self.current_index
        if idx in self._excluded_sweeps:
            self._excluded_sweeps.discard(idx)
            self.info_label.setText(
                f"Sweep {idx + 1} un-excluded  ({len(self._excluded_sweeps)} excluded total)"
            )
        else:
            self._excluded_sweeps.add(idx)
            self.info_label.setText(
                f"Sweep {idx + 1} excluded  ({len(self._excluded_sweeps)} excluded total)"
            )
        self.display_current_data()   # refresh colour immediately

    def _clear_all_exclusions(self):
        """Ctrl+Shift+E: remove every sweep from the exclusion set."""
        n = len(self._excluded_sweeps)
        self._excluded_sweeps.clear()
        self.info_label.setText(f"All exclusions cleared  ({n} sweep(s) restored)")
        self.display_current_data()

    def _toggle_split_view(self):
        """Ctrl+N: toggle side-by-side split view.
        Both panels show the same (current) sweep and navigate together.
        The right panel keeps its zoom when moving between sweeps.
        """
        self._show_split = not self._show_split
        if self._show_split:
            self.pw_split.setVisible(True)
            self.glw.ci.layout.setColumnStretchFactor(0, 1)
            self.glw.ci.layout.setColumnStretchFactor(1, 1)
            # Reset zoom on the right panel so it starts auto-ranged
            self._split_zoom = None
            self.info_label.setText(
                "Split view ON — zoom right panel independently.\n"
                "Both panels navigate together. Ctrl+N to close."
            )
        else:
            self.pw_split.setVisible(False)
            self.glw.ci.layout.setColumnStretchFactor(0, 1)
            self.glw.ci.layout.setColumnStretchFactor(1, 0)
            self.info_label.setText("Split view OFF")
        self.display_current_data()

    def _refresh_split_panel(self):
        """Redraw the right panel with the current sweep, preserving its zoom."""
        if not self._show_split or not self.file_list:
            return
        try:
            data, time_ms, disp_unit, label = self._load_sweep(self.current_index)
        except Exception:
            return

        # Save zoom before clearing
        vr = self.pw_split.viewRange()
        has_zoom = getattr(self, '_split_zoom', None) is not None

        self.pw_split.clear()

        color = '#c0392b' if self.current_index in self._excluded_sweeps else '#f39c12'
        self.pw_split.plot(time_ms, data, pen=pg.mkPen(color, width=1))
        self.pw_split.setLabel('left',   label, units=disp_unit)
        self.pw_split.setLabel('bottom', 'Time', units='ms')
        self.pw_split.setTitle(f"Sweep {self.current_index + 1}  [zoomed]", size='10pt')

        if has_zoom:
            # Restore the saved zoom
            xr, yr = self._split_zoom
            self.pw_split.setXRange(*xr, padding=0)
            self.pw_split.setYRange(*yr, padding=0)
        else:
            self.pw_split.autoRange()

        # Hook into sigRangeChanged to save zoom whenever user drags/zooms
        try:
            self.pw_split.sigRangeChanged.disconnect(self._on_split_zoom_changed)
        except Exception:
            pass
        self.pw_split.sigRangeChanged.connect(self._on_split_zoom_changed)

    def _on_split_zoom_changed(self, _, ranges):
        """Called whenever the right panel's view range changes — save it."""
        self._split_zoom = (ranges[0], ranges[1])

    def _show_shortcuts_dialog(self):
        """Pop up a clean reference of all keyboard shortcuts."""
        shortcuts = [
            ("Navigation",      ""),
            ("←  /  →",         "Previous / Next sweep"),
            ("Ctrl+O",          "Open file"),
            ("",                ""),
            ("Analysis",        ""),
            ("Ctrl+A",          "Run Active (AP) analysis"),
            ("Ctrl+P",          "Run Passive analysis"),
            ("Ctrl+B",          "Pick Baseline window on trace"),
            ("Ctrl+F",          "Pick RC Fit window on trace"),
            ("",                ""),
            ("View",            ""),
            ("Esc",             "Auto-range plot"),
            ("Ctrl+D",          "Toggle dV/dt split view"),
            ("Ctrl+W",          "Close all plugin windows"),
            ("",                ""),
            ("Export & Save",   ""),
            ("Ctrl+E",          "Toggle exclude current sweep from analysis"),
            ("Ctrl+Shift+E",    "Clear ALL exclusions"),
            ("Ctrl+N",          "Toggle split view (both panels navigate, right keeps zoom)"),
            ("Ctrl+S",          "Save results CSV"),
            ("",                ""),
            ("Plugins",         ""),
            ("Ctrl+R",          "Reload plugins"),
            ("",                ""),
            ("Help",            ""),
            ("Ctrl+/",          "Show this shortcuts reference"),
        ]

        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard Shortcuts")
        dlg.setMinimumWidth(380)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(2)

        for key, desc in shortcuts:
            if desc == "":
                if key:
                    # section header
                    lbl = QLabel(f"<b>{key}</b>")
                    lbl.setStyleSheet("color:#e67e22; font-size:12px; padding-top:8px;")
                    layout.addWidget(lbl)
                else:
                    # blank spacer line
                    layout.addWidget(QLabel(""))
            else:
                row = QWidget()
                rl  = QHBoxLayout(row)
                rl.setContentsMargins(4, 1, 4, 1)
                key_lbl = QLabel(f"<code>{key}</code>")
                key_lbl.setStyleSheet(
                    "background:#1a1a2e; color:#3498db; padding:2px 6px; "
                    "border-radius:3px; font-family:monospace; min-width:100px;"
                )
                key_lbl.setFixedWidth(130)
                desc_lbl = QLabel(desc)
                desc_lbl.setStyleSheet("color:#ccc; font-size:11px;")
                rl.addWidget(key_lbl)
                rl.addWidget(desc_lbl)
                rl.addStretch()
                layout.addWidget(row)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(dlg.reject)
        layout.addWidget(bb)
        dlg.exec()

    # ── cat overlay ───────────────────────────────────────────────────────

    def _setup_cat_overlay(self):
        """Place an animated cat GIF in the bottom-right corner of the window."""
        if not os.path.exists(self._cat_gif_path):
            return
        self._cat_label = QLabel(self.centralWidget())
        self._cat_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._cat_label.setStyleSheet("background: transparent;")
        movie = QMovie(self._cat_gif_path)
        self._cat_label.setMovie(movie)
        movie.start()
        self._cat_label.resize(96, 72)
        self._cat_label.show()
        self._reposition_cat()

    def _reposition_cat(self):
        """Keep the cat anchored to the bottom-right of the central widget."""
        if not hasattr(self, '_cat_label'):
            return
        cw = self.centralWidget()
        if cw:
            x = cw.width()  - self._cat_label.width()  - 6
            y = cw.height() - self._cat_label.height() - 6
            self._cat_label.move(x, y)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_cat()

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.KeyPress:
            modifiers = event.modifiers()
            ctrl      = Qt.KeyboardModifier.ControlModifier
            shift     = Qt.KeyboardModifier.ShiftModifier
            key       = event.key()
            if (modifiers & ctrl) and not (modifiers & shift) and key == Qt.Key.Key_E:
                self._toggle_exclude_current_sweep()
                return True
            if (modifiers & ctrl) and (modifiers & shift) and key == Qt.Key.Key_E:
                self._clear_all_exclusions()
                return True
            if (modifiers & ctrl) and not (modifiers & shift) and key == Qt.Key.Key_N:
                self._toggle_split_view()
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.pw.autoRange(); return
        modifiers = event.modifiers()
        ctrl = Qt.KeyboardModifier.ControlModifier
        if modifiers & ctrl:
            k = event.key()
            if k == Qt.Key.Key_E:
                self._toggle_exclude_current_sweep(); return
            if k == Qt.Key.Key_O:
                self.open_any_file(); return
            if k == Qt.Key.Key_R:
                self.load_plugins(); return
            if k == Qt.Key.Key_S:
                self.save_results_csv(); return
            if k == Qt.Key.Key_A:
                self.run_active_analysis(); return
            if k == Qt.Key.Key_P:
                self.run_passive_analysis(); return
            if k == Qt.Key.Key_D:
                self._toggle_dvdt_shortcut(); return
            if k == Qt.Key.Key_W:
                self._close_plugin_windows(); return
            if k == Qt.Key.Key_Slash:
                self._show_shortcuts_dialog(); return
        if not self.file_list: return
        if event.key() == Qt.Key.Key_Right:
            if self.current_index < self.total_sweeps - 1:
                self.current_index += 1
                self.display_current_data()
        elif event.key() == Qt.Key.Key_Left:
            if self.current_index > 0:
                self.current_index -= 1
                self.display_current_data()

    # ══════════════════════════════════════════════════════════════════════
    # UNIFIED SWEEP LOADER
    # Always returns (data: float64 ndarray, time_ms: float64 ndarray,
    #                 display_unit: str, label: str)
    # Handles HEKA .dat, HDF5 .h5, and IBW — callers never branch on format.
    # ══════════════════════════════════════════════════════════════════════

    def _load_sweep(self, flat_index: int) -> tuple:
        """
        Load any sweep by flat index and return normalised numpy arrays.

        Returns
        -------
        data      : np.ndarray  float64, already converted to mV or pA
        time_ms   : np.ndarray  float64, milliseconds from recording start
        disp_unit : str         'mV', 'pA', or raw unit string
        label     : str         channel / signal label
        """
        # ── HEKA .dat ────────────────────────────────────────────────────
        if self.is_dat_mode:
            g, s, sw, t = self.flat_traces[flat_index]
            trace = self.bundle.pul[g][s][sw][t]
            raw, time_ms, sr, yunit, label = read_trace(self.file_list[0], trace)

            if yunit == 'V':
                data, disp_unit = raw.astype(np.float64) * 1000.0, 'mV'
            elif yunit == 'A':
                data, disp_unit = raw.astype(np.float64) * 1e12,   'pA'
            else:
                data, disp_unit = raw.astype(np.float64),           yunit

            return data, time_ms.astype(np.float64), disp_unit, label

        # ── HDF5 .h5 ─────────────────────────────────────────────────────
        elif self.is_h5_mode:
            trace_name = self.h5_keys[flat_index]
            with h5py.File(self.file_list[0], 'r') as f:
                raw    = f[trace_name][:].astype(np.float64)
                x_step = float(f[trace_name].attrs.get('dx', 0.02))
                label  = f[trace_name].attrs.get('label', trace_name)
                yunit  = f[trace_name].attrs.get('unit', '')

            time_ms = np.arange(len(raw), dtype=np.float64) * x_step

            if yunit == 'V':
                data, disp_unit = raw * 1000.0, 'mV'
            elif yunit == 'A':
                data, disp_unit = raw * 1e12,   'pA'
            else:
                data, disp_unit = raw, yunit or 'a.u.'

            return data, time_ms, disp_unit, str(label)

        # ── IBW ──────────────────────────────────────────────────────────
        else:
            file_path = self.file_list[flat_index]
            wdata  = bw.load(file_path)
            raw    = wdata['wave']['wData']
            x_step = wdata['wave']['wave_header']['sfA'][0]

            if raw.ndim == 2 and raw.shape[1] >= 2:
                # Membrane potential rests at ~-60 to -80 mV.
                # Command / stimulus sits near 0 mV.
                # Pick the column with the most negative mean.
                means = [float(np.mean(raw[:, c])) for c in range(raw.shape[1])]
                col   = int(np.argmin(means))
                data  = raw[:, col].astype(np.float64)
            else:
                data  = raw.flatten().astype(np.float64)

            time_ms   = np.arange(len(data), dtype=np.float64) * x_step
            disp_unit = 'mV'
            label     = os.path.basename(file_path)

            return data, time_ms, disp_unit, label

    # ── display ────────────────────────────────────────────────────────────

    def display_current_data(self):
        """Load the current sweep via the unified loader and plot it."""
        if not self.file_list:
            return
        self.pw.clear()
        self.pw_dvdt.clear()

        try:
            data, time_ms, disp_unit, label = self._load_sweep(self.current_index)
        except Exception as e:
            self.info_label.setText(f"Load error: {e}")
            import traceback; traceback.print_exc()
            return

        # cache for analysis / plugins
        self.sweep_data = data
        self.time_vec   = time_ms
        self.x_scale    = float(time_ms[1] - time_ms[0]) if len(time_ms) > 1 else 0.05

        # pick pen colour by format so the display looks the same as before
        if self.current_index in self._excluded_sweeps:
            pen_color = '#c0392b'   # dim red = excluded
        elif self.is_dat_mode:
            pen_color = '#e67e22'
        elif self.is_h5_mode:
            pen_color = '#2ecc71'
        else:
            pen_color = 'w'

        self.pw.plot(time_ms, data, pen=pg.mkPen(pen_color, width=1))
        self.pw.setLabel('left',   label, units=disp_unit)
        self.pw.setLabel('bottom', 'Time', units='ms')
        self.pw.autoRange()

        # ── right split panel (zoom preserved) ───────────────────────────
        self._refresh_split_panel()

        # ── dV/dt panel ───────────────────────────────────────────────────
        if self._show_dvdt and len(data) > 11:
            fs_khz  = (1.0 / self.x_scale) / 1000.0   # samples per ms → kHz
            vs      = savgol_filter(data, 11, 3)        # same smoothing as active analysis
            dvdt    = np.diff(vs) * fs_khz * 1000.0    # V/s  (mV/ms * 1000 = V/s)
            t_dvdt  = time_ms[:-1]                      # one sample shorter after diff
            self.pw_dvdt.plot(t_dvdt, dvdt,
                              pen=pg.mkPen('#9b59b6', width=1))
            self.pw_dvdt.setLabel('left', 'dV/dt', units='V/s')
            # draw threshold line if a value is set
            try:
                thresh = float(self.dvdt_thresh_input.text())
                self.pw_dvdt.addItem(pg.InfiniteLine(
                    pos=thresh, angle=0,
                    pen=pg.mkPen('#e74c3c', width=1,
                                 style=pg.QtCore.Qt.PenStyle.DashLine),
                    label=f"{thresh:.0f} V/s",
                    labelOpts={"color": "#e74c3c", "position": 0.98}
                ))
            except ValueError:
                pass
            self.pw_dvdt.autoRange()

        # info label
        sr_khz = (1.0 / self.x_scale) / 1000.0 if self.x_scale else 0
        dur_ms = time_ms[-1] if len(time_ms) else 0

        if self.is_dat_mode and self.flat_traces:
            g, s, sw, t = self.flat_traces[self.current_index]
            self.info_label.setText(
                f"G{g} · S{s} · Sw{sw+1}  "
                f"(#{self.current_index+1}/{self.total_sweeps})\n"
                f"{label} · {sr_khz:.0f} kHz · {dur_ms:.0f} ms · {disp_unit}"
            )
        elif self.is_h5_mode:
            self.info_label.setText(
                f"H5: {self.h5_keys[self.current_index]}  "
                f"({self.current_index+1}/{self.total_sweeps})\n"
                f"{sr_khz:.1f} kHz · {dur_ms:.0f} ms · {disp_unit}"
            )
        else:
            self.info_label.setText(
                f"IBW: {self.current_index+1}/{len(self.file_list)}\n"
                f"{label}  ·  {sr_khz:.1f} kHz  ·  {dur_ms:.0f} ms  ·  {disp_unit}"
            )

        # ── sweep number label above the plot ─────────────────────────────
        excl_tag = "  ⊘ EXCLUDED" if self.current_index in self._excluded_sweeps else ""
        self.sweep_label.setText(
            f"Sweep  {self.current_index + 1}  /  {self.total_sweeps}{excl_tag}"
        )

    # ── export ─────────────────────────────────────────────────────────────

    def export_plot(self, file_type):
        if self.sweep_data is None: return
        base    = (self.h5_keys[self.current_index] if self.is_h5_mode
                   else f"Sweep_{self.current_index+1}")
        ext     = ".jpg" if file_type == 'jpg' else ".svg"
        filt    = "JPEG (*.jpg)" if file_type == 'jpg' else "SVG (*.svg)"
        path, _ = QFileDialog.getSaveFileName(self, "Export", f"{base}{ext}", filt)
        if not path: return
        try:
            exp = (pg.exporters.ImageExporter(self.pw) if file_type == 'jpg'
                   else pg.exporters.SVGExporter(self.pw))
            exp.export(path)
            self.info_label.setText(f"Exported: {os.path.basename(path)}")
        except Exception as e:
            print(f"Export error: {e}")

    def export_sweep_csv(self):
        if self.sweep_data is None or self.time_vec is None:
            self.info_label.setText("No sweep loaded to export.")
            return

        base    = (self.h5_keys[self.current_index] if self.is_h5_mode
                   else f"Sweep_{self.current_index+1}")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Sweep CSV", f"{base}.csv", "CSV Files (*.csv)"
        )
        if not path: return

        try:
            if self.is_dat_mode and self.flat_traces:
                g, s, sw, t = self.flat_traces[self.current_index]
                tr    = self.bundle.pul[g][s][sw][t]
                yunit = tr.fields['YUnit']
                label = tr.fields['Label'].strip()
                disp_unit = 'mV' if yunit == 'V' else ('pA' if yunit == 'A' else yunit)
                header = f"time_ms,{label}_{disp_unit}"
            else:
                header = "time_ms,signal"

            data = np.column_stack((self.time_vec, self.sweep_data))
            np.savetxt(path, data, delimiter=',', header=header, comments='')
            self.info_label.setText(f"CSV saved: {os.path.basename(path)}")
        except Exception as e:
            self.info_label.setText(f"CSV export error: {e}")

    def export_all_sweeps_csv(self):
        if not self.is_dat_mode or not self.flat_traces:
            self.info_label.setText("All-sweeps export only available for HEKA .dat files.")
            return

        g_cur, s_cur, _, _ = self.flat_traces[self.current_index]
        series_indices = [
            (idx, sw)
            for idx, (g, s, sw, t) in enumerate(self.flat_traces)
            if g == g_cur and s == s_cur
        ]
        if not series_indices: return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export All Sweeps CSV",
            f"G{g_cur}_S{s_cur}_all_sweeps.csv", "CSV Files (*.csv)"
        )
        if not path: return

        try:
            self.info_label.setText("Exporting all sweeps…")
            QApplication.processEvents()

            all_data, time_ref, col_labels = [], None, []

            for flat_idx, sw_idx in series_indices:
                data, time_ms, disp_unit, label = self._load_sweep(flat_idx)
                if time_ref is None:
                    time_ref = time_ms
                all_data.append(data)
                col_labels.append(f"Sweep_{sw_idx+1}_{disp_unit}")

            max_len = max(len(d) for d in all_data)
            padded  = [np.pad(d, (0, max_len - len(d)), constant_values=np.nan)
                       if len(d) < max_len else d for d in all_data]

            if len(time_ref) < max_len:
                dt    = time_ref[1] - time_ref[0]
                extra = np.arange(1, max_len - len(time_ref) + 1) * dt + time_ref[-1]
                time_ref = np.concatenate([time_ref, extra])

            matrix = np.column_stack([time_ref] + padded)
            header = "time_ms," + ",".join(col_labels)
            np.savetxt(path, matrix, delimiter=',', header=header, comments='')
            self.info_label.setText(
                f"Exported {len(series_indices)} sweeps → {os.path.basename(path)}"
            )
        except Exception as e:
            self.info_label.setText(f"CSV export error: {e}")

    # ── interactive pick mode ───────────────────────────────────────────────

    def _start_pick(self, mode):
        if self.sweep_data is None:
            self.info_label.setText("Load a sweep first.")
            return
        self._pick_mode   = f"{mode}_start"
        self._pick_clicks = []
        if self._pick_region is not None:
            self.pw.removeItem(self._pick_region)
            self._pick_region = None
        colour = (52, 152, 219) if mode == "baseline" else (46, 204, 113)
        label  = "Baseline" if mode == "baseline" else "RC Fit"
        self.info_label.setText(
            f"📐 Pick {label} Window\n"
            f"Click START on the trace, then END."
        )
        self.glw.setCursor(pg.QtCore.Qt.CursorShape.CrossCursor)
        self._pick_colour = colour

    def _on_plot_clicked(self, event):
        if self._pick_mode is None: return
        if event.button() != pg.QtCore.Qt.MouseButton.LeftButton: return

        vb  = self.pw.getViewBox()
        pos = vb.mapSceneToView(event.scenePos())
        t   = pos.x()
        self._pick_clicks.append(t)
        event.accept()

        r, g, b = self._pick_colour
        self.pw.addItem(pg.InfiniteLine(
            pos=t, angle=90,
            pen=pg.mkPen((r, g, b, 200), width=1,
                         style=pg.QtCore.Qt.PenStyle.DashLine)
        ))

        if len(self._pick_clicks) == 1:
            t0 = self._pick_clicks[0]
            self._pick_region = pg.LinearRegionItem(
                values=(t0, t0),
                brush=pg.mkBrush(r, g, b, 40),
                pen=pg.mkPen(r, g, b, 120),
                movable=False
            )
            self.pw.addItem(self._pick_region)
            self.glw.scene().sigMouseMoved.connect(self._on_pick_mouse_move)
            mode_prefix = self._pick_mode.replace("_start", "")
            self._pick_mode = f"{mode_prefix}_end"
            label = "Baseline" if "baseline" in self._pick_mode else "RC Fit"
            self.info_label.setText(
                f"📐 Pick {label} Window\n"
                f"Start: {t:.1f} ms — now click END."
            )

        elif len(self._pick_clicks) == 2:
            try:
                self.glw.scene().sigMouseMoved.disconnect(self._on_pick_mouse_move)
            except Exception:
                pass

            t0, t1 = sorted(self._pick_clicks)
            mode_prefix = self._pick_mode.replace("_end", "")

            if "baseline" in mode_prefix:
                self.base_start.setText(f"{t0:.1f}")
                self.base_end.setText(f"{t1:.1f}")
                label = "Baseline"
            else:
                self.rc_start.setText(f"{t0:.1f}")
                self.rc_end.setText(f"{t1:.1f}")
                label = "RC Fit"

            if self._pick_region is not None:
                self._pick_region.setRegion((t0, t1))

            self.info_label.setText(
                f"✅ {label} set: {t0:.1f} – {t1:.1f} ms\n"
                f"Run Passive Analysis to apply."
            )
            self.glw.setCursor(pg.QtCore.Qt.CursorShape.ArrowCursor)
            self._pick_mode   = None
            self._pick_clicks = []

    def _on_pick_mouse_move(self, scene_pos):
        if self._pick_region is None or not self._pick_clicks: return
        vb  = self.pw.getViewBox()
        pos = vb.mapSceneToView(scene_pos)
        t0  = self._pick_clicks[0]
        t1  = pos.x()
        lo, hi = (t0, t1) if t0 < t1 else (t1, t0)
        self._pick_region.setRegion((lo, hi))

    # ── passive analysis ───────────────────────────────────────────────────

    def run_passive_analysis(self):
        if self.sweep_data is None: return
        try:
            fs_ms   = 1.0 / self.x_scale
            i_inj   = float(self.i_inj_input.text())
            bs      = int(float(self.base_start.text()) * fs_ms)
            be      = int(float(self.base_end.text())   * fs_ms)
            rmp     = np.mean(self.sweep_data[bs:be])
            f0, f1  = float(self.rc_start.text()), float(self.rc_end.text())
            fi, fe  = int(f0 * fs_ms), int(f1 * fs_ms)
            y       = self.sweep_data[fi:fe]
            x       = np.linspace(0, f1 - f0, len(y))
            popt, _ = curve_fit(exp_growth, x, y, p0=[20, np.mean(y[-10:]), rmp])
            r_in    = abs((popt[1] - popt[2]) / i_inj) * 1000
            txt = (f"── Passive Analysis ──\n"
                   f"RMP:  {rmp:.2f} mV\n"
                   f"R_in: {r_in:.1f} MΩ\n"
                   f"Tau:  {popt[0]:.1f} ms")
            self.results_text.setPlainText(txt)
            self._last_results = {"type":"passive","RMP_mV":rmp,"Rin_MOhm":r_in,"Tau_ms":popt[0]}
            self.pw.plot(x + f0, exp_growth(x, *popt), pen=pg.mkPen('r', width=2))
            t_bs = self.time_vec[bs]
            t_be = self.time_vec[min(be, len(self.time_vec)-1)]
            base_region = pg.LinearRegionItem(
                values=(t_bs, t_be),
                brush=pg.mkBrush(52, 152, 219, 50),
                pen=pg.mkPen(52, 152, 219, 120),
                movable=False
            )
            self.pw.addItem(base_region)
            rmp_line = pg.InfiniteLine(
                pos=rmp, angle=0,
                pen=pg.mkPen('#3498db', width=1,
                             style=pg.QtCore.Qt.PenStyle.DashLine),
                label=f"RMP {rmp:.1f} mV",
                labelOpts={"color":"#3498db","position":0.05,
                           "anchors":[(0,1),(0,1)]}
            )
            self.pw.addItem(rmp_line)
        except Exception as e:
            self.results_text.setPlainText(f"Passive error: {e}")

    # ── active analysis ────────────────────────────────────────────────────

    def run_active_analysis(self):
        if self.sweep_data is None: return
        self._active_spikes = []
        self.spike_table.setRowCount(0)
        try:
            fs_khz   = 1.0 / self.x_scale
            thresh   = float(self.dvdt_thresh_input.text())
            vs       = savgol_filter(self.sweep_data, 11, 3)
            dvdt     = np.diff(vs) * fs_khz
            peaks, _ = find_peaks(self.sweep_data, height=0,
                                  distance=int(2.0 * fs_khz))
            for p in peaks:
                ws   = max(0, p - int(2.0 * fs_khz))
                wd   = dvdt[ws:p]
                over = np.where(wd >= thresh)[0]
                if not len(over): continue
                ti     = ws + over[0]
                v_t    = self.sweep_data[ti]
                v_p    = self.sweep_data[p]
                v_half = v_t + (v_p - v_t) / 2
                hw_s   = max(0, p - int(fs_khz))
                hw_e   = min(len(self.sweep_data), p + int(4 * fs_khz))
                oh     = np.where(self.sweep_data[hw_s:hw_e] >= v_half)[0]
                hw     = (oh[-1] - oh[0]) * self.x_scale if len(oh) > 1 else 0
                row    = self.spike_table.rowCount()
                self.spike_table.insertRow(row)
                self.spike_table.setItem(row, 0, QTableWidgetItem(f"{v_p:.1f}"))
                self.spike_table.setItem(row, 1, QTableWidgetItem(f"{v_t:.1f}"))
                self.spike_table.setItem(row, 2, QTableWidgetItem(f"{np.max(wd):.1f}"))
                self.spike_table.setItem(row, 3, QTableWidgetItem(f"{hw:.3f}"))
                self._active_spikes.append({"peak_mV":v_p,"thresh_mV":v_t,"upstroke_Vs":np.max(wd),"halfwidth_ms":hw})
                self.pw.plot([ti * self.x_scale], [v_t],
                             symbol='t1', symbolBrush='b', size=10)
                self.pw.plot([p  * self.x_scale], [v_p],
                             symbol='o',  symbolBrush='r', size=10)
            n = len(self._active_spikes)
            txt = f"── Active Analysis ──\n{n} spike(s) detected\n"
            if n > 0:
                txt += "\n".join(
                    f"  #{i+1}  peak={s['peak_mV']:.1f} mV  hw={s['halfwidth_ms']:.3f} ms"
                    for i,s in enumerate(self._active_spikes)
                )
            self.results_text.setPlainText(txt)
            self._last_results = {"type":"active","spikes":self._active_spikes}
        except Exception as e:
            self.results_text.setPlainText(f"Active error: {e}")

    def save_results_csv(self):
        res = getattr(self, '_last_results', None)
        if not res:
            self.results_text.setPlainText("No results to save yet.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results CSV", "results.csv", "CSV Files (*.csv)"
        )
        if not path: return

        import csv
        try:
            rtype = res.get("type", "unknown")
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                if rtype == "passive":
                    writer.writerow(["metric", "value"])
                    writer.writerow(["RMP_mV",   res["RMP_mV"]])
                    writer.writerow(["Rin_MOhm", res["Rin_MOhm"]])
                    writer.writerow(["Tau_ms",   res["Tau_ms"]])
                elif rtype == "active":
                    writer.writerow(["spike", "peak_mV", "thresh_mV", "upstroke_Vs", "halfwidth_ms"])
                    for i, s in enumerate(res.get("spikes", [])):
                        writer.writerow([i+1, s["peak_mV"], s["thresh_mV"],
                                         s["upstroke_Vs"], s["halfwidth_ms"]])
                else:
                    writer.writerow(["key", "value"])
                    for k, v in res.items():
                        if k != "type":
                            writer.writerow([k, v])
            self.info_label.setText(f"Results saved: {os.path.basename(path)}")
        except Exception as e:
            self.results_text.setPlainText(f"Save error: {e}")

    # ── plugin system ──────────────────────────────────────────────────────

    def load_plugins(self):
        self.plugins = {}
        plugin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plugins')
        os.makedirs(plugin_dir, exist_ok=True)

        for path in sorted(glob.glob(os.path.join(plugin_dir, '*.py'))):
            mod_name = os.path.basename(path)[:-3]
            try:
                spec = importlib.util.spec_from_file_location(mod_name, path)
                mod  = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, 'NAME') and hasattr(mod, 'run'):
                    self.plugins[mod.NAME] = mod
                    print(f"Plugin loaded: {mod.NAME}")
                else:
                    print(f"Plugin skipped (missing NAME or run): {mod_name}")
            except Exception as e:
                print(f"Plugin error ({mod_name}): {e}")

        self._build_plugin_buttons()

    def _build_plugin_buttons(self):
        actions = self.plugins_menu.actions()
        for act in actions[2:]:
            self.plugins_menu.removeAction(act)

        if not self.plugins:
            a = self.plugins_menu.addAction("(no plugins found — drop .py into plugins/)")
            a.setEnabled(False)
            return

        for name, mod in self.plugins.items():
            params = getattr(mod, 'PARAMS', {})
            desc   = getattr(mod, 'DESCRIPTION', '')

            if params:
                act = self.plugins_menu.addAction(f"⚙ {name}…")
                act.setToolTip(desc)
                act.triggered.connect(
                    lambda checked, n=name, m=mod: self._open_plugin_dialog(n, m)
                )
            else:
                act = self.plugins_menu.addAction(f"▶ {name}")
                act.setToolTip(desc)
                act.triggered.connect(
                    lambda checked, n=name: self._run_plugin(n, {})
                )

    def _open_plugin_dialog(self, name, mod):
        from PyQt6.QtWidgets import QFormLayout
        params = getattr(mod, 'PARAMS', {})
        # Merge hardcoded defaults with any previously-entered values for this plugin
        cache  = self._plugin_params_cache.get(name, {})
        dlg  = QDialog(self)
        dlg.setWindowTitle(name)
        form = QFormLayout(dlg)
        widgets = {}
        for k, v in params.items():
            # Use cached value if present, otherwise fall back to module default
            display_val = cache.get(k, v)
            le = QLineEdit(str(display_val))
            form.addRow(f"{k}:", le)
            widgets[k] = le
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec():
            # Persist whatever the user typed before running
            self._plugin_params_cache[name] = {k: w.text().strip() for k, w in widgets.items()}
            self._run_plugin(name, widgets)

    def _run_plugin(self, name, param_widgets):
        if not self.file_list:
            self.info_label.setText("Load a file before running a plugin.")
            return

        mod = self.plugins.get(name)
        if mod is None: return

        params = {}
        for k, widget in param_widgets.items():
            raw = widget.text().strip()
            try:
                params[k] = float(raw) if '.' in raw else int(raw)
            except ValueError:
                params[k] = raw

        ctx = PluginContext(self)

        try:
            result = mod.run(ctx, **params)
            if isinstance(result, dict):
                summary = "\n".join(f"  {k}: {v}" for k, v in result.items())
                self.results_text.setPlainText(f"── Plugin: {name} ──\n{summary}")
                self._last_results = {"type": f"plugin:{name}", **{
                    k: v for k, v in result.items()
                    if not isinstance(v, (list, dict))
                }}
        except Exception as e:
            self.results_text.setPlainText(f"Plugin error [{name}]:\n{e}")
            import traceback; traceback.print_exc()


    # ── export traces to GIF ───────────────────────────────────────────────

    @staticmethod
    def _parse_sweep_range(text: str, max_sweep: int) -> list:
        """
        Parse a range string like "1-11, 21, 25-30" into a sorted list of
        0-based flat indices. Returns empty list on parse error.
        Accepts commas or spaces as separators, and single numbers or N-M ranges.
        """
        indices = set()
        for token in re.split(r'[\s,]+', text.strip()):
            if not token:
                continue
            if '-' in token:
                parts = token.split('-')
                if len(parts) == 2:
                    try:
                        lo, hi = int(parts[0]), int(parts[1])
                        for n in range(lo, hi + 1):
                            if 1 <= n <= max_sweep:
                                indices.add(n - 1)
                    except ValueError:
                        return []
            else:
                try:
                    n = int(token)
                    if 1 <= n <= max_sweep:
                        indices.add(n - 1)
                except ValueError:
                    return []
        return sorted(indices)

    def export_traces_gif(self):
        """Open a dialog to type sweep numbers/ranges and export as animated GIF."""
        if not self.file_list or self.total_sweeps == 0:
            self.info_label.setText("Load a file first.")
            return

        # default sweep range: current series if HEKA, else all
        if self.is_dat_mode and self.flat_traces:
            g_cur, s_cur, _, _ = self.flat_traces[self.current_index]
            series_nums = [str(i + 1) for i, (g, s, sw, t) in enumerate(self.flat_traces)
                           if g == g_cur and s == s_cur]
            if series_nums:
                first, last = series_nums[0], series_nums[-1]
                default_range = f"{first}-{last}" if first != last else first
            else:
                default_range = f"1-{self.total_sweeps}"
        else:
            default_range = f"1-{self.total_sweeps}"

        # ── build dialog ──────────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle("Export Traces to GIF")
        dlg.setMinimumWidth(400)
        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)

        # sweep range input
        lay.addWidget(QLabel(
            f"<b>Sweep numbers</b>  "
            f"<span style='color:#888;font-size:11px;'>"
            f"(1–{self.total_sweeps} available)</span>"
        ))
        range_hint = QLabel(
            "Use ranges and/or individual numbers, e.g.  <code>1-11, 21, 25-30</code>"
        )
        range_hint.setStyleSheet("color:#888; font-size:11px;")
        lay.addWidget(range_hint)

        range_edit = QLineEdit(default_range)
        range_edit.setPlaceholderText("e.g.  1-11  or  1, 5, 10-20, 25")
        range_edit.setStyleSheet("font-size:13px; padding:4px;")
        lay.addWidget(range_edit)

        # live feedback label showing how many sweeps will be included
        feedback = QLabel()
        feedback.setStyleSheet("color:#3498db; font-size:11px;")
        lay.addWidget(feedback)

        def update_feedback():
            idxs = self._parse_sweep_range(range_edit.text(), self.total_sweeps)
            if idxs:
                feedback.setText(f"✅  {len(idxs)} sweep(s) selected")
                feedback.setStyleSheet("color:#2ecc71; font-size:11px;")
            else:
                feedback.setText("⚠  No valid sweeps — check your input")
                feedback.setStyleSheet("color:#e74c3c; font-size:11px;")

        range_edit.textChanged.connect(update_feedback)
        update_feedback()

        # options
        form = QFormLayout()
        form.setSpacing(6)

        dur_spin = QDoubleSpinBox()
        dur_spin.setRange(0.05, 10.0)
        dur_spin.setSingleStep(0.05)
        dur_spin.setValue(0.15)
        dur_spin.setSuffix("  s / frame")
        dur_spin.setDecimals(2)
        form.addRow("Frame duration:", dur_spin)

        w_spin = QSpinBox()
        w_spin.setRange(200, 1920); w_spin.setValue(600); w_spin.setSuffix(" px")
        h_spin = QSpinBox()
        h_spin.setRange(100, 1080); h_spin.setValue(300); h_spin.setSuffix(" px")
        form.addRow("Width:", w_spin)
        form.addRow("Height:", h_spin)

        loop_check = QCheckBox("Loop forever")
        loop_check.setChecked(True)
        form.addRow("", loop_check)

        lay.addLayout(form)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)

        if not dlg.exec():
            return

        # ── parse sweep indices ───────────────────────────────────────────
        selected = self._parse_sweep_range(range_edit.text(), self.total_sweeps)
        if not selected:
            self.info_label.setText("No valid sweeps — check your range input.")
            return

        # ── ask for save path ─────────────────────────────────────────────
        path, _ = QFileDialog.getSaveFileName(
            self, "Save GIF", "traces.gif", "GIF Files (*.gif)"
        )
        if not path:
            return

        # ── render ────────────────────────────────────────────────────────
        try:
            from PIL import Image
        except ImportError:
            self.info_label.setText("Pillow not installed — run: pip install pillow")
            return

        frame_w  = w_spin.value()
        frame_h  = h_spin.value()
        duration = int(dur_spin.value() * 1000)
        loop     = 0 if loop_check.isChecked() else 1

        self.info_label.setText(f"Rendering {len(selected)} frames…")
        QApplication.processEvents()

        pil_frames = []

        for i, flat_idx in enumerate(selected):
            tmp_win = pg.GraphicsLayoutWidget()
            tmp_win.resize(frame_w, frame_h)
            tmp_win.setBackground('k')
            plot = tmp_win.addPlot()
            plot.getAxis('bottom').enableAutoSIPrefix(False)

            try:
                data, time_ms, disp_unit, label = self._load_sweep(flat_idx)
            except Exception:
                continue

            pen_c = '#e67e22' if self.is_dat_mode else ('#2ecc71' if self.is_h5_mode else 'w')
            plot.plot(time_ms, data, pen=pg.mkPen(pen_c, width=1))
            plot.setLabel('left',   label,  units=disp_unit)
            plot.setLabel('bottom', 'Time', units='ms')

            text = pg.TextItem(f"#{flat_idx + 1}", color=(180, 180, 180), anchor=(0, 0))
            plot.addItem(text)
            text.setPos(time_ms[0], float(np.nanmax(data)))

            plot.autoRange()
            tmp_win.show()
            QApplication.processEvents()

            qimg = tmp_win.grab().toImage().convertToFormat(
                tmp_win.grab().toImage().Format.Format_RGB888)
            buf  = QByteArray()
            qbuf = QBuffer(buf)
            qbuf.open(QBuffer.OpenModeFlag.WriteOnly)
            qimg.save(qbuf, "PNG")
            qbuf.close()

            from io import BytesIO
            pil_frame = Image.open(BytesIO(bytes(buf))).convert("RGB")
            pil_frame = pil_frame.resize((frame_w, frame_h), Image.LANCZOS)
            pil_frames.append(pil_frame)
            tmp_win.close()

            self.info_label.setText(f"Rendering… {i+1}/{len(selected)}")
            QApplication.processEvents()

        if not pil_frames:
            self.info_label.setText("No frames rendered.")
            return

        pil_frames[0].save(
            path, save_all=True, append_images=pil_frames[1:],
            loop=loop, duration=duration, optimize=False,
        )
        self.info_label.setText(
            f"GIF saved: {os.path.basename(path)}\n"
            f"{len(pil_frames)} frames · {dur_spin.value():.2f}s/frame"
        )


    # ── waterfall plot ─────────────────────────────────────────────────────

    # ══════════════════════════════════════════════════════════════════════
    # EPHYS FULL ANALYSIS — GUI methods
    # ══════════════════════════════════════════════════════════════════════

    def _parse_ea_range(self, text):
        """Parse sweep-range string like '1-11' or '1,3,5-9' → list of 0-based indices."""
        indices = []
        for part in text.split(','):
            part = part.strip()
            if '-' in part:
                try:
                    a, b = part.split('-')
                    indices.extend(range(int(a) - 1, int(b)))
                except ValueError:
                    pass
            elif part.isdigit():
                indices.append(int(part) - 1)
        # clamp to valid range
        return sorted(set(i for i in indices if 0 <= i < self.total_sweeps))

    def _parse_csteps_str(self, text):
        """Parse 'start:step:stop' or comma list into a numpy array of pA values."""
        text = text.strip()
        if ':' in text:
            parts = text.split(':')
            if len(parts) == 3:
                start, step, stop = float(parts[0]), float(parts[1]), float(parts[2])
                return np.arange(start, stop + 1e-9, step).astype(float)
        # fallback: comma list
        return np.array([float(x) for x in text.split(',') if x.strip()])

    def run_ephys_analysis_gui(self):
        """Show parameter dialog then run the full ephys analysis on chosen sweeps."""
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout,
                                     QDialogButtonBox, QLabel, QLineEdit,
                                     QDoubleSpinBox, QProgressDialog)
        from PyQt6.QtCore import Qt

        if not self.file_list:
            self.info_label.setText("No file loaded.")
            return

        # ── dialog ────────────────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle("Ephys Full Analysis — Parameters")
        dlg.setMinimumWidth(400)
        lay = QVBoxLayout(dlg)

        hdr = QLabel("<b>Select sweeps and configure analysis parameters</b>")
        hdr.setStyleSheet("color:#9b59b6; font-size:13px; padding-bottom:4px;")
        lay.addWidget(hdr)

        form = QFormLayout()
        form.setSpacing(6)

        range_edit = QLineEdit(self.ea_range_edit.text())
        form.addRow("Sweep range:", range_edit)

        csteps_edit = QLineEdit(self.ea_csteps_edit.text())
        form.addRow("Current steps (start:step:stop pA):", csteps_edit)

        stepstart_spin = QDoubleSpinBox(); stepstart_spin.setRange(0, 10000); stepstart_spin.setDecimals(1); stepstart_spin.setValue(250); stepstart_spin.setSuffix(" ms")
        form.addRow("Step start:", stepstart_spin)

        stepend_spin = QDoubleSpinBox(); stepend_spin.setRange(0, 10000); stepend_spin.setDecimals(1); stepend_spin.setValue(750); stepend_spin.setSuffix(" ms")
        form.addRow("Step end:", stepend_spin)

        ctau_edit = QLineEdit("-50")
        form.addRow("Tau step (pA):", ctau_edit)

        dvdt_spin = QDoubleSpinBox(); dvdt_spin.setRange(1, 1000); dvdt_spin.setValue(40); dvdt_spin.setSuffix(" mV/ms")
        form.addRow("AP dV/dt threshold:", dvdt_spin)

        lay.addLayout(form)

        # live feedback on sweep count
        fb = QLabel()
        fb.setStyleSheet("color:#2ecc71; font-size:11px;")
        lay.addWidget(fb)

        def update_fb():
            idxs = self._parse_ea_range(range_edit.text())
            if idxs:
                fb.setText(f"✅  {len(idxs)} sweep(s) selected  (#{idxs[0]+1} – #{idxs[-1]+1})")
                fb.setStyleSheet("color:#2ecc71; font-size:11px;")
            else:
                fb.setText("⚠  No valid sweeps")
                fb.setStyleSheet("color:#e74c3c; font-size:11px;")

        range_edit.textChanged.connect(update_fb)
        update_fb()

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)

        if not dlg.exec():
            return

        # ── collect sweeps ────────────────────────────────────────────────
        selected = self._parse_ea_range(range_edit.text())
        if not selected:
            self.info_label.setText("No valid sweeps selected.")
            return

        # persist to sidebar for next run
        self.ea_range_edit.setText(range_edit.text())
        self.ea_csteps_edit.setText(csteps_edit.text())

        csteps = self._parse_csteps_str(csteps_edit.text())
        if len(csteps) != len(selected):
            self.info_label.setText(
                f"⚠ Sweep count ({len(selected)}) ≠ csteps count ({len(csteps)}). Adjust the range or current-steps field so they match."
            )
            return

        self.info_label.setText(f"Loading {len(selected)} sweeps…")
        QApplication.processEvents()

        # load data matrix (samples × sweeps)
        all_data, time_ref = [], None
        for flat_idx in selected:
            try:
                d, t_ms, *_ = self._load_sweep(flat_idx)
            except Exception as e:
                self.info_label.setText(f"Load error sweep {flat_idx+1}: {e}")
                return
            if time_ref is None:
                time_ref = t_ms
            all_data.append(d)

        max_len = max(len(d) for d in all_data)
        v_mat   = np.column_stack([
            np.pad(d, (0, max_len - len(d)), constant_values=np.nan)
            if len(d) < max_len else d for d in all_data
        ]).astype(float)             # (samples, sweeps)

        # time in seconds
        t_sec = (time_ref / 1000.0).astype(float)

        self.info_label.setText("Running analysis…")
        QApplication.processEvents()

        try:
            res = run_ephys_analysis(
                v_mat.copy(), t_sec, csteps,
                stepstart_t=stepstart_spin.value() / 1000.0,
                stepend_t=stepend_spin.value() / 1000.0,
                ctau=float(ctau_edit.text()),
                AP_dv_dt_threshold=dvdt_spin.value(),
            )
        except Exception as e:
            self.info_label.setText(f"Analysis error: {e}")
            import traceback; traceback.print_exc()
            return

        self._ephys_results = res

        # ── show results ──────────────────────────────────────────────────
        lines = [
            f"RMP:          {res['mean_RMP']:.2f} mV",
            f"R_in:         {res['R_in']:.1f} MΩ",
            f"tau_m:        {res['tau_m']*1000:.2f} ms",
            f"Sag ratio:    {res['sag_ratio']:.3f}",
            f"Rheobase:     {res['rheobase']} pA",
            f"AP thresh:    {res['AP1_thresh_y']:.2f} mV",
            f"AP amp:       {res['AP_amp']:.2f} mV",
            f"AP HW:        {res['AP_HW']:.3f} ms",
            f"AP latency:   {res['AP_latency']:.2f} ms",
            f"fAHP amp:     {res['fAHP_amp']:.2f} mV",
            f"mAHP amp:     {res['mAHP_amp']:.2f} mV",
            f"DAP amp:      {res['DAP_amp']:.2f} mV",
            f"Mean ISI:     {res['mean_ISI_last']*1000:.2f} ms",
            f"CV ISI:       {res['cv_ISI_last']:.3f}",
            f"SF adapt:     {res['SF_adaptation_last']:.3f}",
            f"FI slope:     {res['FI_slope']:.3f} Hz/pA",
            f"Max AP freq:  {res['max_AP_freq']:.1f} Hz",
        ]
        self.results_text.setPlainText("\n".join(lines))
        self.info_label.setText(
            f"✅ Ephys analysis done  ({len(selected)} sweeps)\n"
            f"Rheobase: {res['rheobase']} pA  |  R_in: {res['R_in']:.1f} MΩ"
        )

        # ── open plot window ──────────────────────────────────────────────
        self._show_ephys_plots(res)

    def _show_ephys_plots(self, res):
        """Open a 2×3 pyqtgraph window showing all analysis plots."""
        t   = res['_t']
        v   = res['_v']
        cs  = res['_csteps']
        sf  = res['_sf']
        ss  = res['_stepstart']
        se  = res['_stepend']
        saf = res['_safety']

        win = pg.GraphicsLayoutWidget(title="Ephys Full Analysis")
        win.resize(1300, 750)
        win.setBackground('k')

        def mk(row, col, title, xlabel="Time (s)", ylabel="Vm (mV)"):
            p = win.addPlot(row=row, col=col)
            p.setTitle(title, color='w', size='11pt')
            p.setLabel('bottom', xlabel)
            p.setLabel('left', ylabel)
            p.getAxis('bottom').enableAutoSIPrefix(False)
            return p

        taustep  = res['_taustep']
        idx_m50  = int(np.searchsorted(cs, -50))

        # ── [0,0] -50 pA step ─────────────────────────────────────────────
        p = mk(0, 0, "Step: -50 pA")
        p.plot(t, v[:, idx_m50], pen=pg.mkPen('w', width=1))
        RMP_m50 = res['_RMP'][idx_m50]
        ss_m50  = res['_steadystate_mean'][idx_m50]
        p.addItem(pg.InfiniteLine(pos=RMP_m50, angle=0,
                                  pen=pg.mkPen('g', width=2, style=Qt.PenStyle.DashLine),
                                  label="RMP", labelOpts={"color":"g"}))
        p.addItem(pg.InfiniteLine(pos=ss_m50,  angle=0,
                                  pen=pg.mkPen('y', width=2, style=Qt.PenStyle.DashLine),
                                  label="SS",  labelOpts={"color":"y"}))
        t_seg   = res['_t_seg']
        fit_v   = res['_fit_vals']
        p.plot(t_seg + t[ss], fit_v, pen=pg.mkPen('r', width=2))

        # ── [1,0] R_in linear fit ─────────────────────────────────────────
        p2 = mk(1, 0, f"Rm = {res['R_in']:.1f} MΩ", xlabel="Current (pA)", ylabel="dVm (mV)")
        fs, fd = res['_fit_steps'], res['_fit_dv']
        p2.plot(fs, fd, pen=None, symbol='star', symbolSize=10,
                symbolBrush='w', symbolPen=pg.mkPen(None))
        p2.plot(fs, np.polyval(res['_P_R_in'], fs), pen=pg.mkPen('r', width=2, style=Qt.PenStyle.DashLine))

        # ── [0,1] rheobase sweep full ─────────────────────────────────────
        rheo_idx = res['_rheo_idx']
        off      = ss + saf - 1
        p3 = mk(0, 1, f"Rheobase: {res['rheobase']} pA")
        p3.plot(t, v[:, rheo_idx], pen=pg.mkPen('w', width=1))
        def add_star(p, xi, yi, color):
            p.plot([t[xi + off]], [yi], pen=None, symbol='star',
                   symbolSize=12, symbolBrush=color, symbolPen=pg.mkPen(None))
        add_star(p3, res['_AP1_thresh_x'], res['_AP1_thresh_y'], 'r')
        add_star(p3, res['_AP_max_x'],     res['AP_amp'] + res['_AP1_thresh_y'], 'g')
        add_star(p3, res['_fAHP_x'],       res['_fAHP_amp'] + res['_fAHP_thresh_y'], 'b')
        add_star(p3, res['_mAHP_x'],       res['mAHP_amp'] + res['_fAHP_thresh_y'], 'm')
        add_star(p3, res['_DAP_x'],        res['DAP_amp'] + res['_fAHP_amp'] + res['_fAHP_thresh_y'], 'c')
        hw_x = [t[res['_AP_HW_start'] + off - 1], t[res['_AP_HW_end'] + off - 1]]
        hw_y = [res['_AP_half_amp'], res['_AP_half_amp']]
        p3.plot(hw_x, hw_y, pen=pg.mkPen('y', width=2))

        # ── [1,1] zoomed AP — only plot the zoomed slice so autoRange works ──
        cx       = t[res['_AP1_thresh_x'] + off]
        zoom_lo  = cx - 0.003
        zoom_hi  = cx + 0.015
        zoom_mask = (t >= zoom_lo) & (t <= zoom_hi)
        t_zoom   = t[zoom_mask]
        v_zoom   = v[zoom_mask, rheo_idx]

        # helper: map full-trace index → time, then find nearest in zoom slice
        def add_star_zoom(p, xi, yi, color):
            tx = t[xi + off]
            p.plot([tx], [yi], pen=None, symbol='star',
                   symbolSize=12, symbolBrush=color, symbolPen=pg.mkPen(None))

        p4 = mk(1, 1, "AP zoom")
        p4.plot(t_zoom, v_zoom, pen=pg.mkPen('w', width=1))
        add_star_zoom(p4, res['_AP1_thresh_x'], res['_AP1_thresh_y'], 'r') 
        add_star_zoom(p4, res['_AP_max_x'],     res['AP_amp'] + res['_AP1_thresh_y'], 'g')
        add_star_zoom(p4, res['_fAHP_x'],       res['_fAHP_amp'] + res['_fAHP_thresh_y'], 'b')
        add_star_zoom(p4, res['_mAHP_x'],       res['mAHP_amp'] + res['_fAHP_thresh_y'], 'm')
        add_star_zoom(p4, res['_DAP_x'],        res['DAP_amp'] + res['_fAHP_amp'] + res['_fAHP_thresh_y'], 'c')
        p4.plot(hw_x, hw_y, pen=pg.mkPen('y', width=2))
        # autoRange will now fit the zoomed slice correctly

        # ── [0,2] max sweep — red dot at each spike peak ──────────────────
        p5 = mk(0, 2, f"Max step ({cs[-1]:.0f} pA)")
        p5.plot(t, v[:, -1], pen=pg.mkPen('w', width=1))
        AP_tx = res['_AP_thresh_x']
        last_s = v.shape[1] - 1
        AP_times_last = AP_tx[~np.isnan(AP_tx[:, last_s]), last_s].astype(int)
        if len(AP_times_last) > 0:
            AP_dur_samp = res['_stepend'] - res['_stepstart']  # reuse as safe search window
            peak_ts, peak_vs = [], []
            vs_last = v[ss + saf:se - saf, -1]
            for ap_xi in AP_times_last:
                win_end = min(ap_xi + round(sf * 0.003), len(vs_last))
                if ap_xi >= len(vs_last):
                    continue
                local_peak_i = int(np.argmax(vs_last[ap_xi:win_end])) + ap_xi
                peak_ts.append(t[local_peak_i + ss + saf - 1])
                peak_vs.append(float(vs_last[local_peak_i]))
            if peak_ts:
                p5.plot(peak_ts, peak_vs, pen=None, symbol='o',
                        symbolSize=8, symbolBrush='r', symbolPen=pg.mkPen(None))

        # ── [1,2] FI curve ────────────────────────────────────────────────
        p6 = mk(1, 2, "FI Curve", xlabel="Current (pA)", ylabel="AP Freq (Hz)")
        pos  = cs >= 0
        AP_f = res['_AP_freq']
        p6.plot(cs[pos], AP_f[pos], pen=None, symbol='star', symbolSize=10,
                symbolBrush='w', symbolPen=pg.mkPen(None))
        p6.plot(cs[pos], np.polyval(res['_P_FI'], cs[pos]),
                pen=pg.mkPen('r', width=2, style=Qt.PenStyle.DashLine))

        # autoRange every subplot (p4 now only contains the zoomed slice)
        for _p in [p, p2, p3, p4, p5, p6]:
            _p.autoRange()
            _p.enableAutoRange()

        # ── toolbar row: hint label + Save JPG button ────────────────
        win.addLabel(
            "<span style='color:#555; font-size:9pt'>"
            "Double-click any plot to auto-range &nbsp;·&nbsp; S = save JPG</span>",
            row=2, col=0, colspan=2
        )

        def _save_jpg():
            path, _ = QFileDialog.getSaveFileName(
                win, 'Save Ephys Plot', 'ephys_analysis.jpg',
                'JPEG (*.jpg *.jpeg);;PNG (*.png)'
            )
            if not path:
                return
            try:
                exp = pg.exporters.ImageExporter(win.scene())
                exp.parameters()['width'] = 2600
                exp.export(path)
            except Exception as _e:
                self.info_label.setText(f'Export error: {_e}')

        # embed a real QPushButton via a QGraphicsProxyWidget in col 2
        from PyQt6.QtWidgets import QPushButton as _QPB
        _save_btn = _QPB('💾  Save JPG')
        _save_btn.setStyleSheet(
            'background:#2c3e50; color:white; font-weight:bold;'
            ' padding:3px 10px; border-radius:3px;'
        )
        _save_btn.clicked.connect(_save_jpg)
        _proxy = win.scene().addWidget(_save_btn)
        _proxy.setPos(win.width() - 140, win.height() - 34)

        sc = QShortcut(QKeySequence('S'), win)
        sc.activated.connect(_save_jpg)

        if not hasattr(self, '_ephys_plot_windows'):
            self._ephys_plot_windows = []
        self._ephys_plot_windows.append((win, _proxy, _save_btn))  # keep all refs
        win.show()
        win.raise_()
        # reposition button now that the window has its final size
        _proxy.setPos(win.width() - 145, win.height() - 36)

    def save_ephys_results_csv(self):
        if not self._ephys_results:
            self.info_label.setText("No ephys results yet — run analysis first.")
            return
        res = self._ephys_results
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Ephys Results CSV", "ephys_results.csv",
            "CSV (*.csv)"
        )
        if not path:
            return
        keys = ["mean_RMP","R_in","tau_m","sag_ratio","rheobase",
                "AP1_thresh_y","AP_amp","AP_HW","AP_latency",
                "fAHP_amp","mAHP_amp","DAP_amp",
                "mean_ISI_last","cv_ISI_last","SF_adaptation_last",
                "FI_slope","max_AP_freq"]
        import csv
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(keys)
            w.writerow([res[k] for k in keys])
        self.info_label.setText(f"Saved: {os.path.basename(path)}")

    def show_waterfall(self):
        """Dialog to configure then open a waterfall overlay in a new window."""
        if not self.file_list or self.total_sweeps == 0:
            self.info_label.setText("Load a file first.")
            return

        # default range — current series for HEKA, all for others
        if self.is_dat_mode and self.flat_traces:
            g_cur, s_cur, _, _ = self.flat_traces[self.current_index]
            series_idxs = [i for i, (g, s, sw, t) in enumerate(self.flat_traces)
                           if g == g_cur and s == s_cur]
            if series_idxs:
                first, last = series_idxs[0] + 1, series_idxs[-1] + 1
                default_range = f"{first}-{last}" if first != last else str(first)
            else:
                default_range = f"1-{self.total_sweeps}"
        else:
            default_range = f"1-{min(self.total_sweeps, 20)}"

        # ── config dialog ─────────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle("Waterfall Plot")
        dlg.setMinimumWidth(380)
        lay = QVBoxLayout(dlg)
        lay.setSpacing(8)

        lay.addWidget(QLabel(
            f"<b>Sweep numbers</b>  "
            f"<span style='color:#888;font-size:11px;'>(1–{self.total_sweeps} available)</span>"
        ))
        hint = QLabel("Ranges and individual numbers, e.g.  <code>1-20</code>  or  <code>1, 5, 10-20</code>")
        hint.setStyleSheet("color:#888; font-size:11px;")
        lay.addWidget(hint)

        range_edit = QLineEdit(default_range)
        range_edit.setStyleSheet("font-size:13px; padding:4px;")
        lay.addWidget(range_edit)

        feedback = QLabel()
        feedback.setStyleSheet("color:#2ecc71; font-size:11px;")
        lay.addWidget(feedback)

        def update_fb():
            idxs = self._parse_sweep_range(range_edit.text(), self.total_sweeps)
            if idxs:
                feedback.setText(f"✅  {len(idxs)} sweep(s)")
                feedback.setStyleSheet("color:#2ecc71; font-size:11px;")
            else:
                feedback.setText("⚠  No valid sweeps")
                feedback.setStyleSheet("color:#e74c3c; font-size:11px;")

        range_edit.textChanged.connect(update_fb)
        update_fb()

        form = QFormLayout()
        form.setSpacing(6)

        # vertical offset between traces (0 = pure overlay)
        offset_spin = QDoubleSpinBox()
        offset_spin.setRange(0.0, 10000.0)
        offset_spin.setSingleStep(1.0)
        offset_spin.setDecimals(1)
        offset_spin.setValue(0.0)
        offset_spin.setSuffix("  (same units as signal)")
        offset_spin.setSpecialValueText("0  — pure overlay")
        form.addRow("Vertical offset:", offset_spin)

        # colour scheme
        from PyQt6.QtWidgets import QComboBox
        cmap_combo = QComboBox()
        cmap_combo.addItems(["Viridis", "Plasma", "Cool→Warm", "Orange mono",
                             "Green mono", "White mono"])
        form.addRow("Colour scheme:", cmap_combo)

        alpha_spin = QSpinBox()
        alpha_spin.setRange(10, 255)
        alpha_spin.setValue(200)
        alpha_spin.setSuffix("  (opacity)")
        form.addRow("Line opacity:", alpha_spin)

        lw_spin = QDoubleSpinBox()
        lw_spin.setRange(0.5, 5.0)
        lw_spin.setSingleStep(0.5)
        lw_spin.setValue(1.0)
        lw_spin.setDecimals(1)
        form.addRow("Line width:", lw_spin)

        lay.addLayout(form)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)

        if not dlg.exec():
            return

        selected = self._parse_sweep_range(range_edit.text(), self.total_sweeps)
        if not selected:
            self.info_label.setText("No valid sweeps — check your range input.")
            return

        n        = len(selected)
        offset   = offset_spin.value()
        scheme   = cmap_combo.currentText()
        alpha    = alpha_spin.value()
        lw       = lw_spin.value()

        # ── build colour ramp ─────────────────────────────────────────────
        def make_colors(n, scheme):
            colors = []
            for i in range(n):
                t = i / max(n - 1, 1)   # 0..1
                if scheme == "Viridis":
                    r = int(( 0.267 + 0.003*t + 0.331*t**2 - 0.63*t**3 + 1.0*t**4) * 255)
                    g = int(( 0.004 + 0.858*t - 0.311*t**2) * 255)
                    b = int(( 0.329 + 0.548*t - 1.44*t**2  + 1.17*t**3) * 255)
                elif scheme == "Plasma":
                    r = int((0.050 + 2.80*t - 2.60*t**2) * 255)
                    g = int((0.030 + 0.56*t - 0.16*t**2) * 255)
                    b = int((0.530 - 0.60*t + 0.07*t**2) * 255)
                elif scheme == "Cool→Warm":
                    r = int(t * 255)
                    g = int((1 - abs(2*t - 1)) * 200)
                    b = int((1 - t) * 255)
                elif scheme == "Orange mono":
                    v = int(80 + t * 175)
                    r, g, b = v, int(v * 0.55), 0
                elif scheme == "Green mono":
                    v = int(60 + t * 195)
                    r, g, b = 0, v, int(v * 0.3)
                else:  # White mono
                    v = int(120 + t * 135)
                    r, g, b = v, v, v
                colors.append((max(0,min(255,r)),
                                max(0,min(255,g)),
                                max(0,min(255,b))))
            return colors

        colors = make_colors(n, scheme)

        # ── open new plot window ──────────────────────────────────────────
        self.info_label.setText(f"Loading {n} sweeps…")
        QApplication.processEvents()

        win = pg.GraphicsLayoutWidget(title=f"Waterfall — {n} sweeps")
        win.resize(900, 550)
        win.setBackground('k')
        plot = win.addPlot()
        plot.getAxis('bottom').enableAutoSIPrefix(False)

        disp_unit_ref = 'mV'
        label_ref     = 'Signal'

        for i, flat_idx in enumerate(selected):
            try:
                data, time_ms, disp_unit, label = self._load_sweep(flat_idx)
            except Exception:
                continue

            if i == 0:
                disp_unit_ref = disp_unit
                label_ref     = label

            y = data + i * offset
            r, g, b = colors[i]
            pen = pg.mkPen((r, g, b, alpha), width=lw)
            plot.plot(time_ms, y, pen=pen,
                      name=f"#{flat_idx + 1}" if n <= 30 else None)

            self.info_label.setText(f"Loading… {i+1}/{n}")
            QApplication.processEvents()

        plot.setLabel('left',   label_ref, units=disp_unit_ref)
        plot.setLabel('bottom', 'Time',    units='ms')
        plot.setTitle(f"Waterfall  ·  {n} sweeps  ·  offset={offset:.1f}")

        if n <= 30:
            plot.addLegend(offset=(10, 10))

        plot.autoRange()

        # ── toolbar: Export JPG button lives inside the window ────────────
        # We embed a small export button via a proxy widget in the layout
        win.show()
        win.raise_()

        # keep reference so GC doesn't destroy it
        if not hasattr(self, '_waterfall_windows'):
            self._waterfall_windows = []
        self._waterfall_windows.append(win)

        # ── export JPG via a standalone dialog triggered from plot ────────
        # Add a keyboard shortcut: pressing S in the waterfall window saves it
        from PyQt6.QtGui import QKeySequence as QKS

        def save_waterfall_jpg():
            path, _ = QFileDialog.getSaveFileName(
                win, "Save Waterfall JPG", "waterfall.jpg",
                "JPEG (*.jpg *.jpeg);;PNG (*.png)"
            )
            if not path:
                return
            try:
                exporter = pg.exporters.ImageExporter(plot)
                exporter.parameters()['width']  = 1800
                exporter.export(path)
                self.info_label.setText(f"Waterfall saved: {os.path.basename(path)}")
            except Exception as e:
                self.info_label.setText(f"Export error: {e}")

        sc = QShortcut(QKS("S"), win)
        sc.activated.connect(save_waterfall_jpg)

        # also show a small hint label at the bottom of the window
        hint_proxy = pg.LabelItem(
            "Press  S  to save as JPG",
            color=(120, 120, 120), size="9pt"
        )
        win.addItem(hint_proxy, row=1, col=0)

        self.info_label.setText(
            f"Waterfall: {n} sweeps  ·  Press S in window to save JPG"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PatchBrowser()
    win.show()
    win.showMaximized()
    sys.exit(app.exec())