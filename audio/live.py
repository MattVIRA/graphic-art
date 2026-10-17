"""
Live audio analyzer — same feature interface as AudioAnalyzer/AudioPlayer,
but computed in real time from a sounddevice input stream (microphone, or a
loopback device like Background Music / BlackHole to capture an app's output).

No offline pre-analysis is possible, so every feature is normalized against a
slowly-decaying running peak (the visuals adapt to the incoming level).
"""

import threading
import time

import numpy as np
import librosa
import sounddevice as sd


class LiveAnalyzer:
    N_MELS = 64
    N_FFT = 2048

    # (lo, hi) Hz — must match AudioAnalyzer
    _BANDS = {
        "bass": (20, 250), "low_mid": (250, 800), "mid": (800, 4000),
        "high": (4000, 20000), "hz_10_11": (10000, 11000),
        "hz_11_12": (11000, 12000), "hz_12_14": (12000, 14000),
    }
    _EQ_EDGES = [(20, 60), (60, 250), (250, 800), (800, 2500),
                 (2500, 6000), (6000, 10000), (10000, 16000)]

    def __init__(self, device=None, sr: int = 44100):
        self.sr = sr
        self.tempo = 0.0          # no offline beat-tracking in live mode
        self.is_paused = False

        self._lock = threading.Lock()
        self._ring = np.zeros(sr, dtype=np.float32)     # 1 s of audio
        self._write = 0

        self._win = np.hanning(self.N_FFT).astype(np.float32)
        self._mel_fb = librosa.filters.mel(
            sr=sr, n_fft=self.N_FFT, n_mels=self.N_MELS).astype(np.float32)
        freqs = np.fft.rfftfreq(self.N_FFT, 1.0 / sr)
        self._freqs = freqs
        self._masks = {k: (freqs >= lo) & (freqs < hi)
                       for k, (lo, hi) in self._BANDS.items()}
        self._eq_masks = [(freqs >= lo) & (freqs < hi) for lo, hi in self._EQ_EDGES]

        self._peaks = {}          # adaptive normalization state

        # ── T2: streaming beat tracking ──
        # onset-energy envelope sampled on the fixed 512-sample block grid
        # (~86 Hz), tempo by autocorrelation + log-tempo prior (Ellis),
        # beat phase by comb alignment.
        self._block_rate = sr / 512.0
        self._env = np.zeros(1024, dtype=np.float32)   # ~12 s of envelope
        self._blocks = 0
        self._prev_block_rms = 0.0
        self._bpm = 0.0
        self._beat_period = 0.0    # in blocks (float)
        self._beat_anchor = 0      # absolute block index of a beat
        self._last_est = 0
        self._prev_phase = 0.0
        self._beat_flag = False
        # transient tracking (HF spectral flux)
        self._hf_ema = 1e-6
        self._events = []
        self._last_evt = 0.0
        self._density_peak = 1e-6
        self._transients = 0.0
        # percussive hits (spectral flux of a band vs its running mean)
        self._hit_masks = {"kick": (freqs >= 30) & (freqs < 150),
                           "snare": (freqs >= 1500) & (freqs < 8000)}
        self._hit_state = {k: {"prev": 0.0, "mean": 1e-6, "last": 0.0,
                               "id": 0, "amp": 0.0} for k in self._hit_masks}
        # section energy: slow RMS envelope against a slow min / max
        self._sec = 0.0
        self._sec_lo = None
        self._sec_hi = 1e-6

        channels = 1
        try:
            info = sd.query_devices(device, "input")
            channels = max(1, min(2, int(info["max_input_channels"])))
        except Exception:
            pass
        self._stream = sd.InputStream(device=device, channels=channels,
                                      samplerate=sr, blocksize=512,
                                      callback=self._callback)
        self._stream.start()

    # ------------------------------------------------------------------
    # Stream

    def _callback(self, indata, frames, t, status):
        mono = indata.mean(axis=1).astype(np.float32)
        # beat envelope: positive energy flux per block (kicks spike it)
        block_rms = float(np.sqrt(np.mean(mono ** 2)))
        flux = max(0.0, block_rms - self._prev_block_rms)
        self._prev_block_rms = 0.3 * block_rms + 0.7 * self._prev_block_rms
        self._env[self._blocks % len(self._env)] = flux
        self._blocks += 1
        with self._lock:
            n, size, i = len(mono), len(self._ring), self._write
            end = i + n
            if end <= size:
                self._ring[i:end] = mono
            else:
                k = size - i
                self._ring[i:] = mono[:k]
                self._ring[:end - size] = mono[k:]
            self._write = end % size

    def _latest(self, n: int) -> np.ndarray:
        with self._lock:
            w = self._write
            buf = np.concatenate([self._ring[w:], self._ring[:w]])
        return buf[-n:]

    # ------------------------------------------------------------------
    # Player-compatible interface

    @property
    def is_done(self) -> bool:
        return False

    def toggle_pause(self):
        pass                       # live stream: nothing to pause

    def stop(self):
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # T2: streaming tempo estimation

    def _estimate_tempo(self):
        size = len(self._env)
        n = min(self._blocks, size)
        if n < 300:                    # need ~3.5 s of envelope
            return
        # unroll the ring: oldest → newest
        w = self._blocks % size
        env = np.concatenate([self._env[w:], self._env[:w]])[-n:]
        # detrend (remove slow energy drift), keep the onset spikes
        kernel = np.ones(9, dtype=np.float32) / 9.0
        env = np.clip(env - np.convolve(env, kernel, mode="same"), 0.0, None)
        if env.max() < 1e-6:
            return

        ac = np.correlate(env, env, mode="full")[n - 1:]
        lo = int(self._block_rate * 60.0 / 180.0)      # 180 BPM
        hi = min(int(self._block_rate * 60.0 / 60.0), n - 2)   # 60 BPM
        if hi <= lo + 2:
            return
        lags = np.arange(lo, hi)
        bpms = 60.0 * self._block_rate / lags
        # log-tempo prior centered on 120 BPM (Ellis) → avoids octave errors
        prior = np.exp(-0.5 * (np.log2(bpms / 120.0) / 0.9) ** 2)
        score = ac[lo:hi] * prior
        k = int(np.argmax(score))
        lag = float(lags[k])
        # parabolic refinement around the peak
        if 0 < k < len(score) - 1:
            a, b, c = score[k - 1], score[k], score[k + 1]
            denom = a - 2 * b + c
            if abs(denom) > 1e-9:
                lag += 0.5 * (a - c) / denom

        bpm = 60.0 * self._block_rate / lag
        # smooth: track gradually if close, jump if clearly different
        self._bpm = 0.8 * self._bpm + 0.2 * bpm \
            if self._bpm and abs(bpm - self._bpm) / self._bpm < 0.12 else bpm
        self._beat_period = 60.0 * self._block_rate / self._bpm

        # phase: comb alignment — offset whose beat grid gathers the most energy
        ilag = max(2, int(round(self._beat_period)))
        offsets = [env[o::ilag].sum() for o in range(ilag)]
        best = int(np.argmax(offsets))
        start_abs = self._blocks - n
        k_last = (self._blocks - (start_abs + best)) // ilag
        self._beat_anchor = start_abs + best + k_last * ilag

    # ------------------------------------------------------------------
    # Features

    def _norm(self, key: str, val: float, decay: float = 0.9985) -> float:
        """Normalize against a slowly-decaying running peak."""
        pk = max(self._peaks.get(key, 1e-6) * decay, val, 1e-6)
        self._peaks[key] = pk
        return float(val / pk)

    def get_current_features(self) -> dict:
        buf = self._latest(self.N_FFT)
        raw_rms = float(np.sqrt(np.mean(buf ** 2)))
        silent = raw_rms < 1e-4

        spec = np.abs(np.fft.rfft(buf * self._win)).astype(np.float32)

        f = {}
        for key, mask in self._masks.items():
            v = float(spec[mask].mean()) if not silent else 0.0
            f[key] = self._norm(key, v) if not silent else 0.0
        f["rms"] = self._norm("rms", raw_rms) if not silent else 0.0

        # spectral centroid → hue drive
        total = spec.sum() + 1e-8
        f["centroid"] = float((self._freqs * spec).sum() / total / (self.sr / 2.0)) \
            if not silent else 0.0

        # mel spectrum, normalized per-frame then gated by level
        if silent:
            f["spectrum"] = np.zeros(self.N_MELS, dtype=np.float32)
            f["eq_bands"] = np.zeros(len(self._eq_masks), dtype=np.float32)
        else:
            mel = self._mel_fb @ (spec ** 2)
            mel_db = 10.0 * np.log10(mel + 1e-10)
            rng = mel_db.max() - mel_db.min() + 1e-6
            f["spectrum"] = ((mel_db - mel_db.min()) / rng
                             * min(1.0, f["rms"] * 2.0)).astype(np.float32)
            f["eq_bands"] = np.array(
                [self._norm(f"eq{i}", float(spec[m].mean()))
                 for i, m in enumerate(self._eq_masks)], dtype=np.float32)

        # HF transient detection → rolling density (replaces offline analysis)
        now = time.monotonic()
        hf = float(spec[self._masks["high"]].mean())
        self._hf_ema += 0.05 * (hf - self._hf_ema)
        if (not silent and hf > self._hf_ema * 1.6
                and now - self._last_evt > 0.06 and f["rms"] > 0.08):
            self._events.append(now)
            self._last_evt = now
        self._events = [t for t in self._events if now - t < 1.5]
        d = float(len(self._events))
        self._density_peak = max(self._density_peak * 0.9995, d, 1e-6)
        target = d / self._density_peak
        self._transients += 0.06 * (target - self._transients)
        f["transients"] = float(self._transients)

        # tempo + beat phase (re-estimated every ~2 s)
        if self._blocks - self._last_est > int(self._block_rate * 2.0):
            self._last_est = self._blocks
            if not silent:
                self._estimate_tempo()
        if self._beat_period > 1.0 and not silent:
            phase = ((self._blocks - self._beat_anchor) / self._beat_period) % 1.0
            f["is_beat"] = phase < self._prev_phase          # phase wrapped
            self._prev_phase = phase
            f["beat_phase"] = float(phase)
            f["bpm"] = float(self._bpm)
            self.tempo = float(self._bpm)                    # HUD
        else:
            f["is_beat"] = False
            f["beat_phase"] = 0.0
            f["bpm"] = 0.0
        f["is_onset"] = bool(self._events and now - self._events[-1] < 0.05)

        # percussive hits: band flux well above its running mean, with a
        # refractory period (kick 120 ms, snare 90 ms)
        for key, refract in (("kick", 0.12), ("snare", 0.09)):
            st = self._hit_state[key]
            e = float(spec[self._hit_masks[key]].mean()) if not silent else 0.0
            flux = max(0.0, e - st["prev"])
            st["prev"] = e
            st["mean"] += 0.05 * (flux - st["mean"])
            if (not silent and flux > st["mean"] * 2.5 and flux > 1e-5
                    and now - st["last"] > refract):
                st["last"] = now
                st["id"] += 1
                st["amp"] = min(1.0, self._norm(key + "_flux", flux, 0.999))
            f[key + "_id"] = st["id"]
            f[key + "_amp"] = st["amp"]

        # section energy (~3 s at 60 fps), stretched between a slow floor and
        # a slow ceiling → ~0 in breaks, ~1 on drops
        self._sec += 0.005 * (raw_rms - self._sec)
        if self._sec_lo is None:
            self._sec_lo = self._sec
        self._sec_lo = min(self._sec, self._sec_lo + (self._sec - self._sec_lo) * 0.0005)
        self._sec_hi = max(self._sec, self._sec_hi * 0.9997)
        f["section"] = float(np.clip((self._sec - self._sec_lo)
                                     / (self._sec_hi - self._sec_lo + 1e-6), 0.0, 1.0)) \
            if not silent else 0.0
        f["frame_idx"] = 0
        f["waveform"] = buf
        return f
