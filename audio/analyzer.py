import librosa
import numpy as np


class AudioAnalyzer:
    """Pre-analyzes an audio file into per-frame features for real-time playback sync."""

    N_MELS = 64
    HOP_LENGTH = 512
    N_FFT = 2048

    def __init__(self, filepath: str, sr: int = 44100):
        self.sr = sr
        self.hop_length = self.HOP_LENGTH
        self.n_fft = self.N_FFT

        print("  Loading audio...")
        self.y, self.sr = librosa.load(filepath, sr=sr, mono=True)
        self.duration = len(self.y) / self.sr

        print("  Analyzing spectrum and beats...")
        self._analyze()
        print(f"  Done — {self.duration:.1f}s  |  {self.n_frames} frames  |  {self.tempo:.0f} BPM")

    # ------------------------------------------------------------------
    # Private

    def _analyze(self):
        y, sr, hl = self.y, self.sr, self.hop_length

        # STFT magnitude
        D = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=hl))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)

        def band_energy(lo: float, hi: float) -> np.ndarray:
            mask = (freqs >= lo) & (freqs < hi)
            e = D[mask, :].mean(axis=0)
            return e / (e.max() + 1e-8)

        self.bass    = band_energy(20,    250)
        self.high    = band_energy(4000,  20000)

        # 7-band EQ split (SUB → TREBLE) for the user-routable physics matrix
        self.EQ_BANDS = ["SUB", "BASS", "LOW MID", "MID", "HIGH MID", "PRESENCE", "TREBLE"]
        edges = [(20, 60), (60, 250), (250, 800), (800, 2500),
                 (2500, 6000), (6000, 10000), (10000, 16000)]
        self.eq_bands = np.stack([band_energy(lo, hi) for lo, hi in edges])  # (7, n)

        # RMS energy
        rms = librosa.feature.rms(y=y, hop_length=hl)[0]
        self.rms = rms / (rms.max() + 1e-8)

        # Beat tracking
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hl)
        self.tempo = float(np.atleast_1d(tempo)[0])
        self.beat_set = set(beat_frames.tolist())

        # beat phase 0→1 between consecutive beats (interpolated beat count)
        bf = np.asarray(beat_frames, dtype=float)
        if len(bf) >= 2:
            counts = np.interp(np.arange(D.shape[1]), bf, np.arange(len(bf)))
            self.beat_phase = (counts % 1.0).astype(np.float32)
        else:
            self.beat_phase = np.zeros(D.shape[1], dtype=np.float32)

        # Transient density: HF transients counted over a sliding beat window.
        # High during drops / busy drum sections, low during breaks → drives
        # the rotation speed.
        onset_hf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hl, fmin=2000)
        hf_peaks = librosa.onset.onset_detect(
            onset_envelope=onset_hf, sr=sr, hop_length=hl, delta=0.12, wait=2)
        impulses = np.zeros(len(onset_hf), dtype=np.float32)
        impulses[hf_peaks] = 1.0
        beat_frames_len = max(4, int(round((60.0 / max(self.tempo, 1.0)) * sr / hl)))
        win = beat_frames_len * 2                       # 2-beat window
        kernel = np.ones(win, dtype=np.float32)
        density = np.convolve(impulses, kernel, mode="same")
        ref = np.percentile(density, 92) + 1e-8
        density = np.clip(density / ref, 0.0, 1.0)
        # EMA smoothing so the speed ramps instead of jumping
        alpha = 0.06
        for i in range(1, len(density)):
            density[i] = density[i - 1] + alpha * (density[i] - density[i - 1])
        self.transient_density = density

        # Percussive hits for the "choreographed" layer: kick = onsets of the
        # low band, snare/clap = onsets of the upper mids. Each frame carries
        # the id and strength of the latest hit, so the 60 fps render loop
        # never misses one even if it skips analysis frames.
        self.kick_id, self.kick_amp = self._hits(D, freqs, sr, hl, 30, 150, wait=8)
        self.snare_id, self.snare_amp = self._hits(D, freqs, sr, hl, 1500, 8000, wait=5)

        # Section energy: RMS averaged over ~4 s, stretched to the track's own
        # range → ~0 in breaks / intros, ~1 on drops (macro-dynamics)
        k = max(1, int(4.0 * sr / hl))
        slow = np.convolve(rms, np.ones(k) / k, mode="same")
        lo, hi = np.percentile(slow, 5), np.percentile(slow, 95)
        self.section = np.clip((slow - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)

        # Mel spectrogram — shape (N_MELS, n_frames), normalized to [0, 1]
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.N_MELS, hop_length=hl)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        lo, hi = mel_db.min(), mel_db.max()
        self.mel_norm = (mel_db - lo) / (hi - lo + 1e-8)

        self.n_frames = self.mel_norm.shape[1]

    @staticmethod
    def _hits(D, freqs, sr, hl, fmin, fmax, wait):
        """(id, amp) per frame: running count of onsets in [fmin, fmax] Hz and
        the normalized strength of the most recent one (log spectral flux)."""
        band = np.log1p(D[(freqs >= fmin) & (freqs < fmax), :] * 10.0)
        n = D.shape[1]
        env = np.zeros(n, dtype=np.float32)
        env[1:] = np.maximum(np.diff(band, axis=1), 0.0).sum(axis=0)
        env /= env.max() + 1e-8
        peaks = librosa.onset.onset_detect(onset_envelope=env, sr=sr,
                                           hop_length=hl, delta=0.1, wait=wait)
        if len(peaks) == 0:
            return np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.float32)
        strength = env[peaks] / (np.percentile(env[peaks], 90) + 1e-8)
        last = np.searchsorted(peaks, np.arange(n), side="right") - 1
        amp = np.where(last >= 0, np.clip(strength, 0.0, 1.0)[last], 0.0)
        return (last + 1).astype(np.int32), amp.astype(np.float32)

    # ------------------------------------------------------------------
    # Public

    def ms_to_frame(self, ms: float) -> int:
        samples = int(ms / 1000.0 * self.sr)
        return min(samples // self.hop_length, self.n_frames - 1)

    def get_features(self, frame_idx: int) -> dict:
        i = max(0, min(frame_idx, self.n_frames - 1))
        return {
            "bass":     float(self.bass[i]),
            "high":     float(self.high[i]),
            "rms":      float(self.rms[i]),
            "transients": float(self.transient_density[min(i, len(self.transient_density) - 1)]),
            "is_beat":  i in self.beat_set,
            "bpm":      self.tempo,
            "beat_phase": float(self.beat_phase[min(i, len(self.beat_phase) - 1)]),
            "spectrum": self.mel_norm[:, i].copy(),   # shape (N_MELS,)
            "eq_bands": self.eq_bands[:, i].copy(),   # shape (7,) SUB→TREBLE
            "kick_id":  int(self.kick_id[i]),
            "kick_amp": float(self.kick_amp[i]),
            "snare_id": int(self.snare_id[i]),
            "snare_amp": float(self.snare_amp[i]),
            "section":  float(self.section[min(i, len(self.section) - 1)]),
            "frame_idx": i,
        }
