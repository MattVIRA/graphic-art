import numpy as np
import librosa


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
        self.low_mid = band_energy(250,   800)
        self.mid     = band_energy(800,   4000)
        self.high    = band_energy(4000,  20000)

        # 7-band EQ split (SUB → TREBLE) for the user-routable physics matrix
        self.EQ_BANDS = ["SUB", "BASS", "LOW MID", "MID", "HIGH MID", "PRESENCE", "TREBLE"]
        edges = [(20, 60), (60, 250), (250, 800), (800, 2500),
                 (2500, 6000), (6000, 10000), (10000, 16000)]
        self.eq_bands = np.stack([band_energy(lo, hi) for lo, hi in edges])  # (7, n)

        # Narrow kHz accent bands (for fine-grained reactivity)
        self.hz_10_11 = band_energy(10000, 11000)
        self.hz_11_12 = band_energy(11000, 12000)
        self.hz_12_14 = band_energy(12000, 14000)

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

        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hl)
        self.onset_set = set(onset_frames.tolist())

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

        # Spectral centroid normalized to [0, 1]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=self.n_fft, hop_length=hl)[0]
        self.centroid = centroid / (sr / 2.0)

        # Mel spectrogram — shape (N_MELS, n_frames), normalized to [0, 1]
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.N_MELS, hop_length=hl)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        lo, hi = mel_db.min(), mel_db.max()
        self.mel_norm = (mel_db - lo) / (hi - lo + 1e-8)

        self.n_frames = self.mel_norm.shape[1]

    # ------------------------------------------------------------------
    # Public

    def ms_to_frame(self, ms: float) -> int:
        samples = int(ms / 1000.0 * self.sr)
        return min(samples // self.hop_length, self.n_frames - 1)

    def get_features(self, frame_idx: int) -> dict:
        i = max(0, min(frame_idx, self.n_frames - 1))
        return {
            "bass":     float(self.bass[i]),
            "low_mid":  float(self.low_mid[i]),
            "mid":      float(self.mid[i]),
            "high":     float(self.high[i]),
            "hz_10_11": float(self.hz_10_11[i]),
            "hz_11_12": float(self.hz_11_12[i]),
            "hz_12_14": float(self.hz_12_14[i]),
            "rms":      float(self.rms[i]),
            "transients": float(self.transient_density[min(i, len(self.transient_density) - 1)]),
            "is_beat":  i in self.beat_set,
            "bpm":      self.tempo,
            "beat_phase": float(self.beat_phase[min(i, len(self.beat_phase) - 1)]),
            "is_onset": i in self.onset_set,
            "centroid": float(self.centroid[i]),
            "spectrum": self.mel_norm[:, i].copy(),   # shape (N_MELS,)
            "eq_bands": self.eq_bands[:, i].copy(),   # shape (7,) SUB→TREBLE
            "frame_idx": i,
        }

    def get_waveform(self, frame_idx: int, n_samples: int = 2048) -> np.ndarray:
        center = frame_idx * self.hop_length
        half = n_samples // 2
        start = max(0, center - half)
        end = start + n_samples
        chunk = self.y[start:min(end, len(self.y))]
        if len(chunk) < n_samples:
            chunk = np.pad(chunk, (0, n_samples - len(chunk)))
        return chunk
