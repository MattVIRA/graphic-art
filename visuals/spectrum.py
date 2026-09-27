import math
import colorsys

import pygame
import numpy as np


class SpectrumVisual:
    """
    Radial (circular) spectrum — frequency bars radiate outward from the center.
    360° represents the full mel-frequency range.
    Each bar has a peak marker with slow decay.
    """

    INNER_RADIUS = 80
    MAX_BAR_LEN = 220
    PEAK_DECAY = 0.012

    def __init__(self, surface: pygame.Surface, center: tuple, n_bars: int = 64):
        self.surface = surface
        self.cx, self.cy = center
        self.n_bars = n_bars

        self.peaks = np.zeros(n_bars)

        # Pre-compute angles
        self._angles = [
            (i / n_bars) * 2 * math.pi - math.pi / 2
            for i in range(n_bars)
        ]

        # Reusable overlay for per-frame alpha effects
        self._overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)

    def draw(self, features: dict):
        spectrum: np.ndarray = features["spectrum"]  # shape (N_MELS,)
        centroid: float = features["centroid"]

        # Resize spectrum to n_bars
        if len(spectrum) != self.n_bars:
            indices = np.linspace(0, len(spectrum) - 1, self.n_bars).astype(int)
            spectrum = spectrum[indices]

        # Update peak decay
        self.peaks = np.where(spectrum > self.peaks, spectrum, self.peaks - self.PEAK_DECAY)
        self.peaks = np.clip(self.peaks, 0, 1)

        inner_r = self.INNER_RADIUS
        max_len = self.MAX_BAR_LEN
        cx, cy = self.cx, self.cy

        self._overlay.fill((0, 0, 0, 0))

        for i in range(self.n_bars):
            angle = self._angles[i]
            val = float(spectrum[i])
            peak = float(self.peaks[i])

            if val < 0.01:
                continue

            bar_len = inner_r + int(val * max_len)
            peak_len = inner_r + int(peak * max_len)

            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            x1 = int(cx + cos_a * inner_r)
            y1 = int(cy + sin_a * inner_r)
            x2 = int(cx + cos_a * bar_len)
            y2 = int(cy + sin_a * bar_len)

            # Hue: shifts with bar position + spectral centroid offset
            hue = ((i / self.n_bars) + centroid * 0.3) % 1.0
            sat = 0.7 + val * 0.3
            r_f, g_f, b_f = colorsys.hsv_to_rgb(hue, sat, 1.0)
            bar_color = (int(r_f * 255), int(g_f * 255), int(b_f * 255))

            # Glow (wide, semi-transparent)
            glow_alpha = int(val * 80)
            glow_color = (*bar_color, glow_alpha)
            pygame.draw.line(self._overlay, glow_color, (x1, y1), (x2, y2), 5)

            # Main bar
            pygame.draw.line(self._overlay, (*bar_color, 220), (x1, y1), (x2, y2), 2)

            # Peak dot
            px = int(cx + cos_a * peak_len)
            py = int(cy + sin_a * peak_len)
            pygame.draw.circle(self._overlay, (255, 255, 255, 200), (px, py), 2)

        # Inner circle fill (subtle center glow)
        pygame.draw.circle(self._overlay, (30, 30, 60, 60), (cx, cy), inner_r)

        self.surface.blit(self._overlay, (0, 0))
