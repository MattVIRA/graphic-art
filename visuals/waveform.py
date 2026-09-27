import pygame
import numpy as np


class WaveformVisual:
    """
    Oscilloscope strip — drawn as a horizontally mirrored waveform.
    Color shifts from cold blue to hot white based on amplitude.
    """

    def __init__(self, surface: pygame.Surface, rect: tuple):
        self.surface = surface
        self.rect = rect  # (x, y, w, h)

    def draw(self, features: dict):
        waveform: np.ndarray = features["waveform"]
        rms: float = features["rms"]
        high: float = features["high"]

        x, y, w, h = self.rect
        cy = y + h // 2
        n = len(waveform)

        # Downsample to fit width
        indices = np.linspace(0, n - 1, w).astype(int)
        samples = waveform[indices]

        # Color: cold blue → electric cyan → white as amplitude rises
        r = int(min(255, 20  + rms * 220))
        g = int(min(255, 140 + rms * 115))
        b = int(min(255, 220 + rms * 35))
        color = (r, g, b)

        # Glow color slightly dimmer
        glow_color = (r // 3, g // 3, b // 3)

        thickness = max(1, int(1 + high * 2.5))

        # Build point list
        points = [(x + i, cy - int(samples[i] * (h * 0.44))) for i in range(w)]

        if len(points) < 2:
            return

        # Glow pass (thicker, dimmer)
        if thickness > 1:
            pygame.draw.lines(self.surface, glow_color, False, points, thickness + 3)

        # Main line
        pygame.draw.lines(self.surface, color, False, points, thickness)
