import colorsys
import math

import pygame


class LightEffects:
    """
    Ambient background effects: beat flash, pulsing bass rings, halo glow.
    Drawn underneath spectrum and waveform.
    """

    def __init__(self, surface: pygame.Surface):
        self.surface = surface
        self.w, self.h = surface.get_size()
        self.cx = self.w // 2
        self.cy = self.h // 2

        # Animated state
        self.beat_flash = 0.0
        self.circle_radius = 50.0
        self.hue = 0.0
        self.hue_velocity = 0.0

        self._overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)

    def draw(self, features: dict):
        bass: float = features["bass"]
        rms: float = features["rms"]
        is_beat: bool = features["is_beat"]
        is_onset: bool = features["is_onset"]
        centroid: float = features["centroid"]

        # Hue drifts toward the spectral centroid with inertia
        target_hue = centroid * 0.85
        delta = (target_hue - self.hue + 0.5) % 1.0 - 0.5
        self.hue_velocity = self.hue_velocity * 0.9 + delta * 0.006
        self.hue = (self.hue + self.hue_velocity) % 1.0

        # Beat flash
        if is_beat:
            self.beat_flash = min(1.0, self.beat_flash + 0.75)
        elif is_onset:
            self.beat_flash = min(1.0, self.beat_flash + 0.25)
        self.beat_flash *= 0.80

        # Bass circle radius
        target_r = 60 + bass * 260
        self.circle_radius = self.circle_radius * 0.72 + target_r * 0.28

        r_f, g_f, b_f = colorsys.hsv_to_rgb(self.hue, 0.85, 1.0)
        color = (int(r_f * 255), int(g_f * 255), int(b_f * 255))

        self._overlay.fill((0, 0, 0, 0))

        # --- Ambient radial glow (large, very transparent) ---
        glow_alpha = int(rms * 35 + self.beat_flash * 30)
        if glow_alpha > 0:
            pygame.draw.circle(
                self._overlay,
                (*color, glow_alpha),
                (self.cx, self.cy),
                int(min(self.w, self.h) * 0.48),
            )

        # --- Outer vignette rings (two complementary hues) ---
        comp_r, comp_g, comp_b = colorsys.hsv_to_rgb((self.hue + 0.5) % 1.0, 0.7, 1.0)
        comp = (int(comp_r * 255), int(comp_g * 255), int(comp_b * 255))

        for ring_idx in range(4):
            radius = int(self.circle_radius) - ring_idx * 22
            if radius <= 0:
                continue
            alpha = int((1.0 - ring_idx * 0.22) * bass * 190)
            width = max(1, 3 - ring_idx)
            ring_color = color if ring_idx % 2 == 0 else comp
            pygame.draw.circle(self._overlay, (*ring_color, alpha), (self.cx, self.cy), radius, width)

        # --- Beat flash full-screen overlay ---
        if self.beat_flash > 0.01:
            flash_alpha = int(self.beat_flash * 55)
            pygame.draw.rect(self._overlay, (*color, flash_alpha), (0, 0, self.w, self.h))

        # --- Corner light leaks (subtle chromatic aberration feel) ---
        corners = [(0, 0), (self.w, 0), (0, self.h), (self.w, self.h)]
        for cx_c, cy_c in corners:
            leak_alpha = int(rms * 25)
            if leak_alpha > 0:
                pygame.draw.circle(self._overlay, (*color, leak_alpha), (cx_c, cy_c), int(self.w * 0.35))

        self.surface.blit(self._overlay, (0, 0))
