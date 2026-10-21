"""HUD : une ligne d'état en haut de la fenêtre (texte pygame → texture GL)."""

import pygame


class Hud:
    """Renders a one-line status bar to a pygame surface → uploaded as a GL texture.
    Drawn at the framebuffer resolution (scale = pixel ratio) for sharp text."""

    HEIGHT = 40     # logical px

    def __init__(self, width: int, scale: float = 1.0):
        self.scale = scale
        self.width = int(width * scale)
        self.height = int(self.HEIGHT * scale)
        self.font = pygame.font.SysFont("monospace", int(14 * scale))
        self.surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self._last = None

    def build(self, features, tempo, paused, mode, n, notice=None):
        """`notice` (T1.15) : message d'erreur affiché en rouge à la place de
        l'aide, le temps que l'appelant le garde."""
        line = (f"BPM {tempo:.0f}  |  RMS {features['rms']:.2f}  |  "
                f"BASS {features['bass']:.2f}  |  {(mode or '…').upper()}  |  {n} pts")
        hint = ("SPACE pause   F plein écran   Cmd/Ctrl+glisser orbite   "
                "Maj+glisser déplacer   molette zoom   C vue de départ   "
                "U tore   ESC quitter")
        if paused:
            line += "   [PAUSED]"
        key = (line, hint, notice)
        if key == self._last:
            return self.surf
        self._last = key
        self.surf.fill((0, 0, 0, 0))
        k = self.scale
        self.surf.blit(self.font.render(line, True, (120, 120, 150)),
                       (int(12 * k), int(4 * k)))
        second, color = (f"ERREUR  {notice}", (235, 110, 110)) if notice else (hint, (70, 60, 80))
        self.surf.blit(self.font.render(second, True, color),
                       (int(12 * k), int(21 * k)))
        return self.surf

    def to_bytes(self, surf):
        # flipped=True → GL bottom-up origin so the strip displays upright
        return pygame.image.tostring(surf, "RGBA", True)
