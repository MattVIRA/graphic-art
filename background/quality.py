"""Niveaux de qualité du fond (T1.16) et résolution automatique.

Un niveau = des `#define` du shader (nombre de pas, ombres, occlusion) :
chaque niveau est un programme compilé à part, au démarrage. « Auto » prend
le niveau Moyen et baisse la résolution du fond par crans quand le temps
GPU mesuré dépasse le budget, puis la remonte quand il y a de la marge.
"""

# niveaux Bas, Moyen et Haut ; Auto utilise le niveau Moyen. Les ombres du
# relief vont loin (une montagne sur la vallée) : plus de pas qu'avant, chacun
# ne lit que les cartes de hauteur. La marche aussi : le Volume (warp,
# cavités) raccourcit les pas, à 128 les rayons rasants s'épuisaient en
# stries au loin ; seuls les rayons qui en ont besoin vont jusqu'au bout
LEVELS = [
    {"MAX_STEPS": 96, "SHADOW_STEPS": 0, "AO_SAMPLES": 0, "HIT_PIX": 3.0},
    {"MAX_STEPS": 192, "SHADOW_STEPS": 24, "AO_SAMPLES": 3, "HIT_PIX": 1.0},
    {"MAX_STEPS": 288, "SHADOW_STEPS": 40, "AO_SAMPLES": 5, "HIT_PIX": 1.0},
]
AUTO = 3
BUDGET_MS = 2.0          # temps GPU visé pour le fond
OVER, UNDER = 1.15, 0.55  # hystérésis : au-dessus → on baisse, en dessous → on remonte
DOWN_AFTER_S = 0.5       # dépassement soutenu avant de baisser d'un cran
UP_AFTER_S = 2.0         # marge soutenue avant de remonter


def level(quality: int) -> int:
    """Index dans LEVELS du programme à utiliser."""
    return 1 if quality == AUTO else min(max(int(quality), 0), len(LEVELS) - 1)


class AutoResolution:
    """Choisit l'index de résolution effectif (0 = ¼ … ) en mode Auto."""

    def __init__(self, budget_ms: float = BUDGET_MS):
        self.budget = budget_ms
        self.wanted = None
        self.current = 0
        self._over = self._under = 0.0

    def step(self, wanted: int, auto: bool, gpu_ms, dt: float) -> int:
        """wanted : index choisi dans le panneau ; gpu_ms : dernière mesure
        (None = pas encore de mesure)."""
        if wanted != self.wanted or not auto:
            self.wanted = self.current = wanted
            self._over = self._under = 0.0
            return self.current
        if gpu_ms is None:
            return self.current
        if gpu_ms > self.budget * OVER and self.current > 0:
            self._over += dt
            self._under = 0.0
            if self._over >= DOWN_AFTER_S:
                self.current -= 1
                self._over = 0.0
        elif gpu_ms < self.budget * UNDER and self.current < wanted:
            self._under += dt
            self._over = 0.0
            if self._under >= UP_AFTER_S:
                self.current += 1
                self._under = 0.0
        else:
            self._over = self._under = 0.0
        return self.current
