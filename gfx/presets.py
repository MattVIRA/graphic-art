"""
Stockage des presets (T24) — sans tkinter, testable seul.

Fichiers à la racine du projet :
    presets.default.json   versionné : presets de départ, copié au 1er lancement
    presets.json           données utilisateur (gitignoré) : presets globaux
                           nommés, "_tab_presets", "_autosave", "_window"
    presets.backup.json    copie de presets.json avant chaque écriture
    presets.corrupt-<date>.json   presets.json illisible, mis de côté

Écriture atomique (fichier temporaire + os.replace) : un crash pendant
l'écriture ne laisse jamais un presets.json tronqué.

Presets intégrés (codés dans gfx/controls.py) : jamais copiés dans le
fichier, fusionnés à l'affichage → un preset intégré ajouté plus tard
apparaît chez tout le monde.
"""

import json
import os
import shutil
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER_PATH = os.path.join(ROOT, "presets.json")
DEFAULT_PATH = os.path.join(ROOT, "presets.default.json")
BACKUP_PATH = os.path.join(ROOT, "presets.backup.json")

TAB_KEY = "_tab_presets"


def _read(path: str):
    """dict lu depuis `path` ; None si absent ; ValueError si illisible."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(str(e)) from e
    if not isinstance(d, dict):
        raise ValueError("le contenu n'est pas un objet JSON")
    return d


def _read_or_none(path: str):
    try:
        return _read(path)
    except ValueError:
        return None


def _write_atomic(path: str, d: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load() -> tuple[dict, str | None]:
    """Contenu de presets.json + message d'avertissement éventuel.

    Premier lancement : presets.default.json est copié. Fichier illisible :
    renommé en presets.corrupt-<date>.json, puis restauré depuis
    presets.backup.json (ou les presets de départ à défaut)."""
    try:
        d = _read(USER_PATH)
    except ValueError as e:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        corrupt = os.path.join(ROOT, f"presets.corrupt-{stamp}.json")
        os.replace(USER_PATH, corrupt)
        restored = _read_or_none(BACKUP_PATH)
        origin = "la copie de secours"
        if restored is None:
            restored = _read_or_none(DEFAULT_PATH) or {}
            origin = "les presets de départ"
        _write_atomic(USER_PATH, restored)
        msg = (f"presets.json illisible ({e}) : mis de côté sous "
               f"{os.path.basename(corrupt)}, restauré depuis {origin}.")
        print(f"[warning] {msg}")
        return restored, msg
    if d is None:   # premier lancement
        d = _read_or_none(DEFAULT_PATH) or {}
        _write_atomic(USER_PATH, d)
    return d, None


def save(d: dict):
    """Écrit presets.json de façon atomique, après une copie de secours
    de la version précédente (si elle est lisible)."""
    if _read_or_none(USER_PATH) is not None:
        shutil.copyfile(USER_PATH, BACKUP_PATH)
    _write_atomic(USER_PATH, d)


def update(fn):
    """Relit le fichier, applique fn(d) puis l'enregistre. Relire juste
    avant d'écrire évite d'écraser ce que l'autre process (panneau /
    boucle principale) a enregistré entre-temps."""
    d, _ = load()
    fn(d)
    save(d)
    return d


# ── Presets globaux (snapshot de tous les onglets) ────────────────────
def global_names(d: dict) -> list[str]:
    return sorted(n for n, v in d.items()
                  if not n.startswith("_") and isinstance(v, dict))


# ── Presets d'onglet : {onglet: {nom: {clé: val}}} sous "_tab_presets" ─
def _is_stale_copy(user: dict, builtin: dict) -> bool:
    """Ancienne copie d'un preset intégré (avant T24 les intégrés étaient
    recopiés dans le fichier) : mêmes valeurs, éventuellement moins de
    clés (bandes ajoutées depuis)."""
    return all(builtin.get(k) == v for k, v in user.items())


def tab_user_presets(d: dict, tab: str, builtin: dict) -> dict:
    """Presets utilisateur de l'onglet, sans les vieilles copies des
    presets intégrés (le preset intégré à jour les remplace)."""
    store = d.get(TAB_KEY)
    user = store.get(tab) if isinstance(store, dict) else None
    if not isinstance(user, dict):
        return {}
    return {n: dict(v) for n, v in user.items() if isinstance(v, dict)
            and not (n in builtin and _is_stale_copy(v, builtin[n]))}


def merged_tab_presets(user: dict, builtin: dict) -> dict:
    """{nom: (valeurs, intégré?)} : les intégrés, puis les presets
    utilisateur (un preset utilisateur du même nom l'emporte)."""
    out = {n: (dict(v), True) for n, v in builtin.items()}
    out.update({n: (dict(v), False) for n, v in user.items()})
    return out


def set_tab_user_presets(d: dict, tab: str, user: dict):
    if not isinstance(d.get(TAB_KEY), dict):
        d[TAB_KEY] = {}
    d[TAB_KEY][tab] = {n: dict(v) for n, v in user.items()}
