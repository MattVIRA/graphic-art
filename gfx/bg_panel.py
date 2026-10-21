"""Onglet « Fond » du panneau (T1.16), généré depuis background/schema.py.

Les widgets viennent des aides de `gfx.controls._run_panel` (slider, liste,
case, application de valeurs), passées dans `ui` : ce module ne fait que
la mise en page. Groupes repliables (▸ / ▾), un dé et un cadenas par
groupe ; le cadenas exclut le groupe de tous les tirages.
"""

import logging
import os
import random
import tkinter as tk
from tkinter import ttk

from background import schema

OPEN_GROUPS = ("ld", "vo")   # dépliés à l'ouverture

log = logging.getLogger(__name__)


def _widget(parent, ui, p):
    if p.check:
        ui.check(parent, p.label, p.key, pady=(4, 0))
    elif p.options is not None:
        ui.combo(parent, p.label, p.key, list(p.options), p.tip)
    else:
        ui.slider(parent, p.label, p.key, p.lo, p.hi, p.step, p.tip)\
            .pack(fill="x", padx=8, pady=(4, 0))


def _locked(ui) -> list:
    return [g for g, key in schema.LOCK_KEYS.items() if ui.checks[key].get()]


def _new_shape(ui, seed=None):
    """🎲 Nouveau paysage : tout ce qui n'est pas verrouillé, tiré d'un seed."""
    if seed is None:
        seed = random.randrange(10000)
    vals = schema.random_values(seed, locked=_locked(ui))
    vals.update(bg_seed=float(seed), bg_visible=1.0)
    ui.apply(vals)
    ui.status(f"Fond : nouveau paysage (seed {seed})")


def _group_dice(ui, group, title):
    if group in _locked(ui):
        ui.status(f"« {title} » est verrouillé (🔒)")
        return
    ui.apply(schema.random_values(random.randrange(1 << 30), groups=[group]))
    ui.status(f"Fond : « {title} » tiré au dé")


def _collapsible(parent, ui, group, title):
    """En-tête (▸ titre, 🎲, 🔒) + corps repliable ; rend le corps."""
    head = ttk.Frame(parent)
    head.pack(fill="x", padx=10, pady=(10, 0))
    body = ttk.Frame(parent)
    is_open = {"v": group in OPEN_GROUPS}

    def toggle():
        is_open["v"] = not is_open["v"]
        if is_open["v"]:
            body.pack(fill="x", padx=4, after=head)
        else:
            body.pack_forget()
        btn.configure(text=("▾  " if is_open["v"] else "▸  ") + title)

    btn = ttk.Button(head, text=("▾  " if is_open["v"] else "▸  ") + title,
                     command=toggle)
    btn.pack(side="left", fill="x", expand=True)
    if group in schema.LOCK_KEYS:
        ui.check(head, "🔒", schema.LOCK_KEYS[group], side="right")
        ttk.Button(head, text="🎲", width=3,
                   command=lambda: _group_dice(ui, group, title)).pack(side="right")
    if is_open["v"]:
        body.pack(fill="x", padx=4, after=head)
    return body


# ── groupe Texture (T1.17) : le motif en boutons à image, essentiel / avancé ──
TEX_COLS = 3   # boutons par ligne
THUMB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "textures")


def _keep_pack(frame):
    """Retient la mise en page des enfants de `frame` (pour _show)."""
    for child in frame.winfo_children():
        info = child.pack_info()
        info.pop("in", None)
        child._pack = info


def _show(frame, visible):
    """Affiche ou masque le contenu de `frame` sans changer sa place."""
    for child in frame.winfo_children():
        managed = bool(child.winfo_manager())
        if visible and not managed:
            child.pack(**child._pack)
        elif not visible and managed:
            child.pack_forget()
    # un cadre vidé garde sa dernière hauteur (pack ne la recalcule pas
    # sans enfant) : on la remet à zéro, sinon l'espace reste
    if not visible:
        frame.configure(height=1)


def _thumb(index):
    """Image d'aperçu d'un motif (tools/texture_thumbs.py), None si absente
    ou illisible : le bouton garde alors son seul libellé."""
    path = os.path.join(THUMB_DIR, f"motif_{index}.png")
    if not os.path.exists(path):
        return None
    try:
        return tk.PhotoImage(file=path)
    except tk.TclError:
        log.warning("image d'aperçu illisible : %s", path)
        return None


def _motif_row(parent, ui, key, options):
    """Le menu des motifs en boutons à image ; rend les boutons (choisi = en
    couleur)."""
    row = ttk.Frame(parent)
    row.pack(fill="x", padx=10, pady=(6, 0))
    buttons = []
    for i, name in enumerate(options):
        img = _thumb(i)
        b = ttk.Button(row, text=name, command=lambda i=i: ui.apply({key: float(i)}),
                       image=img, compound="top" if img else "none")
        b._img = img   # tkinter ne garde pas l'image : sans référence, elle disparaît
        b.grid(row=i // TEX_COLS, column=i % TEX_COLS, sticky="ew", padx=2, pady=2)
        buttons.append(b)
    for c in range(min(TEX_COLS, len(options))):
        row.columnconfigure(c, weight=1)
    return buttons


def _current(ui, key) -> float:
    """Valeur affichée d'un réglage du fond (slider, liste ou case)."""
    if key in ui.combos:
        return float(ui.combos[key][0].current())
    if key in ui.checks:
        return 1.0 if ui.checks[key].get() else 0.0
    return float(ui.scales[key].get())


def _texture(body, ui):
    """Le groupe Texture : motifs en boutons à image, réglages essentiels
    visibles, avancés repliés ; les réglages propres à un autre choix
    (`only`) et tous les réglages quand le motif est « Aucun » sont masqués
    (relus toutes les 300 ms). Le 🎲 du groupe tire le motif et ses réglages."""
    ttk.Style().configure("Chosen.TButton", background="#2f5d8c", foreground="#ffffff")
    motif = schema.BY_KEY["bg_tex_motif"]
    hidden = ttk.Frame(body)   # la liste, jamais affichée : elle porte la valeur
    ui.combo(hidden, motif.label, motif.key, list(motif.options), motif.tip)
    buttons = _motif_row(body, ui, motif.key, motif.options)
    hint = ttk.Frame(body)
    hint.pack(fill="x")
    ttk.Label(hint, text="Aucun : la matière du groupe « Matière et lumière ».\n"
                         "Un motif la remplace par du chrome (rugosité, métal,\n"
                         "grain, relief, veines, arêtes).",
              style="Muted.TLabel", justify="left").pack(anchor="w", padx=14, pady=(6, 0))
    _keep_pack(hint)

    settings = ttk.Frame(body)
    settings.pack(fill="x")
    essential = ttk.Frame(settings)
    essential.pack(fill="x")
    toggle = ttk.Button(settings, text="▸  Réglages avancés")
    toggle.pack(fill="x", padx=14, pady=(10, 0))
    advanced = ttk.Frame(settings)
    advanced.pack(fill="x")
    wrappers = {}
    for p in schema.params_of("tex"):
        if p.key == motif.key:
            continue
        parent = essential if p.key in schema.TEX_ESSENTIAL else advanced
        w = ttk.Frame(parent)
        w.pack(fill="x")
        _widget(w, ui, p)
        _keep_pack(w)
        wrappers[p.key] = w
    _keep_pack(advanced)
    _show(advanced, False)
    _keep_pack(settings)

    def flip():
        opened = not advanced.winfo_children()[0].winfo_manager()
        _show(advanced, opened)
        toggle.configure(text=("▾  " if opened else "▸  ") + "Réglages avancés")

    toggle.configure(command=flip)

    def refresh():
        cur = int(_current(ui, motif.key))
        for i, b in enumerate(buttons):
            b.configure(style="Chosen.TButton" if i == cur else "TButton")
        _show(hint, cur == 0)
        _show(settings, cur > 0)
        for key, w in wrappers.items():
            only = schema.BY_KEY[key].only
            _show(w, only is None or int(_current(ui, only[0])) in only[1])
        body.after(300, refresh)

    refresh()


def _top(tab, ui):
    for p in schema.params_of("top"):
        if p.key != "bg_seed":
            _widget(tab, ui, p)
    row = ttk.Frame(tab)
    row.pack(fill="x", padx=10, pady=(8, 0))
    ttk.Button(row, text="🎲  Nouveau paysage", command=lambda: _new_shape(ui)
               ).pack(side="left", fill="x", expand=True)
    replay = ttk.Button(row, text="↻  Rejouer le seed",
                        command=lambda: _new_shape(ui, int(ui.scales["bg_seed"].get())))
    replay.pack(side="left", fill="x", expand=True, padx=(4, 0))
    ui.tooltip(replay, "Retrouve le paysage tiré avec ce seed\n(groupes verrouillés exceptés)")
    _widget(tab, ui, schema.BY_KEY["bg_seed"])


def build(tab, ui):
    """Construit l'onglet Fond dans `tab` (cadre défilant de make_tab)."""
    _top(tab, ui)
    for group, title in schema.GROUPS[1:]:
        body = _collapsible(tab, ui, group, title)
        if group.startswith("macro"):
            ttk.Button(body, text="↺  Toutes au neutre (0,5)",
                       command=lambda g=group: ui.apply(
                           {p.key: 0.5 for p in schema.params_of(g)})
                       ).pack(fill="x", padx=8, pady=(4, 0))
        if group == "tex":
            _texture(body, ui)
            ttk.Frame(body).pack(pady=2)
            continue
        for p in schema.params_of(group):
            if p.key == "bg_grid_size":
                ui.check(body, "Grille du décor", "show_grid", pady=(6, 0))
            _widget(body, ui, p)
        ttk.Frame(body).pack(pady=2)   # respiration en bas du groupe

    # un préréglage de matière règle d'un coup plusieurs sliders
    combo, options = ui.combos["bg_mat_preset"]

    def on_material(_e=None):
        values = schema.MATERIALS.get(options[combo.current()])
        if values:
            ui.apply(values)

    combo.bind("<<ComboboxSelected>>", on_material, add="+")
