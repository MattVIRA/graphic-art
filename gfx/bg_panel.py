"""Onglet « Fond » du panneau (T16), généré depuis background/schema.py.

Les widgets viennent des aides de `gfx.controls._run_panel` (slider, liste,
case, application de valeurs), passées dans `ui` : ce module ne fait que
la mise en page. Groupes repliables (▸ / ▾), un dé et un cadenas par
groupe ; le cadenas exclut le groupe de tous les tirages.
"""

import random
from tkinter import ttk

from background import schema

OPEN_GROUPS = ("macro_form", "sp", "sh")   # dépliés à l'ouverture


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
    """🎲 Nouvelle forme : tout ce qui n'est pas verrouillé, tiré d'un seed."""
    if seed is None:
        seed = random.randrange(10000)
    vals = schema.random_values(seed, locked=_locked(ui))
    vals.update(bg_seed=float(seed), bg_visible=1.0)
    ui.apply(vals)
    ui.status(f"Fond : nouvelle forme (seed {seed})")


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


def _top(tab, ui):
    for p in schema.params_of("top"):
        if p.key != "bg_seed":
            _widget(tab, ui, p)
    row = ttk.Frame(tab)
    row.pack(fill="x", padx=10, pady=(8, 0))
    ttk.Button(row, text="🎲  Nouvelle forme", command=lambda: _new_shape(ui)
               ).pack(side="left", fill="x", expand=True)
    replay = ttk.Button(row, text="↻  Rejouer le seed",
                        command=lambda: _new_shape(ui, int(ui.scales["bg_seed"].get())))
    replay.pack(side="left", fill="x", expand=True, padx=(4, 0))
    ui.tooltip(replay, "Retrouve la forme tirée avec ce seed\n(groupes verrouillés exceptés)")
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
        for p in schema.params_of(group):
            _widget(body, ui, p)
        if group == "solid":
            ui.check(body, "Grille du décor", "show_grid", pady=(6, 0))
        ttk.Frame(body).pack(pady=2)   # respiration en bas du groupe

    # un préréglage de matière règle d'un coup plusieurs sliders
    combo, options = ui.combos["bg_mat_preset"]

    def on_material(_e=None):
        values = schema.MATERIALS.get(options[combo.current()])
        if values:
            ui.apply(values)

    combo.bind("<<ComboboxSelected>>", on_material, add="+")
