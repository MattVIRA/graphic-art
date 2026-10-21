"""Texture du relief (T1.17) : réglages bg_tex_*, sans GL — table du schéma,
texture fixe (ni musique ni macros), dé, uniforms."""

import dataclasses
import random

import pytest

from background import schema
from background.params import BackgroundState

QUIET = {"rms": 0.0, "bpm": 0.0, "beat_phase": 0.0}
LOUD = {"rms": 0.9, "bpm": 128.0, "beat_phase": 0.4}
TEX = schema.params_of("tex")


def frame(p, features=QUIET, level=0.0, steps=1):
    state = BackgroundState()
    for _ in range(steps):
        f = state.step(p, features, [level] * 11, 1 / 60)
    return f


def settings(**kw) -> dict:
    p = schema.defaults()
    p.update({"bg_visible": 1.0, **kw})
    return p


def test_texture_is_off_by_default_and_in_its_group():
    assert schema.BY_KEY["bg_tex_motif"].default == 0
    assert schema.TEX_MOTIFS[0] == "Aucun"
    assert all(p.key.startswith("bg_tex_") for p in TEX)
    assert "tex" in schema.LOCK_KEYS


def test_texture_has_no_music_and_no_macros():
    assert all(p.audio is None and not p.macros for p in TEX)


@pytest.mark.parametrize("change", [dict(audio=("kick", 0.3)),
                                    dict(audio=("energy", 0.3)),
                                    dict(macros={"glow": 0.5})])
def test_validate_refuses_music_and_macros_on_the_texture(change):
    bad = dataclasses.replace(schema.BY_KEY["bg_tex_shine"], **change)
    assert schema._check_texture(bad)


def test_only_targets_a_texture_menu():
    assert schema._check_texture(dataclasses.replace(
        schema.BY_KEY["bg_tex_angle"], only=("bg_tex_scale", (1,))))
    assert schema._check_texture(dataclasses.replace(
        schema.BY_KEY["bg_tex_angle"], only=("bg_tex_motif", (99,))))
    owners = {p.only[0] for p in TEX if p.only}
    assert owners <= {"bg_tex_motif", "bg_tex_orient"}


def test_texture_does_not_react_to_music():
    """Texture fixe : à réglages égaux, mêmes uniforms bg_tex_* au calme et
    sur une musique forte."""
    p = settings(bg_tex_motif=1.0, bg_mov_music=1.0)
    calm, loud = frame(p, QUIET, 0.0, 30), frame(p, LOUD, 1.0, 30)
    for q in TEX:
        if q.shader:
            assert calm.uniforms[q.uniform] == loud.uniforms[q.uniform], q.key


def test_texture_settings_leave_the_other_uniforms_alone():
    """« Aucun » rend la même image qu'avant : les réglages de la texture ne
    touchent aucun autre uniform."""
    rng = random.Random(3)
    base = frame(settings()).uniforms
    p = settings(**{q.key: rng.uniform(q.lo, q.hi) for q in TEX})
    other = frame(p).uniforms
    tex = {q.uniform for q in TEX if q.shader}
    for k, v in base.items():
        if k not in tex:
            assert other[k] == v, k


def test_dice_is_deterministic_and_draws_the_motif():
    a = schema.random_values(7, groups=["tex"])
    assert a == schema.random_values(7, groups=["tex"])
    assert "bg_tex_motif" in a
    assert set(a) == {p.key for p in TEX if p.rand}


def test_motif_weights_keep_none_rare():
    draws = [schema.random_values(s, groups=["tex"])["bg_tex_motif"] for s in range(2000)]
    share = draws.count(0.0) / len(draws)
    assert 0.15 < share < 0.25


def test_draws_stay_in_their_dice_range():
    for s in range(50):
        for k, v in schema.random_values(s, groups=["tex"]).items():
            p = schema.BY_KEY[k]
            assert p.lo <= v <= p.hi
            if v != 0.0 or p.zero == 0.0:
                assert p.rand[0] - 1e-6 <= v <= p.rand[1] + 1e-6, k


def test_lock_keeps_the_texture():
    assert not any(k.startswith("bg_tex_")
                   for k in schema.random_values(5, locked=["tex"]))


def test_old_presets_give_no_texture():
    """Un preset d'avant T1.17 n'a aucune clé bg_tex_* : le panneau remet les
    réglages absents à leur défaut, donc Motif « Aucun »."""
    for name, vals in schema.AMBIANCES.items():
        if name not in schema.TEX_AMBIANCES:
            assert not any(k.startswith("bg_tex_") for k in vals)
    assert frame(settings()).uniforms["u_tex_motif"] == 0.0


def test_five_texture_recipes_on_existing_landscapes():
    assert len(schema.TEX_AMBIANCES) == 5
    for name, (base, _tex) in schema.TEX_AMBIANCES.items():
        vals = schema.AMBIANCES[name]
        assert vals["bg_tex_motif"] > 0, name
        assert {k: v for k, v in vals.items() if not k.startswith("bg_tex_")} \
            == schema.AMBIANCES[base]
