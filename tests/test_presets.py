"""Stockage des presets (T24) : enregistrer / relire / supprimer, fichier
corrompu, premier lancement, fusion des presets intégrés."""

import json
import os

import pytest

from gfx import presets as store


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    """Redirige tous les fichiers de presets dans un dossier temporaire."""
    monkeypatch.setattr(store, "ROOT", str(tmp_path))
    monkeypatch.setattr(store, "USER_PATH", str(tmp_path / "presets.json"))
    monkeypatch.setattr(store, "DEFAULT_PATH",
                        str(tmp_path / "presets.default.json"))
    monkeypatch.setattr(store, "BACKUP_PATH",
                        str(tmp_path / "presets.backup.json"))
    return tmp_path


def write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_first_launch_copies_defaults(tmp_store):
    write(tmp_store / "presets.default.json", {"Classico": {"grain": 0.1}})
    d, notice = store.load()
    assert d == {"Classico": {"grain": 0.1}}
    assert notice is None
    assert json.loads((tmp_store / "presets.json").read_text()) == d


def test_no_file_at_all(tmp_store):
    assert store.load() == ({}, None)


def test_save_reload_delete_global(tmp_store):
    store.update(lambda d: d.__setitem__("Mon preset", {"grain": 0.3}))
    store.update(lambda d: d.__setitem__("_autosave", {"grain": 0.0}))
    d, _ = store.load()
    assert store.global_names(d) == ["Mon preset"]   # "_…" cachés
    assert d["Mon preset"] == {"grain": 0.3}

    store.update(lambda d: d.pop("Mon preset"))
    d, _ = store.load()
    assert store.global_names(d) == []
    assert "_autosave" in d


def test_autosave_keeps_presets_saved_by_the_panel(tmp_store):
    """Le panneau et la boucle principale écrivent le même fichier : chacun
    relit avant d'écrire, rien n'est écrasé."""
    store.update(lambda d: d.__setitem__("A", {"grain": 0.1}))   # panneau
    store.update(lambda d: d.__setitem__("_autosave", {"x": 1.0}))  # main
    d, _ = store.load()
    assert "A" in d and "_autosave" in d


def test_backup_before_each_write(tmp_store):
    store.save({"A": {"grain": 0.1}})
    assert not (tmp_store / "presets.backup.json").exists()
    store.save({"A": {"grain": 0.1}, "B": {"grain": 0.2}})
    backup = json.loads((tmp_store / "presets.backup.json").read_text())
    assert backup == {"A": {"grain": 0.1}}
    assert not (tmp_store / "presets.json.tmp").exists()


def test_corrupt_file_restored_from_backup(tmp_store):
    store.save({"A": {"grain": 0.1}})
    store.save({"A": {"grain": 0.1}, "B": {"grain": 0.2}})
    (tmp_store / "presets.json").write_text('{"A": {"grain": 0.1}, "B"',
                                            encoding="utf-8")
    d, notice = store.load()
    assert d == {"A": {"grain": 0.1}}   # version d'avant la dernière écriture
    assert notice and "illisible" in notice
    corrupt = [f for f in os.listdir(tmp_store)
               if f.startswith("presets.corrupt-")]
    assert len(corrupt) == 1   # l'original est conservé, pas écrasé
    assert (tmp_store / corrupt[0]).read_text().startswith('{"A"')
    # le fichier restauré est de nouveau lisible
    assert store.load() == ({"A": {"grain": 0.1}}, None)


def test_corrupt_file_without_backup_falls_back_to_defaults(tmp_store):
    write(tmp_store / "presets.default.json", {"Classico": {"grain": 0.1}})
    (tmp_store / "presets.json").write_text("[1, 2]", encoding="utf-8")
    d, notice = store.load()
    assert d == {"Classico": {"grain": 0.1}}
    assert "presets de départ" in notice


def test_corrupt_file_is_not_backed_up(tmp_store):
    """Écrire par-dessus un fichier illisible ne doit pas remplacer une
    bonne copie de secours par le fichier cassé."""
    write(tmp_store / "presets.backup.json", {"A": {"grain": 0.1}})
    (tmp_store / "presets.json").write_text("{oops", encoding="utf-8")
    store.save({"B": {}})
    assert json.loads((tmp_store / "presets.backup.json").read_text()) \
        == {"A": {"grain": 0.1}}


# ── Presets d'onglet + intégrés ───────────────────────────────────────
BUILTIN = {
    "Défaut": {"band0_fx": 0.0, "band8_fx": 0.0},
    "Studio": {"band0_fx": 6.0, "band8_fx": 10.0},
}


def test_builtins_always_listed_and_not_written(tmp_store):
    """Un preset intégré ajouté après coup (Studio, T21) apparaît même si
    l'onglet existe déjà dans le fichier, sans effacer les presets
    utilisateur."""
    write(tmp_store / "presets.json",
          {"_tab_presets": {"Physique": {"Mien": {"band0_fx": 3.0}}}})
    d, _ = store.load()
    user = store.tab_user_presets(d, "Physique", BUILTIN)
    merged = store.merged_tab_presets(user, BUILTIN)
    assert merged["Studio"] == (BUILTIN["Studio"], True)
    assert merged["Mien"] == ({"band0_fx": 3.0}, False)

    store.update(lambda d: store.set_tab_user_presets(d, "Physique", user))
    d, _ = store.load()
    assert list(d["_tab_presets"]["Physique"]) == ["Mien"]


def test_stale_builtin_copies_are_dropped(tmp_store):
    """Avant T24 les intégrés étaient recopiés dans le fichier, figés à
    leurs clés de l'époque : le preset intégré à jour les remplace."""
    d = {"_tab_presets": {"Physique": {
        "Défaut": {"band0_fx": 0.0},              # vieille copie
        "Studio": {"band0_fx": 1.0},              # modifié par l'utilisateur
    }}}
    user = store.tab_user_presets(d, "Physique", BUILTIN)
    assert list(user) == ["Studio"]
    merged = store.merged_tab_presets(user, BUILTIN)
    assert merged["Défaut"] == (BUILTIN["Défaut"], True)
    assert merged["Studio"] == ({"band0_fx": 1.0}, False)   # l'utilisateur gagne


def test_deleting_user_override_brings_builtin_back():
    user = {"Studio": {"band0_fx": 1.0}}
    del user["Studio"]
    assert store.merged_tab_presets(user, BUILTIN)["Studio"][1] is True


def test_real_physics_builtins_include_studio():
    from gfx.controls import PHYSICS_PRESETS
    # fichier d'avant T24 : onglet Physique déjà présent, sans Studio
    old = {n: {k: v for k, v in vals.items()
               if not k.startswith(("band8", "band9", "band10"))}
           for n, vals in PHYSICS_PRESETS.items() if n != "Studio"}
    d = {"_tab_presets": {"Physique": old}}
    user = store.tab_user_presets(d, "Physique", PHYSICS_PRESETS)
    merged = store.merged_tab_presets(user, PHYSICS_PRESETS)
    assert user == {}
    assert set(merged) == set(PHYSICS_PRESETS)
    assert all(is_builtin for _, is_builtin in merged.values())
