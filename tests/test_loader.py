"""Chargement des modèles (T1.15, app/loader.py) : chemin invalide refusé
tout de suite, fichier illisible signalé à la fin du thread, seul le
dernier modèle demandé est installé."""

import time

import pytest

from app.errors import UserError
from app.loader import ModelLoader


class FakeModel:
    model_radius = 100.0
    max_points = 20_000

    def __init__(self):
        self.installed = []

    def set_model(self, prepared, up):
        self.installed.append((prepared[0].name, up))


def run_until_idle(loader, timeout=30.0):
    end = time.monotonic() + timeout
    while loader.busy:
        assert time.monotonic() < end, "chargement trop long"
        loader.update()
        time.sleep(0.01)


@pytest.fixture
def setup():
    model, reports = FakeModel(), []
    loader = ModelLoader(model, report=lambda msg, exc=None: reports.append(msg))
    yield loader, model, reports
    loader.shutdown()


def test_missing_unsupported_or_empty_paths_are_refused(setup, tmp_path):
    loader, _, _ = setup
    with pytest.raises(UserError, match="introuvable"):
        loader.load(str(tmp_path / "absent.glb"), 0)
    (tmp_path / "photo.png").write_bytes(b"png")
    with pytest.raises(UserError, match="Format non géré"):
        loader.load(str(tmp_path / "photo.png"), 0)
    (tmp_path / "vide.obj").write_bytes(b"")
    with pytest.raises(UserError, match="vide"):
        loader.load(str(tmp_path / "vide.obj"), 0)
    assert not loader.busy


def test_primitive_is_installed(setup):
    loader, model, reports = setup
    loader.load("primitive:box", 1)
    run_until_idle(loader)
    assert model.installed == [("primitive:box", 1)] and not reports


def test_corrupt_file_is_reported_and_nothing_installed(setup, tmp_path):
    loader, model, reports = setup
    bad = tmp_path / "abime.glb"
    bad.write_bytes(b"glTF pas un vrai fichier" * 20)
    loader.load(str(bad), 0)
    run_until_idle(loader)
    assert model.installed == []
    assert len(reports) == 1 and "abime.glb" in reports[0]


def test_an_error_is_not_lost_when_another_model_follows(setup, tmp_path):
    loader, model, reports = setup
    bad = tmp_path / "abime.glb"
    bad.write_bytes(b"glTF pas un vrai fichier" * 20)
    loader.load(str(bad), 0)
    loader.load("primitive:box", 0)
    run_until_idle(loader)
    assert len(reports) == 1
    assert model.installed == [("primitive:box", 0)]


def test_only_the_last_requested_model_is_installed(setup):
    loader, model, _ = setup
    loader.load("primitive:box", 0)
    loader.load("primitive:icosphere", 0)
    run_until_idle(loader)
    assert model.installed == [("primitive:icosphere", 0)]
