"""Garde de la boucle (T1.15, app/errors.py) : erreurs nouvelles / répétées,
arrêt après N frames en erreur d'affilée."""

from app.errors import FrameGuard, signature


def boom(msg="cassé"):
    try:
        raise ValueError(msg)
    except ValueError as e:
        return e


def test_same_error_has_the_same_signature():
    assert signature(boom()) == signature(boom())
    assert signature(boom("a")) != signature(boom("b"))


def test_new_error_reported_once_then_counted():
    g = FrameGuard(limit=30)
    assert g.failed(boom()) == (True, False)
    assert g.failed(boom()) == (False, False)
    assert g.failed(boom("autre")) == (True, False)
    assert sum(g.counts.values()) == 3


def test_stops_after_limit_consecutive_errors():
    g = FrameGuard(limit=3)
    assert not g.failed(boom())[1]
    assert not g.failed(boom())[1]
    assert g.failed(boom())[1]


def test_a_good_frame_resets_the_count():
    g = FrameGuard(limit=3)
    g.failed(boom())
    g.failed(boom())
    g.ok()
    assert not g.failed(boom())[1]
    assert not g.failed(boom())[1]
