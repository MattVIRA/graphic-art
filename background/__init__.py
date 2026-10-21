"""Fond de scène en formes SDF raymarchées (T1.16).

- `schema` : la déclaration de tous les paramètres (une ligne = un réglage
  du panneau, un uniform, une plage de dé, une réponse aux macros) ;
- `macros`, `palettes`, `params`, `quality` : la logique sans GL, testée
  dans tests/test_background.py ;
- `pass_` : la passe moderngl (programmes par type et par qualité, texture
  du fond, mesure du temps GPU) ;
- `shaders/` : le raymarcher commun et une fonction `map()` par type.

Le package ne connaît ni `audio/` ni `app/` : `app/decor.py` le branche.
"""
