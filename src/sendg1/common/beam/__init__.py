"""Beam-walking library: terrain generation and MDP terms.

Lives under ``common`` rather than ``tasks/beam`` on purpose.
``common/obs/terms.py`` declares the privileged beam observations, and importing
anything under ``sendg1.tasks`` runs task registration -- which itself imports
``terms.py``. Keeping these modules free of ``sendg1.tasks`` imports is what
keeps that from becoming an import cycle.
"""
