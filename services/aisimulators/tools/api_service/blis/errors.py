"""Errors the BLIS adapter raises.

Kept apart from the modules that raise them so the catalog reader and the
runtime resolver can share the types without importing each other.
"""

from __future__ import annotations


class BlisCatalogError(RuntimeError):
    """BLIS could not report, or reported an unreadable, catalog."""


class BlisUnsupportedRequest(ValueError):
    """BLIS cannot represent a model, hardware, fabric, or engine choice."""
