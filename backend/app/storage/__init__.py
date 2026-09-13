"""Storage abstraction (Phase 5). See app/storage/base.py for the interface
and the honest scope note on current integration status."""

from app.storage.base import StorageBackend, StorageError, get_backend

__all__ = ["StorageBackend", "StorageError", "get_backend"]
