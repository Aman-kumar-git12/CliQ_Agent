# Forward exports from the modularized expertise sub-package to keep external imports intact.
from .expertise_impl.service import expertise_service

__all__ = ["expertise_service"]
