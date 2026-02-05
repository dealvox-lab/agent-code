"""
Data Ingestion Layer for Multi-Modal LLM Training
Provides loaders for S3, local file systems, and streaming data
"""

from .base_loader import (
    BaseLoader,
    DataSample,
    DataModality,
    LoaderConfig
)

from .s3_loader import S3Loader
from .local_loader import LocalLoader
from .multi_modal_loader import MultiModalLoader, DataSource

__version__ = "1.0.0"

__all__ = [
    # Base classes
    "BaseLoader",
    "DataSample",
    "DataModality",
    "LoaderConfig",
    
    # Loaders
    "S3Loader",
    "LocalLoader",
    "MultiModalLoader",
    
    # Utilities
    "DataSource",
]
