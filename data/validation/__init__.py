"""
Data Validation Layer for Multi-Modal LLM Training
Provides validators for text, image, audio, video, and schema validation
"""

# Base validation framework
from .base_validator import (
    BaseValidator,
    ValidatorConfig,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus
)

# Schema validation
from .schema_validator import (
    SchemaValidator,
    Schema,
    FieldRule,
    FieldType,
    SchemaBuilder
)

__version__ = "1.0.0"

__all__ = [
    # Base classes
    "BaseValidator",
    "ValidatorConfig",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "ValidationStatus",
    
    # Schema validation
    "SchemaValidator",
    "Schema",
    "FieldRule",
    "FieldType",
    "SchemaBuilder",
]

# Package metadata
__author__ = "Multi-Modal LLM Training Team"
__description__ = "Comprehensive validation layer for multi-modal machine learning data"
