"""
Base Validator for Data Validation Layer
Provides abstract base class and core validation framework
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationStatus(Enum):
    """Overall validation status"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"  # Some checks passed, some failed


@dataclass
class ValidationIssue:
    """
    Represents a single validation issue
    """
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    validator: Optional[str] = None
    code: Optional[str] = None
    suggestion: Optional[str] = None
    
    def __repr__(self):
        return f"ValidationIssue({self.severity.value}: {self.message})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'severity': self.severity.value,
            'message': self.message,
            'field': self.field,
            'value': str(self.value) if self.value is not None else None,
            'validator': self.validator,
            'code': self.code,
            'suggestion': self.suggestion
        }


@dataclass
class ValidationResult:
    """
    Standardized validation result
    """
    status: ValidationStatus
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validator_name: Optional[str] = None
    timestamp: Optional[str] = None
    data_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize timestamp if not provided"""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed"""
        return self.status == ValidationStatus.PASSED
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any errors"""
        return any(
            issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
            for issue in self.issues
        )
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings"""
        return any(
            issue.severity == ValidationSeverity.WARNING
            for issue in self.issues
        )
    
    @property
    def error_count(self) -> int:
        """Count of error-level issues"""
        return sum(
            1 for issue in self.issues
            if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
        )
    
    @property
    def warning_count(self) -> int:
        """Count of warning-level issues"""
        return sum(
            1 for issue in self.issues
            if issue.severity == ValidationSeverity.WARNING
        )
    
    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue"""
        self.issues.append(issue)
        
        # Update status based on severity
        if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            if self.status == ValidationStatus.PASSED:
                self.status = ValidationStatus.PARTIAL
    
    def add_error(self, message: str, field: Optional[str] = None, **kwargs):
        """Convenience method to add an error"""
        self.add_issue(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            message=message,
            field=field,
            **kwargs
        ))
    
    def add_warning(self, message: str, field: Optional[str] = None, **kwargs):
        """Convenience method to add a warning"""
        self.add_issue(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            message=message,
            field=field,
            **kwargs
        ))
    
    def add_info(self, message: str, field: Optional[str] = None, **kwargs):
        """Convenience method to add an info"""
        self.add_issue(ValidationIssue(
            severity=ValidationSeverity.INFO,
            message=message,
            field=field,
            **kwargs
        ))
    
    def merge(self, other: 'ValidationResult'):
        """Merge another validation result into this one"""
        self.issues.extend(other.issues)
        self.metadata.update(other.metadata)
        
        # Update status
        if other.has_errors or self.has_errors:
            if other.status == ValidationStatus.FAILED or self.status == ValidationStatus.FAILED:
                self.status = ValidationStatus.FAILED
            else:
                self.status = ValidationStatus.PARTIAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'status': self.status.value,
            'is_valid': self.is_valid,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'issues': [issue.to_dict() for issue in self.issues],
            'metadata': self.metadata,
            'validator_name': self.validator_name,
            'timestamp': self.timestamp,
            'data_id': self.data_id
        }
    
    def __repr__(self):
        return f"ValidationResult(status={self.status.value}, errors={self.error_count}, warnings={self.warning_count})"


@dataclass
class ValidatorConfig:
    """Base configuration for validators"""
    enabled: bool = True
    strict_mode: bool = False  # Fail on warnings in strict mode
    skip_on_error: bool = False  # Skip remaining checks if error found
    verbose: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'enabled': self.enabled,
            'strict_mode': self.strict_mode,
            'skip_on_error': self.skip_on_error,
            'verbose': self.verbose
        }


class BaseValidator(ABC):
    """
    Abstract base class for all validators
    Defines the interface and common functionality
    """
    
    def __init__(self, config: ValidatorConfig):
        """
        Initialize base validator
        
        Args:
            config: ValidatorConfig object with settings
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if config.verbose:
            self.logger.setLevel(logging.DEBUG)
        
        self.logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """
        Validate data
        
        Args:
            data: Input data to validate
            **kwargs: Additional validation parameters
            
        Returns:
            ValidationResult object
        """
        pass
    
    def validate_batch(self, data_batch: List[Any], **kwargs) -> List[ValidationResult]:
        """
        Validate a batch of data
        
        Args:
            data_batch: List of input data
            **kwargs: Additional parameters
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        
        for i, data in enumerate(data_batch):
            try:
                result = self.validate(data, **kwargs)
                result.data_id = result.data_id or f"batch_item_{i}"
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error validating batch item {i}: {e}")
                
                # Create error result
                error_result = ValidationResult(
                    status=ValidationStatus.FAILED,
                    validator_name=self.__class__.__name__,
                    data_id=f"batch_item_{i}"
                )
                error_result.add_error(
                    message=f"Validation failed with exception: {str(e)}",
                    code="VALIDATION_EXCEPTION"
                )
                results.append(error_result)
        
        return results
    
    def _create_result(self, data_id: Optional[str] = None) -> ValidationResult:
        """
        Create a new validation result
        
        Args:
            data_id: Optional data identifier
            
        Returns:
            ValidationResult initialized with validator info
        """
        return ValidationResult(
            status=ValidationStatus.PASSED,
            validator_name=self.__class__.__name__,
            data_id=data_id
        )
    
    def _check_range(self, value: Union[int, float], 
                     min_value: Optional[Union[int, float]] = None,
                     max_value: Optional[Union[int, float]] = None,
                     field_name: str = "value") -> Optional[ValidationIssue]:
        """
        Check if value is within range
        
        Args:
            value: Value to check
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            field_name: Name of the field being validated
            
        Returns:
            ValidationIssue if out of range, None otherwise
        """
        if min_value is not None and value < min_value:
            return ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} ({value}) is below minimum ({min_value})",
                field=field_name,
                value=value,
                code="VALUE_BELOW_MIN",
                suggestion=f"Ensure {field_name} is at least {min_value}"
            )
        
        if max_value is not None and value > max_value:
            return ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} ({value}) exceeds maximum ({max_value})",
                field=field_name,
                value=value,
                code="VALUE_ABOVE_MAX",
                suggestion=f"Ensure {field_name} does not exceed {max_value}"
            )
        
        return None
    
    def _check_type(self, value: Any, expected_type: type,
                    field_name: str = "value") -> Optional[ValidationIssue]:
        """
        Check if value is of expected type
        
        Args:
            value: Value to check
            expected_type: Expected type
            field_name: Name of the field
            
        Returns:
            ValidationIssue if type mismatch, None otherwise
        """
        if not isinstance(value, expected_type):
            return ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} has incorrect type: expected {expected_type.__name__}, got {type(value).__name__}",
                field=field_name,
                value=value,
                code="TYPE_MISMATCH",
                suggestion=f"Convert {field_name} to {expected_type.__name__}"
            )
        
        return None
    
    def _check_not_none(self, value: Any, field_name: str = "value") -> Optional[ValidationIssue]:
        """
        Check if value is not None
        
        Args:
            value: Value to check
            field_name: Name of the field
            
        Returns:
            ValidationIssue if None, None otherwise
        """
        if value is None:
            return ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} is None/null",
                field=field_name,
                value=value,
                code="NULL_VALUE",
                suggestion=f"Provide a valid value for {field_name}"
            )
        
        return None
    
    def _check_not_empty(self, value: Union[str, list, dict],
                        field_name: str = "value") -> Optional[ValidationIssue]:
        """
        Check if value is not empty
        
        Args:
            value: Value to check (string, list, or dict)
            field_name: Name of the field
            
        Returns:
            ValidationIssue if empty, None otherwise
        """
        if not value:
            return ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} is empty",
                field=field_name,
                value=value,
                code="EMPTY_VALUE",
                suggestion=f"Provide a non-empty value for {field_name}"
            )
        
        return None
    
    def _check_enum(self, value: Any, allowed_values: List[Any],
                   field_name: str = "value") -> Optional[ValidationIssue]:
        """
        Check if value is in allowed values
        
        Args:
            value: Value to check
            allowed_values: List of allowed values
            field_name: Name of the field
            
        Returns:
            ValidationIssue if not in allowed values, None otherwise
        """
        if value not in allowed_values:
            return ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} ({value}) not in allowed values: {allowed_values}",
                field=field_name,
                value=value,
                code="INVALID_ENUM_VALUE",
                suggestion=f"Use one of: {', '.join(map(str, allowed_values))}"
            )
        
        return None
    
    def _check_pattern(self, value: str, pattern: str,
                      field_name: str = "value") -> Optional[ValidationIssue]:
        """
        Check if string matches pattern
        
        Args:
            value: String to check
            pattern: Regex pattern
            field_name: Name of the field
            
        Returns:
            ValidationIssue if pattern doesn't match, None otherwise
        """
        import re
        
        if not re.match(pattern, value):
            return ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} does not match required pattern: {pattern}",
                field=field_name,
                value=value,
                code="PATTERN_MISMATCH",
                suggestion=f"Ensure {field_name} matches pattern: {pattern}"
            )
        
        return None
    
    def _apply_check(self, result: ValidationResult, 
                    check_func: Callable, *args, **kwargs):
        """
        Apply a validation check and add to result
        
        Args:
            result: ValidationResult to update
            check_func: Validation check function
            *args, **kwargs: Arguments for check function
        """
        issue = check_func(*args, **kwargs)
        
        if issue:
            result.add_issue(issue)
            
            # In strict mode, warnings become errors
            if self.config.strict_mode and issue.severity == ValidationSeverity.WARNING:
                issue.severity = ValidationSeverity.ERROR
            
            # Skip remaining checks if configured
            if self.config.skip_on_error and issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                return True  # Signal to skip remaining checks
        
        return False  # Continue with remaining checks
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get validator configuration
        
        Returns:
            Configuration dictionary
        """
        return {
            'validator': self.__class__.__name__,
            'config': self.config.to_dict()
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(enabled={self.config.enabled})"


# Example usage
if __name__ == "__main__":
    # Create a simple validator implementation for demonstration
    class SimpleValidator(BaseValidator):
        """Example validator implementation"""
        
        def validate(self, data: Any, **kwargs) -> ValidationResult:
            result = self._create_result(data_id="example_001")
            
            # Example checks
            self._apply_check(result, self._check_not_none, data, "data")
            
            if isinstance(data, (int, float)):
                self._apply_check(result, self._check_range, data, 
                                min_value=0, max_value=100, field_name="value")
            
            # Set final status
            if result.has_errors:
                result.status = ValidationStatus.FAILED
            
            return result
    
    # Test the validator
    config = ValidatorConfig(enabled=True, strict_mode=False)
    validator = SimpleValidator(config)
    
    # Valid data
    result1 = validator.validate(50)
    print(f"\nValidation 1: {result1}")
    print(f"Is valid: {result1.is_valid}")
    
    # Invalid data (out of range)
    result2 = validator.validate(150)
    print(f"\nValidation 2: {result2}")
    print(f"Is valid: {result2.is_valid}")
    print(f"Issues: {[str(issue) for issue in result2.issues]}")
    
    # Batch validation
    batch_results = validator.validate_batch([25, 75, 125, None, 10])
    print(f"\nBatch validation results:")
    for i, result in enumerate(batch_results):
        print(f"  Item {i}: {result.status.value} - {result.error_count} errors")
