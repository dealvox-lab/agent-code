"""
Schema Validator for Data Validation Layer
Provides schema definition and validation for structured data
"""

from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
import re

from .base_validator import (
    BaseValidator,
    ValidatorConfig,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus
)


class FieldType(Enum):
    """Supported field types for schema validation"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    ANY = "any"
    OBJECT = "object"
    BYTES = "bytes"
    NULL = "null"


@dataclass
class FieldRule:
    """
    Validation rule for a single field
    """
    name: str
    field_type: Union[FieldType, Type]
    required: bool = True
    nullable: bool = False
    
    # Type-specific constraints
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    
    # List/Dict constraints
    item_type: Optional[Union[FieldType, Type]] = None
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    
    # Nested schema (for dict/object types)
    nested_schema: Optional['Schema'] = None
    
    # Custom validation function
    custom_validator: Optional[Callable[[Any], Union[bool, str]]] = None
    
    # Metadata
    description: Optional[str] = None
    default: Optional[Any] = None
    
    def __repr__(self):
        return f"FieldRule(name={self.name}, type={self.field_type}, required={self.required})"


@dataclass
class Schema:
    """
    Schema definition for structured data validation
    """
    name: str
    fields: List[FieldRule] = field(default_factory=list)
    strict: bool = False  # If True, reject unknown fields
    description: Optional[str] = None
    
    def add_field(self, rule: FieldRule):
        """Add a field rule to schema"""
        self.fields.append(rule)
    
    def get_field(self, name: str) -> Optional[FieldRule]:
        """Get field rule by name"""
        for field_rule in self.fields:
            if field_rule.name == name:
                return field_rule
        return None
    
    def __repr__(self):
        return f"Schema(name={self.name}, fields={len(self.fields)})"


class SchemaValidator(BaseValidator):
    """
    Validates data against defined schemas
    Supports nested objects, custom validators, and complex constraints
    """
    
    def __init__(self, schema: Schema, config: Optional[ValidatorConfig] = None):
        """
        Initialize schema validator
        
        Args:
            schema: Schema definition to validate against
            config: Validator configuration
        """
        if config is None:
            config = ValidatorConfig()
        
        super().__init__(config)
        self.schema = schema
        self.logger.info(f"Schema validator initialized for: {schema.name}")
    
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """
        Validate data against schema
        
        Args:
            data: Input data (typically dict)
            **kwargs: Additional parameters
            
        Returns:
            ValidationResult object
        """
        result = self._create_result(data_id=kwargs.get('data_id'))
        result.metadata['schema_name'] = self.schema.name
        
        # Check if data is dict-like
        if not isinstance(data, dict):
            result.add_error(
                message=f"Data must be a dictionary, got {type(data).__name__}",
                code="INVALID_DATA_TYPE"
            )
            result.status = ValidationStatus.FAILED
            return result
        
        # Validate each field in schema
        for field_rule in self.schema.fields:
            field_name = field_rule.name
            field_value = data.get(field_name)
            
            # Check required fields
            if field_rule.required and field_name not in data:
                result.add_error(
                    message=f"Required field '{field_name}' is missing",
                    field=field_name,
                    code="MISSING_REQUIRED_FIELD",
                    suggestion=f"Add '{field_name}' field to the data"
                )
                continue
            
            # Skip validation if field not present and not required
            if field_name not in data:
                continue
            
            # Validate the field
            self._validate_field(field_rule, field_value, result)
        
        # Check for unknown fields in strict mode
        if self.schema.strict:
            schema_field_names = {rule.name for rule in self.schema.fields}
            unknown_fields = set(data.keys()) - schema_field_names
            
            for unknown_field in unknown_fields:
                result.add_warning(
                    message=f"Unknown field '{unknown_field}' not in schema",
                    field=unknown_field,
                    code="UNKNOWN_FIELD",
                    suggestion="Remove field or add it to the schema"
                )
        
        # Set final status
        if result.has_errors:
            result.status = ValidationStatus.FAILED
        elif result.has_warnings:
            result.status = ValidationStatus.PARTIAL
        
        return result
    
    def _validate_field(self, rule: FieldRule, value: Any, result: ValidationResult):
        """
        Validate a single field against its rule
        
        Args:
            rule: Field validation rule
            value: Field value
            result: ValidationResult to update
        """
        field_name = rule.name
        
        # Check nullable
        if value is None:
            if not rule.nullable:
                result.add_error(
                    message=f"Field '{field_name}' cannot be null",
                    field=field_name,
                    value=value,
                    code="NULL_NOT_ALLOWED"
                )
            return
        
        # Type validation
        if not self._validate_type(rule, value, result):
            return  # Skip further validation if type is wrong
        
        # Type-specific validations
        if rule.field_type in [FieldType.STRING]:
            self._validate_string(rule, value, result)
        
        elif rule.field_type in [FieldType.INTEGER, FieldType.FLOAT]:
            self._validate_number(rule, value, result)
        
        elif rule.field_type == FieldType.LIST:
            self._validate_list(rule, value, result)
        
        elif rule.field_type == FieldType.DICT:
            self._validate_dict(rule, value, result)
        
        # Custom validator
        if rule.custom_validator:
            self._apply_custom_validator(rule, value, result)
    
    def _validate_type(self, rule: FieldRule, value: Any, result: ValidationResult) -> bool:
        """
        Validate field type
        
        Args:
            rule: Field rule
            value: Field value
            result: ValidationResult to update
            
        Returns:
            True if type is valid, False otherwise
        """
        field_name = rule.name
        field_type = rule.field_type
        
        # Skip if ANY type
        if field_type == FieldType.ANY:
            return True
        
        # Type mapping
        type_map = {
            FieldType.STRING: str,
            FieldType.INTEGER: int,
            FieldType.FLOAT: (int, float),
            FieldType.BOOLEAN: bool,
            FieldType.LIST: list,
            FieldType.DICT: dict,
            FieldType.BYTES: bytes,
            FieldType.NULL: type(None)
        }
        
        # Check if custom type
        if isinstance(field_type, type):
            expected_type = field_type
        else:
            expected_type = type_map.get(field_type)
        
        if expected_type and not isinstance(value, expected_type):
            result.add_error(
                message=f"Field '{field_name}' has incorrect type: expected {field_type.value if isinstance(field_type, FieldType) else field_type.__name__}, got {type(value).__name__}",
                field=field_name,
                value=value,
                code="TYPE_MISMATCH"
            )
            return False
        
        return True
    
    def _validate_string(self, rule: FieldRule, value: str, result: ValidationResult):
        """Validate string field"""
        field_name = rule.name
        
        # Length constraints
        if rule.min_length is not None and len(value) < rule.min_length:
            result.add_error(
                message=f"Field '{field_name}' length ({len(value)}) is below minimum ({rule.min_length})",
                field=field_name,
                value=value,
                code="STRING_TOO_SHORT"
            )
        
        if rule.max_length is not None and len(value) > rule.max_length:
            result.add_error(
                message=f"Field '{field_name}' length ({len(value)}) exceeds maximum ({rule.max_length})",
                field=field_name,
                value=value,
                code="STRING_TOO_LONG"
            )
        
        # Pattern validation
        if rule.pattern and not re.match(rule.pattern, value):
            result.add_error(
                message=f"Field '{field_name}' does not match required pattern: {rule.pattern}",
                field=field_name,
                value=value,
                code="PATTERN_MISMATCH"
            )
        
        # Allowed values
        if rule.allowed_values and value not in rule.allowed_values:
            result.add_error(
                message=f"Field '{field_name}' value '{value}' not in allowed values: {rule.allowed_values}",
                field=field_name,
                value=value,
                code="INVALID_ENUM_VALUE"
            )
    
    def _validate_number(self, rule: FieldRule, value: Union[int, float], result: ValidationResult):
        """Validate numeric field"""
        field_name = rule.name
        
        # Range validation
        if rule.min_value is not None and value < rule.min_value:
            result.add_error(
                message=f"Field '{field_name}' value ({value}) is below minimum ({rule.min_value})",
                field=field_name,
                value=value,
                code="VALUE_BELOW_MIN"
            )
        
        if rule.max_value is not None and value > rule.max_value:
            result.add_error(
                message=f"Field '{field_name}' value ({value}) exceeds maximum ({rule.max_value})",
                field=field_name,
                value=value,
                code="VALUE_ABOVE_MAX"
            )
        
        # Allowed values
        if rule.allowed_values and value not in rule.allowed_values:
            result.add_error(
                message=f"Field '{field_name}' value {value} not in allowed values: {rule.allowed_values}",
                field=field_name,
                value=value,
                code="INVALID_ENUM_VALUE"
            )
    
    def _validate_list(self, rule: FieldRule, value: List, result: ValidationResult):
        """Validate list field"""
        field_name = rule.name
        
        # Length constraints
        if rule.min_items is not None and len(value) < rule.min_items:
            result.add_error(
                message=f"Field '{field_name}' has too few items ({len(value)} < {rule.min_items})",
                field=field_name,
                value=value,
                code="TOO_FEW_ITEMS"
            )
        
        if rule.max_items is not None and len(value) > rule.max_items:
            result.add_error(
                message=f"Field '{field_name}' has too many items ({len(value)} > {rule.max_items})",
                field=field_name,
                value=value,
                code="TOO_MANY_ITEMS"
            )
        
        # Item type validation
        if rule.item_type:
            type_map = {
                FieldType.STRING: str,
                FieldType.INTEGER: int,
                FieldType.FLOAT: (int, float),
                FieldType.BOOLEAN: bool,
                FieldType.DICT: dict,
            }
            
            expected_type = type_map.get(rule.item_type, rule.item_type)
            
            for i, item in enumerate(value):
                if not isinstance(item, expected_type):
                    result.add_error(
                        message=f"Field '{field_name}[{i}]' has incorrect type: expected {rule.item_type.value if isinstance(rule.item_type, FieldType) else rule.item_type.__name__}, got {type(item).__name__}",
                        field=f"{field_name}[{i}]",
                        value=item,
                        code="ITEM_TYPE_MISMATCH"
                    )
    
    def _validate_dict(self, rule: FieldRule, value: Dict, result: ValidationResult):
        """Validate dict/object field"""
        field_name = rule.name
        
        # Nested schema validation
        if rule.nested_schema:
            nested_validator = SchemaValidator(rule.nested_schema, self.config)
            nested_result = nested_validator.validate(value)
            
            # Merge results
            for issue in nested_result.issues:
                # Prefix field names with parent field
                if issue.field:
                    issue.field = f"{field_name}.{issue.field}"
                else:
                    issue.field = field_name
                result.add_issue(issue)
    
    def _apply_custom_validator(self, rule: FieldRule, value: Any, result: ValidationResult):
        """Apply custom validation function"""
        field_name = rule.name
        
        try:
            custom_result = rule.custom_validator(value)
            
            # Handle boolean return
            if isinstance(custom_result, bool):
                if not custom_result:
                    result.add_error(
                        message=f"Field '{field_name}' failed custom validation",
                        field=field_name,
                        value=value,
                        code="CUSTOM_VALIDATION_FAILED"
                    )
            # Handle string return (error message)
            elif isinstance(custom_result, str):
                result.add_error(
                    message=custom_result,
                    field=field_name,
                    value=value,
                    code="CUSTOM_VALIDATION_FAILED"
                )
        except Exception as e:
            result.add_error(
                message=f"Custom validator error for '{field_name}': {str(e)}",
                field=field_name,
                value=value,
                code="CUSTOM_VALIDATOR_ERROR"
            )


class SchemaBuilder:
    """
    Fluent builder for creating schemas
    """
    
    def __init__(self, name: str, description: Optional[str] = None):
        """
        Initialize schema builder
        
        Args:
            name: Schema name
            description: Schema description
        """
        self.schema = Schema(name=name, description=description)
    
    def add_string(self, name: str, required: bool = True, 
                   min_length: Optional[int] = None,
                   max_length: Optional[int] = None,
                   pattern: Optional[str] = None,
                   allowed_values: Optional[List[str]] = None,
                   **kwargs) -> 'SchemaBuilder':
        """Add string field"""
        rule = FieldRule(
            name=name,
            field_type=FieldType.STRING,
            required=required,
            min_length=min_length,
            max_length=max_length,
            pattern=pattern,
            allowed_values=allowed_values,
            **kwargs
        )
        self.schema.add_field(rule)
        return self
    
    def add_integer(self, name: str, required: bool = True,
                   min_value: Optional[int] = None,
                   max_value: Optional[int] = None,
                   allowed_values: Optional[List[int]] = None,
                   **kwargs) -> 'SchemaBuilder':
        """Add integer field"""
        rule = FieldRule(
            name=name,
            field_type=FieldType.INTEGER,
            required=required,
            min_value=min_value,
            max_value=max_value,
            allowed_values=allowed_values,
            **kwargs
        )
        self.schema.add_field(rule)
        return self
    
    def add_float(self, name: str, required: bool = True,
                 min_value: Optional[float] = None,
                 max_value: Optional[float] = None,
                 **kwargs) -> 'SchemaBuilder':
        """Add float field"""
        rule = FieldRule(
            name=name,
            field_type=FieldType.FLOAT,
            required=required,
            min_value=min_value,
            max_value=max_value,
            **kwargs
        )
        self.schema.add_field(rule)
        return self
    
    def add_boolean(self, name: str, required: bool = True, **kwargs) -> 'SchemaBuilder':
        """Add boolean field"""
        rule = FieldRule(
            name=name,
            field_type=FieldType.BOOLEAN,
            required=required,
            **kwargs
        )
        self.schema.add_field(rule)
        return self
    
    def add_list(self, name: str, required: bool = True,
                item_type: Optional[FieldType] = None,
                min_items: Optional[int] = None,
                max_items: Optional[int] = None,
                **kwargs) -> 'SchemaBuilder':
        """Add list field"""
        rule = FieldRule(
            name=name,
            field_type=FieldType.LIST,
            required=required,
            item_type=item_type,
            min_items=min_items,
            max_items=max_items,
            **kwargs
        )
        self.schema.add_field(rule)
        return self
    
    def add_dict(self, name: str, required: bool = True,
                nested_schema: Optional[Schema] = None,
                **kwargs) -> 'SchemaBuilder':
        """Add dict/object field"""
        rule = FieldRule(
            name=name,
            field_type=FieldType.DICT,
            required=required,
            nested_schema=nested_schema,
            **kwargs
        )
        self.schema.add_field(rule)
        return self
    
    def add_custom(self, name: str, field_type: Union[FieldType, Type],
                  validator: Optional[Callable] = None,
                  **kwargs) -> 'SchemaBuilder':
        """Add custom field with validator"""
        rule = FieldRule(
            name=name,
            field_type=field_type,
            custom_validator=validator,
            **kwargs
        )
        self.schema.add_field(rule)
        return self
    
    def set_strict(self, strict: bool = True) -> 'SchemaBuilder':
        """Set strict mode (reject unknown fields)"""
        self.schema.strict = strict
        return self
    
    def build(self) -> Schema:
        """Build and return the schema"""
        return self.schema


# Example usage
if __name__ == "__main__":
    # Build a user schema
    user_schema = (
        SchemaBuilder("User", description="User profile schema")
        .add_string("username", min_length=3, max_length=50)
        .add_string("email", pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
        .add_integer("age", min_value=0, max_value=150)
        .add_list("tags", required=False, item_type=FieldType.STRING, max_items=10)
        .add_boolean("is_active", required=False, default=True)
        .set_strict(True)
        .build()
    )
    
    # Create validator
    validator = SchemaValidator(user_schema)
    
    # Valid data
    valid_data = {
        "username": "john_doe",
        "email": "john@example.com",
        "age": 30,
        "tags": ["python", "ml"],
        "is_active": True
    }
    
    result1 = validator.validate(valid_data)
    print(f"\nValidation 1: {result1}")
    print(f"Is valid: {result1.is_valid}")
    
    # Invalid data
    invalid_data = {
        "username": "jd",  # Too short
        "email": "invalid-email",  # Wrong pattern
        "age": 200,  # Too high
        "tags": ["a"] * 20,  # Too many items
        "unknown_field": "value"  # Unknown field (strict mode)
    }
    
    result2 = validator.validate(invalid_data)
    print(f"\nValidation 2: {result2}")
    print(f"Is valid: {result2.is_valid}")
    print(f"Errors: {result2.error_count}, Warnings: {result2.warning_count}")
    print("\nIssues:")
    for issue in result2.issues:
        print(f"  - [{issue.severity.value}] {issue.message}")
