"""
Text Validators for Data Validation Layer
Comprehensive text validation including length, content, language, encoding, and quality
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import re
import json
import unicodedata

from .base_validator import (
    BaseValidator,
    ValidatorConfig,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus
)


@dataclass
class TextValidatorConfig(ValidatorConfig):
    """Configuration for text validators"""
    # Length constraints
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_words: Optional[int] = None
    max_words: Optional[int] = None
    min_sentences: Optional[int] = None
    max_sentences: Optional[int] = None
    
    # Content validation
    check_profanity: bool = False
    check_harmful_content: bool = False
    allowed_languages: Optional[List[str]] = None
    
    # Encoding
    required_encoding: str = 'utf-8'
    
    # Quality
    min_readability_score: Optional[float] = None
    max_special_char_ratio: Optional[float] = None
    
    # Format
    validate_json: bool = False
    validate_xml: bool = False


class TextLengthValidator(BaseValidator):
    """
    Validates text length constraints
    Supports character, word, and sentence-level validation
    """
    
    def __init__(self, config: TextValidatorConfig):
        super().__init__(config)
        self.config: TextValidatorConfig = config
    
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """
        Validate text length
        
        Args:
            data: Input text string
            **kwargs: Additional parameters
            
        Returns:
            ValidationResult object
        """
        result = self._create_result(data_id=kwargs.get('data_id'))
        
        # Type check
        if not isinstance(data, str):
            result.add_error(
                message=f"Expected string, got {type(data).__name__}",
                code="INVALID_TYPE"
            )
            result.status = ValidationStatus.FAILED
            return result
        
        text = data
        
        # Character length validation
        char_length = len(text)
        
        if self.config.min_length is not None:
            if char_length < self.config.min_length:
                result.add_error(
                    message=f"Text too short: {char_length} characters (minimum: {self.config.min_length})",
                    field="length",
                    value=char_length,
                    code="TEXT_TOO_SHORT",
                    suggestion=f"Text should be at least {self.config.min_length} characters"
                )
        
        if self.config.max_length is not None:
            if char_length > self.config.max_length:
                result.add_error(
                    message=f"Text too long: {char_length} characters (maximum: {self.config.max_length})",
                    field="length",
                    value=char_length,
                    code="TEXT_TOO_LONG",
                    suggestion=f"Text should not exceed {self.config.max_length} characters"
                )
        
        # Word count validation
        words = text.split()
        word_count = len(words)
        
        if self.config.min_words is not None:
            if word_count < self.config.min_words:
                result.add_warning(
                    message=f"Too few words: {word_count} words (minimum: {self.config.min_words})",
                    field="word_count",
                    value=word_count,
                    code="TOO_FEW_WORDS"
                )
        
        if self.config.max_words is not None:
            if word_count > self.config.max_words:
                result.add_warning(
                    message=f"Too many words: {word_count} words (maximum: {self.config.max_words})",
                    field="word_count",
                    value=word_count,
                    code="TOO_MANY_WORDS"
                )
        
        # Sentence count validation
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        if self.config.min_sentences is not None:
            if sentence_count < self.config.min_sentences:
                result.add_info(
                    message=f"Few sentences: {sentence_count} sentences (recommended minimum: {self.config.min_sentences})",
                    field="sentence_count",
                    value=sentence_count,
                    code="FEW_SENTENCES"
                )
        
        if self.config.max_sentences is not None:
            if sentence_count > self.config.max_sentences:
                result.add_info(
                    message=f"Many sentences: {sentence_count} sentences (recommended maximum: {self.config.max_sentences})",
                    field="sentence_count",
                    value=sentence_count,
                    code="MANY_SENTENCES"
                )
        
        # Add metadata
        result.metadata.update({
            'char_length': char_length,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': char_length / word_count if word_count > 0 else 0,
            'avg_sentence_length': word_count / sentence_count if sentence_count > 0 else 0
        })
        
        # Set final status
        if result.has_errors:
            result.status = ValidationStatus.FAILED
        
        return result


class TextContentValidator(BaseValidator):
    """
    Validates text content for profanity, harmful content, and quality
    """
    
    # Basic profanity word list (expandable)
    PROFANITY_WORDS = {
        'badword1', 'badword2', 'offensive'
        # Add actual profanity filter or use external library
    }
    
    # Harmful content patterns
    HARMFUL_PATTERNS = [
        r'\b(kill|murder|bomb)\b',  # Violence (basic example)
        # Add more patterns as needed
    ]
    
    def __init__(self, config: TextValidatorConfig):
        super().__init__(config)
        self.config: TextValidatorConfig = config
    
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """
        Validate text content
        
        Args:
            data: Input text string
            **kwargs: Additional parameters
            
        Returns:
            ValidationResult object
        """
        result = self._create_result(data_id=kwargs.get('data_id'))
        
        if not isinstance(data, str):
            result.add_error(
                message=f"Expected string, got {type(data).__name__}",
                code="INVALID_TYPE"
            )
            result.status = ValidationStatus.FAILED
            return result
        
        text = data.lower()
        
        # Profanity check
        if self.config.check_profanity:
            found_profanity = self._check_profanity(text)
            if found_profanity:
                result.add_warning(
                    message=f"Profanity detected: {', '.join(found_profanity)}",
                    field="content",
                    code="PROFANITY_DETECTED",
                    suggestion="Remove or replace inappropriate language"
                )
        
        # Harmful content check
        if self.config.check_harmful_content:
            harmful_matches = self._check_harmful_content(text)
            if harmful_matches:
                result.add_error(
                    message=f"Potentially harmful content detected: {len(harmful_matches)} matches",
                    field="content",
                    code="HARMFUL_CONTENT",
                    suggestion="Review and remove harmful content"
                )
        
        # Special character ratio
        if self.config.max_special_char_ratio is not None:
            special_ratio = self._calculate_special_char_ratio(data)
            if special_ratio > self.config.max_special_char_ratio:
                result.add_warning(
                    message=f"High special character ratio: {special_ratio:.2%} (max: {self.config.max_special_char_ratio:.2%})",
                    field="special_chars",
                    value=special_ratio,
                    code="HIGH_SPECIAL_CHAR_RATIO"
                )
            
            result.metadata['special_char_ratio'] = special_ratio
        
        # Set final status
        if result.has_errors:
            result.status = ValidationStatus.FAILED
        
        return result
    
    def _check_profanity(self, text: str) -> List[str]:
        """Check for profanity words"""
        words = set(text.split())
        return list(words & self.PROFANITY_WORDS)
    
    def _check_harmful_content(self, text: str) -> List[str]:
        """Check for harmful content patterns"""
        matches = []
        for pattern in self.HARMFUL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)
        return matches
    
    def _calculate_special_char_ratio(self, text: str) -> float:
        """Calculate ratio of special characters to total characters"""
        if not text:
            return 0.0
        
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return special_chars / len(text)


class TextEncodingValidator(BaseValidator):
    """
    Validates text encoding and character validity
    """
    
    def __init__(self, config: TextValidatorConfig):
        super().__init__(config)
        self.config: TextValidatorConfig = config
    
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """
        Validate text encoding
        
        Args:
            data: Input text string or bytes
            **kwargs: Additional parameters
            
        Returns:
            ValidationResult object
        """
        result = self._create_result(data_id=kwargs.get('data_id'))
        
        # Handle bytes input
        if isinstance(data, bytes):
            try:
                text = data.decode(self.config.required_encoding)
                result.add_info(
                    message=f"Successfully decoded bytes using {self.config.required_encoding}",
                    code="ENCODING_SUCCESS"
                )
            except UnicodeDecodeError as e:
                result.add_error(
                    message=f"Failed to decode bytes using {self.config.required_encoding}: {str(e)}",
                    code="ENCODING_ERROR",
                    suggestion=f"Ensure data is encoded in {self.config.required_encoding}"
                )
                result.status = ValidationStatus.FAILED
                return result
        elif isinstance(data, str):
            text = data
            
            # Try to encode/decode to verify
            try:
                text.encode(self.config.required_encoding)
            except UnicodeEncodeError as e:
                result.add_error(
                    message=f"Text contains characters not encodable in {self.config.required_encoding}: {str(e)}",
                    code="ENCODING_ERROR"
                )
        else:
            result.add_error(
                message=f"Expected string or bytes, got {type(data).__name__}",
                code="INVALID_TYPE"
            )
            result.status = ValidationStatus.FAILED
            return result
        
        # Check for control characters
        control_chars = [c for c in text if unicodedata.category(c) == 'Cc' and c not in '\n\r\t']
        if control_chars:
            result.add_warning(
                message=f"Found {len(control_chars)} control characters",
                code="CONTROL_CHARACTERS",
                suggestion="Remove control characters"
            )
        
        # Check for invalid unicode
        invalid_chars = [c for c in text if unicodedata.category(c) == 'Cn']  # Undefined
        if invalid_chars:
            result.add_warning(
                message=f"Found {len(invalid_chars)} invalid unicode characters",
                code="INVALID_UNICODE"
            )
        
        result.metadata.update({
            'encoding': self.config.required_encoding,
            'control_char_count': len(control_chars),
            'invalid_char_count': len(invalid_chars)
        })
        
        if result.has_errors:
            result.status = ValidationStatus.FAILED
        
        return result


class TextFormatValidator(BaseValidator):
    """
    Validates structured text formats (JSON, XML, CSV, etc.)
    """
    
    def __init__(self, config: TextValidatorConfig):
        super().__init__(config)
        self.config: TextValidatorConfig = config
    
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """
        Validate text format
        
        Args:
            data: Input text string
            **kwargs: Additional parameters (format_type='json'|'xml'|'csv')
            
        Returns:
            ValidationResult object
        """
        result = self._create_result(data_id=kwargs.get('data_id'))
        
        if not isinstance(data, str):
            result.add_error(
                message=f"Expected string, got {type(data).__name__}",
                code="INVALID_TYPE"
            )
            result.status = ValidationStatus.FAILED
            return result
        
        format_type = kwargs.get('format_type', 'json')
        
        # JSON validation
        if format_type == 'json' or self.config.validate_json:
            self._validate_json(data, result)
        
        # XML validation
        if format_type == 'xml' or self.config.validate_xml:
            self._validate_xml(data, result)
        
        # CSV validation
        if format_type == 'csv':
            self._validate_csv(data, result)
        
        if result.has_errors:
            result.status = ValidationStatus.FAILED
        
        return result
    
    def _validate_json(self, text: str, result: ValidationResult):
        """Validate JSON format"""
        try:
            parsed = json.loads(text)
            result.add_info(
                message="Valid JSON format",
                code="VALID_JSON"
            )
            result.metadata['json_type'] = type(parsed).__name__
            if isinstance(parsed, dict):
                result.metadata['json_keys'] = len(parsed.keys())
            elif isinstance(parsed, list):
                result.metadata['json_items'] = len(parsed)
        except json.JSONDecodeError as e:
            result.add_error(
                message=f"Invalid JSON: {str(e)}",
                code="INVALID_JSON",
                suggestion="Check JSON syntax for errors"
            )
    
    def _validate_xml(self, text: str, result: ValidationResult):
        """Validate XML format"""
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(text)
            result.add_info(
                message="Valid XML format",
                code="VALID_XML"
            )
            result.metadata['xml_root_tag'] = root.tag
        except ET.ParseError as e:
            result.add_error(
                message=f"Invalid XML: {str(e)}",
                code="INVALID_XML",
                suggestion="Check XML syntax for errors"
            )
        except ImportError:
            result.add_warning(
                message="XML validation skipped (xml.etree not available)",
                code="XML_VALIDATION_SKIPPED"
            )
    
    def _validate_csv(self, text: str, result: ValidationResult):
        """Validate CSV format"""
        import csv
        import io
        
        try:
            reader = csv.reader(io.StringIO(text))
            rows = list(reader)
            
            if not rows:
                result.add_warning(
                    message="CSV is empty",
                    code="EMPTY_CSV"
                )
                return
            
            # Check column consistency
            header_cols = len(rows[0])
            inconsistent_rows = []
            
            for i, row in enumerate(rows[1:], start=1):
                if len(row) != header_cols:
                    inconsistent_rows.append(i)
            
            if inconsistent_rows:
                result.add_error(
                    message=f"Inconsistent column count in rows: {inconsistent_rows[:5]}",
                    code="INCONSISTENT_CSV_COLUMNS",
                    suggestion="Ensure all rows have the same number of columns"
                )
            else:
                result.add_info(
                    message="Valid CSV format",
                    code="VALID_CSV"
                )
            
            result.metadata.update({
                'csv_rows': len(rows),
                'csv_columns': header_cols,
                'inconsistent_rows': len(inconsistent_rows)
            })
            
        except Exception as e:
            result.add_error(
                message=f"CSV parsing error: {str(e)}",
                code="CSV_PARSE_ERROR"
            )


class TextLanguageValidator(BaseValidator):
    """
    Validates text language and detects language
    Note: Basic implementation - for production use langdetect or similar
    """
    
    def __init__(self, config: TextValidatorConfig):
        super().__init__(config)
        self.config: TextValidatorConfig = config
    
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """
        Validate text language
        
        Args:
            data: Input text string
            **kwargs: Additional parameters
            
        Returns:
            ValidationResult object
        """
        result = self._create_result(data_id=kwargs.get('data_id'))
        
        if not isinstance(data, str):
            result.add_error(
                message=f"Expected string, got {type(data).__name__}",
                code="INVALID_TYPE"
            )
            result.status = ValidationStatus.FAILED
            return result
        
        # Basic language detection (character-based heuristic)
        detected_lang = self._detect_language_simple(data)
        result.metadata['detected_language'] = detected_lang
        
        # Check against allowed languages
        if self.config.allowed_languages:
            if detected_lang not in self.config.allowed_languages:
                result.add_warning(
                    message=f"Detected language '{detected_lang}' not in allowed languages: {self.config.allowed_languages}",
                    field="language",
                    value=detected_lang,
                    code="LANGUAGE_NOT_ALLOWED",
                    suggestion=f"Text should be in one of: {', '.join(self.config.allowed_languages)}"
                )
        
        return result
    
    def _detect_language_simple(self, text: str) -> str:
        """
        Simple language detection based on character ranges
        For production, use langdetect or similar library
        """
        # Count character types
        latin_chars = sum(1 for c in text if ord(c) < 0x250)
        cjk_chars = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)  # Chinese
        arabic_chars = sum(1 for c in text if 0x0600 <= ord(c) <= 0x06FF)
        cyrillic_chars = sum(1 for c in text if 0x0400 <= ord(c) <= 0x04FF)
        
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return "unknown"
        
        # Simple majority detection
        if cjk_chars / total_chars > 0.3:
            return "zh"
        elif arabic_chars / total_chars > 0.3:
            return "ar"
        elif cyrillic_chars / total_chars > 0.3:
            return "ru"
        elif latin_chars / total_chars > 0.5:
            return "en"  # Default to English for Latin
        else:
            return "unknown"


class TextQualityValidator(BaseValidator):
    """
    Validates text quality metrics (readability, complexity, etc.)
    """
    
    def __init__(self, config: TextValidatorConfig):
        super().__init__(config)
        self.config: TextValidatorConfig = config
    
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """
        Validate text quality
        
        Args:
            data: Input text string
            **kwargs: Additional parameters
            
        Returns:
            ValidationResult object
        """
        result = self._create_result(data_id=kwargs.get('data_id'))
        
        if not isinstance(data, str):
            result.add_error(
                message=f"Expected string, got {type(data).__name__}",
                code="INVALID_TYPE"
            )
            result.status = ValidationStatus.FAILED
            return result
        
        # Calculate readability score (Flesch Reading Ease approximation)
        readability = self._calculate_readability(data)
        result.metadata['readability_score'] = readability
        
        if self.config.min_readability_score is not None:
            if readability < self.config.min_readability_score:
                result.add_warning(
                    message=f"Low readability score: {readability:.2f} (minimum: {self.config.min_readability_score})",
                    field="readability",
                    value=readability,
                    code="LOW_READABILITY",
                    suggestion="Simplify sentence structure and word choice"
                )
        
        # Calculate other quality metrics
        metrics = self._calculate_quality_metrics(data)
        result.metadata.update(metrics)
        
        # Check for excessive repetition
        if metrics.get('unique_word_ratio', 1.0) < 0.3:
            result.add_warning(
                message=f"Low vocabulary diversity: {metrics['unique_word_ratio']:.2%}",
                code="LOW_VOCABULARY_DIVERSITY"
            )
        
        return result
    
    def _calculate_readability(self, text: str) -> float:
        """
        Calculate Flesch Reading Ease score (approximation)
        Score: 0-100 (higher = easier to read)
        """
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        words = text.split()
        word_count = len(words)
        
        if sentence_count == 0 or word_count == 0:
            return 0.0
        
        # Count syllables (rough approximation)
        syllable_count = sum(self._count_syllables(word) for word in words)
        
        # Flesch Reading Ease formula
        score = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
        
        return max(0.0, min(100.0, score))
    
    def _count_syllables(self, word: str) -> int:
        """Rough syllable count"""
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent 'e'
        if word.endswith('e'):
            count -= 1
        
        return max(1, count)
    
    def _calculate_quality_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate various quality metrics"""
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return {}
        
        unique_words = len(set(word.lower() for word in words))
        
        return {
            'unique_word_ratio': unique_words / word_count,
            'avg_word_length': sum(len(w) for w in words) / word_count,
            'unique_word_count': unique_words,
            'total_word_count': word_count
        }


# Example usage
if __name__ == "__main__":
    # Test text length validator
    config = TextValidatorConfig(
        min_length=10,
        max_length=1000,
        min_words=5,
        max_words=200
    )
    
    length_validator = TextLengthValidator(config)
    
    text1 = "Short"
    result1 = length_validator.validate(text1)
    print(f"\nLength validation (short text): {result1}")
    print(f"Issues: {[str(i) for i in result1.issues]}")
    
    # Test content validator
    content_config = TextValidatorConfig(
        check_profanity=True,
        max_special_char_ratio=0.1
    )
    
    content_validator = TextContentValidator(content_config)
    
    text2 = "This is a clean text sample!"
    result2 = content_validator.validate(text2)
    print(f"\nContent validation: {result2}")
    print(f"Metadata: {result2.metadata}")
    
    # Test format validator
    format_config = TextValidatorConfig(validate_json=True)
    format_validator = TextFormatValidator(format_config)
    
    json_text = '{"name": "test", "value": 123}'
    result3 = format_validator.validate(json_text)
    print(f"\nFormat validation (JSON): {result3}")
    print(f"Metadata: {result3.metadata}")
    
    # Test quality validator
    quality_config = TextValidatorConfig(min_readability_score=60.0)
    quality_validator = TextQualityValidator(quality_config)
    
    text4 = "The quick brown fox jumps over the lazy dog. This is a simple sentence for testing."
    result4 = quality_validator.validate(text4)
    print(f"\nQuality validation: {result4}")
    print(f"Readability: {result4.metadata.get('readability_score', 'N/A')}")
