"""
Text Processor for NLP Preprocessing
Handles tokenization, cleaning, normalization, and encoding
"""

import re
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import unicodedata

from .base_processor import BaseProcessor, ProcessorConfig


@dataclass
class TextProcessorConfig(ProcessorConfig):
    """Configuration for text preprocessing"""
    # Cleaning options
    lowercase: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_phone_numbers: bool = True
    remove_numbers: bool = False
    remove_punctuation: bool = False
    remove_extra_whitespace: bool = True
    
    # Normalization options
    unicode_normalization: str = 'NFKC'  # NFKC, NFC, NFD, NFKD
    remove_accents: bool = False
    
    # Tokenization options
    tokenize: bool = True
    tokenizer_type: str = 'whitespace'  # whitespace, word, sentence
    max_length: Optional[int] = None
    min_length: Optional[int] = 1
    
    # Special tokens
    add_special_tokens: bool = False
    bos_token: str = '<BOS>'
    eos_token: str = '<EOS>'
    pad_token: str = '<PAD>'
    unk_token: str = '<UNK>'
    
    # Encoding options
    encoding: str = 'utf-8'


class TextProcessor(BaseProcessor):
    """
    Comprehensive text preprocessing for NLP tasks
    Supports multiple languages and customizable pipelines
    """
    
    # Regular expressions for common patterns
    URL_PATTERN = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_PATTERN = re.compile(r'(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}')
    NUMBER_PATTERN = re.compile(r'\b\d+\b')
    WHITESPACE_PATTERN = re.compile(r'\s+')
    PUNCTUATION_PATTERN = re.compile(r'[^\w\s]')
    
    def __init__(self, config: TextProcessorConfig):
        """
        Initialize text processor
        
        Args:
            config: TextProcessorConfig object
        """
        super().__init__(config)
        self.config: TextProcessorConfig = config
        
        # Build vocabulary if needed
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
    
    def _get_modality(self) -> str:
        return "text"
    
    def validate(self, data: Any) -> bool:
        """
        Validate text input
        
        Args:
            data: Input data (should be string)
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(data, str):
            self.logger.warning(f"Invalid data type: {type(data)}, expected str")
            return False
        
        if len(data.strip()) == 0:
            self.logger.warning("Empty text after stripping whitespace")
            return False
        
        return True
    
    def process(self, data: str, **kwargs) -> Union[str, List[str], List[int]]:
        """
        Process a single text sample
        
        Args:
            data: Input text string
            **kwargs: Additional parameters
            
        Returns:
            Processed text (string, tokens, or token IDs)
        """
        text = data
        
        # Step 1: Unicode normalization
        if self.config.unicode_normalization:
            text = unicodedata.normalize(self.config.unicode_normalization, text)
        
        # Step 2: Remove accents
        if self.config.remove_accents:
            text = self._remove_accents(text)
        
        # Step 3: Lowercase
        if self.config.lowercase:
            text = text.lower()
        
        # Step 4: Remove URLs
        if self.config.remove_urls:
            text = self.URL_PATTERN.sub(' ', text)
        
        # Step 5: Remove emails
        if self.config.remove_emails:
            text = self.EMAIL_PATTERN.sub(' ', text)
        
        # Step 6: Remove phone numbers
        if self.config.remove_phone_numbers:
            text = self.PHONE_PATTERN.sub(' ', text)
        
        # Step 7: Remove numbers
        if self.config.remove_numbers:
            text = self.NUMBER_PATTERN.sub(' ', text)
        
        # Step 8: Remove punctuation
        if self.config.remove_punctuation:
            text = self.PUNCTUATION_PATTERN.sub(' ', text)
        
        # Step 9: Remove extra whitespace
        if self.config.remove_extra_whitespace:
            text = self.WHITESPACE_PATTERN.sub(' ', text).strip()
        
        # Step 10: Length filtering
        if self.config.min_length and len(text) < self.config.min_length:
            self.logger.debug(f"Text too short: {len(text)} < {self.config.min_length}")
            return ""
        
        if self.config.max_length and len(text) > self.config.max_length:
            text = text[:self.config.max_length]
        
        # Step 11: Tokenization
        if self.config.tokenize:
            tokens = self._tokenize(text)
            
            # Add special tokens
            if self.config.add_special_tokens:
                tokens = [self.config.bos_token] + tokens + [self.config.eos_token]
            
            return tokens
        
        return text
    
    def process_batch(self, data_batch: List[str], **kwargs) -> List[Union[str, List[str]]]:
        """
        Process a batch of text samples
        
        Args:
            data_batch: List of text strings
            **kwargs: Additional parameters
            
        Returns:
            List of processed texts
        """
        return [self.process(text, **kwargs) for text in data_batch]
    
    def _remove_accents(self, text: str) -> str:
        """
        Remove accents from text
        
        Args:
            text: Input text
            
        Returns:
            Text without accents
        """
        # Normalize to NFD (decomposed form)
        nfd = unicodedata.normalize('NFD', text)
        # Filter out combining characters
        return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text based on configured method
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if self.config.tokenizer_type == 'whitespace':
            return text.split()
        
        elif self.config.tokenizer_type == 'word':
            # Simple word tokenization
            return re.findall(r'\b\w+\b', text)
        
        elif self.config.tokenizer_type == 'sentence':
            # Simple sentence tokenization
            return re.split(r'[.!?]+', text)
        
        else:
            raise ValueError(f"Unknown tokenizer type: {self.config.tokenizer_type}")
    
    def build_vocabulary(self, texts: List[str], min_freq: int = 1) -> Dict[str, int]:
        """
        Build vocabulary from texts
        
        Args:
            texts: List of text samples
            min_freq: Minimum frequency for token inclusion
            
        Returns:
            Vocabulary dictionary (token -> ID)
        """
        from collections import Counter
        
        # Count tokens
        token_counts = Counter()
        for text in texts:
            tokens = self.process(text)
            if isinstance(tokens, list):
                token_counts.update(tokens)
        
        # Filter by frequency
        vocab = {
            self.config.pad_token: 0,
            self.config.unk_token: 1,
            self.config.bos_token: 2,
            self.config.eos_token: 3
        }
        
        idx = 4
        for token, count in token_counts.most_common():
            if count >= min_freq:
                vocab[token] = idx
                idx += 1
        
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        
        self.logger.info(f"Built vocabulary with {len(vocab)} tokens")
        return vocab
    
    def encode(self, tokens: List[str]) -> List[int]:
        """
        Encode tokens to IDs using vocabulary
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of token IDs
        """
        if not self.vocab:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        unk_id = self.vocab[self.config.unk_token]
        return [self.vocab.get(token, unk_id) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> List[str]:
        """
        Decode token IDs to tokens
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            List of tokens
        """
        if not self.reverse_vocab:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        return [self.reverse_vocab.get(idx, self.config.unk_token) for idx in token_ids]
    
    def get_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """
        Get statistics about text data
        
        Args:
            texts: List of text samples
            
        Returns:
            Dictionary with statistics
        """
        lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        return {
            'total_samples': len(texts),
            'avg_char_length': sum(lengths) / len(lengths) if lengths else 0,
            'min_char_length': min(lengths) if lengths else 0,
            'max_char_length': max(lengths) if lengths else 0,
            'avg_word_count': sum(word_counts) / len(word_counts) if word_counts else 0,
            'total_characters': sum(lengths),
            'vocabulary_size': len(self.vocab) if self.vocab else 0
        }


# Example usage
if __name__ == "__main__":
    # Create processor
    config = TextProcessorConfig(
        lowercase=True,
        remove_urls=True,
        remove_punctuation=False,
        tokenize=True,
        tokenizer_type='word'
    )
    
    processor = TextProcessor(config)
    
    # Process single text
    text = "Check out this link: https://example.com! Email me at test@email.com"
    processed = processor.process(text)
    print(f"Original: {text}")
    print(f"Processed: {processed}")
    
    # Process batch
    texts = [
        "Hello World!",
        "Natural Language Processing is fun.",
        "Text preprocessing is important."
    ]
    
    batch_processed = processor.process_batch(texts)
    print(f"\nBatch processed: {batch_processed}")
    
    # Build vocabulary
    vocab = processor.build_vocabulary(texts)
    print(f"\nVocabulary size: {len(vocab)}")
    
    # Encode/decode
    tokens = processed if isinstance(processed, list) else [processed]
    encoded = processor.encode(tokens)
    decoded = processor.decode(encoded)
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
