"""
Base Processor for Multi-Modal Data Preprocessing
All processors inherit from this abstract class
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """Base configuration for all processors"""
    enabled: bool = True
    normalize: bool = True
    cache_processed: bool = False
    verbose: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'enabled': self.enabled,
            'normalize': self.normalize,
            'cache_processed': self.cache_processed,
            'verbose': self.verbose
        }


@dataclass
class ProcessedSample:
    """
    Standard structure for processed data samples
    """
    id: str
    original_data: Any  # Original input data
    processed_data: Any  # Processed output data
    metadata: Dict[str, Any]  # Processing metadata
    modality: str  # text, image, audio, video
    
    def __repr__(self):
        return f"ProcessedSample(id={self.id}, modality={self.modality})"


class BaseProcessor(ABC):
    """
    Abstract base class for all data processors
    Defines the interface for preprocessing operations
    """
    
    def __init__(self, config: ProcessorConfig):
        """
        Initialize base processor
        
        Args:
            config: ProcessorConfig object with settings
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cache: Dict[str, Any] = {}
        
        if config.verbose:
            self.logger.setLevel(logging.DEBUG)
        
        self.logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """
        Process a single data sample
        
        Args:
            data: Input data to process
            **kwargs: Additional processing parameters
            
        Returns:
            Processed data
        """
        pass
    
    @abstractmethod
    def process_batch(self, data_batch: List[Any], **kwargs) -> List[Any]:
        """
        Process a batch of data samples
        
        Args:
            data_batch: List of input data samples
            **kwargs: Additional processing parameters
            
        Returns:
            List of processed data samples
        """
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """
        Validate input data before processing
        
        Args:
            data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def preprocess(self, data: Any, **kwargs) -> ProcessedSample:
        """
        Full preprocessing pipeline for a single sample
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            ProcessedSample object
        """
        # Validate
        if not self.validate(data):
            raise ValueError(f"Invalid input data for {self.__class__.__name__}")
        
        # Check cache
        cache_key = self._get_cache_key(data)
        if self.config.cache_processed and cache_key in self._cache:
            self.logger.debug(f"Using cached result for {cache_key}")
            return self._cache[cache_key]
        
        # Process
        processed = self.process(data, **kwargs)
        
        # Create result
        result = ProcessedSample(
            id=cache_key,
            original_data=data if not self.config.cache_processed else None,
            processed_data=processed,
            metadata=self._get_processing_metadata(data, processed),
            modality=self._get_modality()
        )
        
        # Cache if enabled
        if self.config.cache_processed:
            self._cache[cache_key] = result
        
        return result
    
    def preprocess_batch(self, data_batch: List[Any], **kwargs) -> List[ProcessedSample]:
        """
        Full preprocessing pipeline for a batch
        
        Args:
            data_batch: List of input data
            **kwargs: Additional parameters
            
        Returns:
            List of ProcessedSample objects
        """
        results = []
        
        for i, data in enumerate(data_batch):
            try:
                result = self.preprocess(data, **kwargs)
                results.append(result)
                
                if self.config.verbose and (i + 1) % 100 == 0:
                    self.logger.debug(f"Processed {i + 1}/{len(data_batch)} samples")
            except Exception as e:
                self.logger.warning(f"Failed to process sample {i}: {e}")
        
        self.logger.info(f"Successfully processed {len(results)}/{len(data_batch)} samples")
        return results
    
    def _get_cache_key(self, data: Any) -> str:
        """
        Generate cache key for data
        
        Args:
            data: Input data
            
        Returns:
            Cache key string
        """
        # Simple hash-based key
        return f"{self.__class__.__name__}_{hash(str(data))}"
    
    def _get_processing_metadata(self, original: Any, processed: Any) -> Dict[str, Any]:
        """
        Generate metadata about the processing
        
        Args:
            original: Original input data
            processed: Processed output data
            
        Returns:
            Metadata dictionary
        """
        return {
            'processor': self.__class__.__name__,
            'config': self.config.to_dict(),
            'original_type': type(original).__name__,
            'processed_type': type(processed).__name__
        }
    
    @abstractmethod
    def _get_modality(self) -> str:
        """
        Get the modality this processor handles
        
        Returns:
            Modality string (text, image, audio, video)
        """
        pass
    
    def clear_cache(self):
        """Clear the processing cache"""
        self._cache.clear()
        self.logger.info("Cache cleared")
    
    def get_cache_size(self) -> int:
        """Get number of cached items"""
        return len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics
        
        Returns:
            Dictionary with statistics
        """
        return {
            'processor': self.__class__.__name__,
            'modality': self._get_modality(),
            'cache_size': self.get_cache_size(),
            'config': self.config.to_dict()
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(enabled={self.config.enabled})"
