"""
Base loader class for data ingestion
All loaders inherit from this abstract class
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataModality(Enum):
    """Supported data modalities"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


@dataclass
class DataSample:
    """
    Standard data sample structure for all modalities
    """
    id: str
    modality: DataModality
    data: Any  # Raw data (text string, image bytes, audio array, etc.)
    metadata: Dict[str, Any]
    source: str
    
    def __repr__(self):
        return f"DataSample(id={self.id}, modality={self.modality.value}, source={self.source})"


@dataclass
class LoaderConfig:
    """Configuration for data loaders"""
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    cache_enabled: bool = False
    max_samples: Optional[int] = None
    modalities: List[DataModality] = None
    
    def __post_init__(self):
        if self.modalities is None:
            self.modalities = [DataModality.TEXT, DataModality.IMAGE]


class BaseLoader(ABC):
    """
    Abstract base class for all data loaders
    Defines the interface that all loaders must implement
    """
    
    def __init__(self, config: LoaderConfig):
        """
        Initialize base loader
        
        Args:
            config: LoaderConfig object with loader settings
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_config()
        self.logger.info(f"Initialized {self.__class__.__name__}")
    
    def _validate_config(self):
        """Validate loader configuration"""
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.config.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
    
    @abstractmethod
    def load(self, source: str, **kwargs) -> List[DataSample]:
        """
        Load data from source
        
        Args:
            source: Data source identifier (path, URL, etc.)
            **kwargs: Additional loader-specific arguments
            
        Returns:
            List of DataSample objects
        """
        pass
    
    @abstractmethod
    def load_batch(self, source: str, **kwargs) -> Generator[List[DataSample], None, None]:
        """
        Load data in batches (generator)
        
        Args:
            source: Data source identifier
            **kwargs: Additional arguments
            
        Yields:
            Batches of DataSample objects
        """
        pass
    
    @abstractmethod
    def validate_source(self, source: str) -> bool:
        """
        Validate if source is accessible and valid
        
        Args:
            source: Data source identifier
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def get_metadata(self, source: str) -> Dict[str, Any]:
        """
        Get metadata about the data source
        
        Args:
            source: Data source identifier
            
        Returns:
            Dictionary with metadata (count, size, modalities, etc.)
        """
        return {
            "source": source,
            "loader": self.__class__.__name__,
            "config": self.config.__dict__
        }
    
    def _filter_by_modality(self, samples: List[DataSample]) -> List[DataSample]:
        """
        Filter samples by configured modalities
        
        Args:
            samples: List of DataSample objects
            
        Returns:
            Filtered list of DataSample objects
        """
        if not self.config.modalities:
            return samples
        
        filtered = [s for s in samples if s.modality in self.config.modalities]
        self.logger.info(f"Filtered {len(samples)} -> {len(filtered)} samples by modality")
        return filtered
    
    def _apply_max_samples(self, samples: List[DataSample]) -> List[DataSample]:
        """
        Limit number of samples if max_samples is set
        
        Args:
            samples: List of DataSample objects
            
        Returns:
            Limited list of DataSample objects
        """
        if self.config.max_samples and len(samples) > self.config.max_samples:
            self.logger.info(f"Limiting samples from {len(samples)} to {self.config.max_samples}")
            return samples[:self.config.max_samples]
        return samples
    
    def count_by_modality(self, samples: List[DataSample]) -> Dict[str, int]:
        """
        Count samples by modality
        
        Args:
            samples: List of DataSample objects
            
        Returns:
            Dictionary with modality counts
        """
        counts = {}
        for sample in samples:
            modality_key = sample.modality.value
            counts[modality_key] = counts.get(modality_key, 0) + 1
        return counts
    
    def __repr__(self):
        return f"{self.__class__.__name__}(batch_size={self.config.batch_size})"
