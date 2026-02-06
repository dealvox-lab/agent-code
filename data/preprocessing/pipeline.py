"""
Preprocessing Pipeline Orchestrator
Coordinates multi-modal preprocessing workflows
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import logging

from .base_processor import BaseProcessor, ProcessedSample
from .text_processor import TextProcessor, TextProcessorConfig
from .image_processor import ImageProcessor, ImageProcessorConfig
from .audio_processor import AudioProcessor, AudioProcessorConfig
from .video_processor import VideoProcessor, VideoProcessorConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for preprocessing pipeline"""
    # Processor configurations
    text_config: Optional[TextProcessorConfig] = None
    image_config: Optional[ImageProcessorConfig] = None
    audio_config: Optional[AudioProcessorConfig] = None
    video_config: Optional[VideoProcessorConfig] = None
    
    # Pipeline options
    parallel_processing: bool = False
    num_workers: int = 4
    error_handling: str = 'skip'  # 'skip', 'raise', 'log'
    save_intermediate: bool = False
    
    def __post_init__(self):
        """Initialize default configs if not provided"""
        if self.text_config is None:
            self.text_config = TextProcessorConfig()
        if self.image_config is None:
            self.image_config = ImageProcessorConfig()
        if self.audio_config is None:
            self.audio_config = AudioProcessorConfig(enabled=False)
        if self.video_config is None:
            self.video_config = VideoProcessorConfig(enabled=False)


class PreprocessingPipeline:
    """
    Multi-modal preprocessing pipeline
    Orchestrates text, image, audio, and video preprocessing
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize preprocessing pipeline
        
        Args:
            config: PipelineConfig object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize processors
        self.processors: Dict[str, BaseProcessor] = {}
        self._initialize_processors()
        
        self.logger.info(f"Pipeline initialized with {len(self.processors)} active processors")
    
    def _initialize_processors(self):
        """Initialize all enabled processors"""
        if self.config.text_config.enabled:
            self.processors['text'] = TextProcessor(self.config.text_config)
            self.logger.info("Text processor enabled")
        
        if self.config.image_config.enabled:
            self.processors['image'] = ImageProcessor(self.config.image_config)
            self.logger.info("Image processor enabled")
        
        if self.config.audio_config.enabled:
            try:
                self.processors['audio'] = AudioProcessor(self.config.audio_config)
                self.logger.info("Audio processor enabled")
            except ImportError as e:
                self.logger.warning(f"Audio processor not available: {e}")
        
        if self.config.video_config.enabled:
            try:
                self.processors['video'] = VideoProcessor(self.config.video_config)
                self.logger.info("Video processor enabled")
            except ImportError as e:
                self.logger.warning(f"Video processor not available: {e}")
    
    def process(self, data: Any, modality: str, **kwargs) -> ProcessedSample:
        """
        Process a single data sample
        
        Args:
            data: Input data
            modality: Data modality ('text', 'image', 'audio', 'video')
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessedSample object
        """
        if modality not in self.processors:
            raise ValueError(f"No processor available for modality: {modality}")
        
        processor = self.processors[modality]
        
        try:
            return processor.preprocess(data, **kwargs)
        except Exception as e:
            if self.config.error_handling == 'raise':
                raise
            elif self.config.error_handling == 'log':
                self.logger.error(f"Error processing {modality}: {e}")
            return None
    
    def process_batch(self, data_batch: List[Any], modality: str, **kwargs) -> List[ProcessedSample]:
        """
        Process a batch of data samples
        
        Args:
            data_batch: List of input data
            modality: Data modality
            **kwargs: Additional parameters
            
        Returns:
            List of ProcessedSample objects
        """
        if modality not in self.processors:
            raise ValueError(f"No processor available for modality: {modality}")
        
        processor = self.processors[modality]
        
        if self.config.parallel_processing:
            return self._process_batch_parallel(data_batch, processor, **kwargs)
        else:
            return processor.preprocess_batch(data_batch, **kwargs)
    
    def _process_batch_parallel(self, data_batch: List[Any], 
                                processor: BaseProcessor, **kwargs) -> List[ProcessedSample]:
        """
        Process batch in parallel using multiprocessing
        
        Args:
            data_batch: List of input data
            processor: Processor instance
            **kwargs: Additional parameters
            
        Returns:
            List of ProcessedSample objects
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        results = []
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = {
                executor.submit(processor.preprocess, data, **kwargs): i 
                for i, data in enumerate(data_batch)
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    if self.config.error_handling == 'raise':
                        raise
                    elif self.config.error_handling == 'log':
                        self.logger.error(f"Error in parallel processing: {e}")
        
        return results
    
    def process_multimodal(self, data_dict: Dict[str, Any], **kwargs) -> Dict[str, ProcessedSample]:
        """
        Process multi-modal data
        
        Args:
            data_dict: Dictionary with modality -> data mapping
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with modality -> ProcessedSample mapping
        """
        results = {}
        
        for modality, data in data_dict.items():
            try:
                result = self.process(data, modality, **kwargs)
                if result is not None:
                    results[modality] = result
            except Exception as e:
                if self.config.error_handling == 'raise':
                    raise
                self.logger.error(f"Error processing {modality}: {e}")
        
        return results
    
    def process_multimodal_batch(self, batch: List[Dict[str, Any]], 
                                 **kwargs) -> List[Dict[str, ProcessedSample]]:
        """
        Process batch of multi-modal samples
        
        Args:
            batch: List of dictionaries with modality -> data mapping
            **kwargs: Additional parameters
            
        Returns:
            List of dictionaries with processed samples
        """
        return [self.process_multimodal(sample, **kwargs) for sample in batch]
    
    def get_processor(self, modality: str) -> Optional[BaseProcessor]:
        """
        Get processor for specific modality
        
        Args:
            modality: Modality name
            
        Returns:
            Processor instance or None
        """
        return self.processors.get(modality)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from all processors
        
        Returns:
            Dictionary with processor statistics
        """
        stats = {
            'active_processors': list(self.processors.keys()),
            'processors': {}
        }
        
        for modality, processor in self.processors.items():
            stats['processors'][modality] = processor.get_stats()
        
        return stats
    
    def clear_all_caches(self):
        """Clear caches from all processors"""
        for processor in self.processors.values():
            processor.clear_cache()
        self.logger.info("Cleared all processor caches")
    
    def validate_data(self, data: Any, modality: str) -> bool:
        """
        Validate data for specific modality
        
        Args:
            data: Input data
            modality: Data modality
            
        Returns:
            True if valid, False otherwise
        """
        if modality not in self.processors:
            return False
        
        return self.processors[modality].validate(data)
    
    def __repr__(self):
        return f"PreprocessingPipeline(processors={list(self.processors.keys())})"


class ChainedPipeline:
    """
    Chain multiple preprocessing operations sequentially
    Useful for complex preprocessing workflows
    """
    
    def __init__(self):
        """Initialize chained pipeline"""
        self.operations: List[tuple] = []
        self.logger = logging.getLogger(__name__)
    
    def add_operation(self, func: callable, name: str, **kwargs):
        """
        Add preprocessing operation to chain
        
        Args:
            func: Processing function
            name: Operation name
            **kwargs: Arguments for the function
        """
        self.operations.append((name, func, kwargs))
        self.logger.info(f"Added operation: {name}")
    
    def process(self, data: Any) -> Any:
        """
        Process data through all operations
        
        Args:
            data: Input data
            
        Returns:
            Processed data
        """
        result = data
        
        for name, func, kwargs in self.operations:
            try:
                result = func(result, **kwargs)
                self.logger.debug(f"Completed operation: {name}")
            except Exception as e:
                self.logger.error(f"Error in operation {name}: {e}")
                raise
        
        return result
    
    def clear(self):
        """Clear all operations"""
        self.operations.clear()
        self.logger.info("Cleared all operations")


# Example usage
if __name__ == "__main__":
    # Create pipeline configuration
    config = PipelineConfig(
        text_config=TextProcessorConfig(
            lowercase=True,
            remove_urls=True,
            tokenize=True
        ),
        image_config=ImageProcessorConfig(
            resize=True,
            target_size=(224, 224),
            normalize_pixels=True
        ),
        parallel_processing=False,
        error_handling='log'
    )
    
    # Initialize pipeline
    pipeline = PreprocessingPipeline(config)
    
    # Process text
    text_sample = "Check out https://example.com for more info!"
    text_result = pipeline.process(text_sample, 'text')
    print(f"Text processed: {text_result}")
    
    # Process image
    from PIL import Image
    image_sample = Image.new('RGB', (512, 512), color=(255, 128, 0))
    image_result = pipeline.process(image_sample, 'image')
    print(f"Image processed shape: {image_result.processed_data.shape}")
    
    # Process multi-modal
    multimodal_data = {
        'text': "Sample text",
        'image': image_sample
    }
    multimodal_result = pipeline.process_multimodal(multimodal_data)
    print(f"Multi-modal processed: {list(multimodal_result.keys())}")
    
    # Get statistics
    stats = pipeline.get_statistics()
    print(f"\nPipeline statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
