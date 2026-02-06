"""
Preprocessing Layer for Multi-Modal LLM Training
Provides processors for text, image, audio, and video data
"""

from .base_processor import (
    BaseProcessor,
    ProcessorConfig,
    ProcessedSample
)

from .text_processor import (
    TextProcessor,
    TextProcessorConfig
)

from .image_processor import (
    ImageProcessor,
    ImageProcessorConfig
)

# Optional imports with availability checks
try:
    from .audio_processor import (
        AudioProcessor,
        AudioProcessorConfig
    )
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    AudioProcessor = None
    AudioProcessorConfig = None

try:
    from .video_processor import (
        VideoProcessor,
        VideoProcessorConfig
    )
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False
    VideoProcessor = None
    VideoProcessorConfig = None

from .pipeline import (
    PreprocessingPipeline,
    PipelineConfig,
    ChainedPipeline
)

__version__ = "1.0.0"

__all__ = [
    # Base classes
    "BaseProcessor",
    "ProcessorConfig",
    "ProcessedSample",
    
    # Text
    "TextProcessor",
    "TextProcessorConfig",
    
    # Image
    "ImageProcessor",
    "ImageProcessorConfig",
    
    # Audio (optional)
    "AudioProcessor",
    "AudioProcessorConfig",
    
    # Video (optional)
    "VideoProcessor",
    "VideoProcessorConfig",
    
    # Pipeline
    "PreprocessingPipeline",
    "PipelineConfig",
    "ChainedPipeline",
    
    # Availability flags
    "AUDIO_AVAILABLE",
    "VIDEO_AVAILABLE",
]
