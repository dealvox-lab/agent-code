"""
Complete examples for the Preprocessing Layer
Demonstrates all processors and common workflows
"""

import numpy as np
from PIL import Image

from data.preprocessing import (
    TextProcessor,
    TextProcessorConfig,
    ImageProcessor,
    ImageProcessorConfig,
    PreprocessingPipeline,
    PipelineConfig,
    ChainedPipeline
)

# Check for optional dependencies
try:
    from data.preprocessing import AudioProcessor, AudioProcessorConfig, AUDIO_AVAILABLE
except ImportError:
    AUDIO_AVAILABLE = False

try:
    from data.preprocessing import VideoProcessor, VideoProcessorConfig, VIDEO_AVAILABLE
except ImportError:
    VIDEO_AVAILABLE = False


def example_1_text_processing():
    """Example 1: Text Preprocessing"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Text Processing")
    print("="*60)
    
    # Create config
    config = TextProcessorConfig(
        lowercase=True,
        remove_urls=True,
        remove_emails=True,
        remove_punctuation=False,
        tokenize=True,
        tokenizer_type='word',
        max_length=100
    )
    
    processor = TextProcessor(config)
    
    # Single text
    text = "Check out https://example.com or email me at test@email.com!"
    processed = processor.process(text)
    print(f"Original: {text}")
    print(f"Processed: {processed}")
    
    # Batch processing
    texts = [
        "Natural Language Processing is amazing!",
        "Machine Learning changes everything.",
        "Deep Learning for the win!"
    ]
    
    batch_processed = processor.process_batch(texts)
    print(f"\nBatch processed {len(batch_processed)} texts")
    
    # Build vocabulary
    vocab = processor.build_vocabulary(texts, min_freq=1)
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Sample tokens: {list(vocab.keys())[:10]}")
    
    # Encode/decode
    tokens = batch_processed[0]
    encoded = processor.encode(tokens)
    decoded = processor.decode(encoded)
    print(f"\nOriginal tokens: {tokens}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Statistics
    stats = processor.get_statistics(texts)
    print(f"\nText statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def example_2_image_processing():
    """Example 2: Image Preprocessing"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Image Processing")
    print("="*60)
    
    # Create config with ImageNet normalization
    config = ImageProcessorConfig(
        resize=True,
        target_size=(224, 224),
        normalize_pixels=True,
        pixel_range='0-1',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        output_format='numpy',
        augment=False
    )
    
    processor = ImageProcessor(config)
    
    # Create sample images
    sample_img1 = Image.new('RGB', (512, 384), color=(255, 128, 64))
    sample_img2 = Image.new('RGB', (800, 600), color=(64, 128, 255))
    
    # Process single image
    processed = processor.process(sample_img1)
    print(f"Original size: {sample_img1.size}")
    print(f"Processed shape: {processed.shape}")
    print(f"Processed dtype: {processed.dtype}")
    print(f"Pixel range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Batch processing
    images = [sample_img1, sample_img2]
    batch_processed = processor.process_batch(images)
    print(f"\nBatch processed {len(batch_processed)} images")
    
    # Statistics
    stats = processor.get_statistics(images)
    print(f"\nImage statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # With augmentation
    aug_config = ImageProcessorConfig(
        resize=True,
        target_size=(224, 224),
        augment=True,
        horizontal_flip=True,
        rotation_range=15,
        brightness_range=(0.8, 1.2)
    )
    
    aug_processor = ImageProcessor(aug_config)
    augmented = aug_processor.process(sample_img1)
    print(f"\nAugmented image shape: {augmented.shape}")


def example_3_audio_processing():
    """Example 3: Audio Preprocessing"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Audio Processing")
    print("="*60)
    
    if not AUDIO_AVAILABLE:
        print("⚠️  Audio processing not available. Install with:")
        print("   pip install librosa soundfile")
        return
    
    # Create config
    config = AudioProcessorConfig(
        sample_rate=16000,
        mono=True,
        trim_silence=True,
        normalize_amplitude=True,
        extract_features=True,
        feature_type='mel_spectrogram',
        n_mels=80
    )
    
    processor = AudioProcessor(config)
    
    # Create synthetic audio (sine wave)
    duration = 2.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz (A4 note)
    
    # Process audio
    processed = processor.process((audio, sample_rate))
    print(f"Original audio shape: {audio.shape}")
    print(f"Processed features shape: {processed.shape}")
    print(f"Feature type: {config.feature_type}")
    
    # Extract multiple features
    multiple_features = processor.extract_multiple_features(audio)
    print(f"\nExtracted features:")
    for name, feature in multiple_features.items():
        print(f"  {name}: {feature.shape}")
    
    # Statistics
    stats = processor.get_statistics([(audio, sample_rate)])
    print(f"\nAudio statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def example_4_video_processing():
    """Example 4: Video Preprocessing"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Video Processing")
    print("="*60)
    
    if not VIDEO_AVAILABLE:
        print("⚠️  Video processing not available. Install with:")
        print("   pip install opencv-python")
        return
    
    # Create config
    config = VideoProcessorConfig(
        max_frames=16,
        frame_sampling='uniform',
        resize_frames=True,
        target_size=(224, 224),
        normalize_pixels=True,
        output_format='tensor',
        stack_frames=True
    )
    
    processor = VideoProcessor(config)
    
    print("Video processor initialized")
    print(f"Config: max_frames={config.max_frames}, target_size={config.target_size}")
    print("\nNote: Actual video processing requires a video file")
    print("Example: processed = processor.process('video.mp4')")


def example_5_multimodal_pipeline():
    """Example 5: Multi-Modal Pipeline"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Multi-Modal Pipeline")
    print("="*60)
    
    # Create pipeline config
    config = PipelineConfig(
        text_config=TextProcessorConfig(
            lowercase=True,
            tokenize=True,
            max_length=512
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
    
    # Process different modalities
    text = "A beautiful sunset over the ocean"
    image = Image.new('RGB', (800, 600), color=(255, 165, 0))
    
    text_result = pipeline.process(text, 'text')
    image_result = pipeline.process(image, 'image')
    
    print(f"Text processed: {text_result.processed_data}")
    print(f"Image processed shape: {image_result.processed_data.shape}")
    
    # Multi-modal processing
    multimodal_data = {
        'text': text,
        'image': image
    }
    
    results = pipeline.process_multimodal(multimodal_data)
    print(f"\nMulti-modal results: {list(results.keys())}")
    
    # Batch multi-modal processing
    batch = [
        {'text': "Sample 1", 'image': Image.new('RGB', (100, 100))},
        {'text': "Sample 2", 'image': Image.new('RGB', (100, 100))}
    ]
    
    batch_results = pipeline.process_multimodal_batch(batch)
    print(f"Processed {len(batch_results)} multi-modal samples")
    
    # Get pipeline statistics
    stats = pipeline.get_statistics()
    print(f"\nPipeline statistics:")
    print(f"Active processors: {stats['active_processors']}")


def example_6_custom_chain():
    """Example 6: Custom Processing Chain"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Custom Processing Chain")
    print("="*60)
    
    # Create custom operations
    def step1(data):
        """Convert to lowercase"""
        return data.lower()
    
    def step2(data):
        """Remove extra spaces"""
        return ' '.join(data.split())
    
    def step3(data):
        """Add prefix"""
        return f"[PROCESSED] {data}"
    
    # Build chain
    chain = ChainedPipeline()
    chain.add_operation(step1, 'lowercase')
    chain.add_operation(step2, 'clean_spaces')
    chain.add_operation(step3, 'add_prefix')
    
    # Process
    text = "  HELLO   WORLD  "
    result = chain.process(text)
    
    print(f"Original: '{text}'")
    print(f"Result: '{result}'")


def example_7_end_to_end_workflow():
    """Example 7: Complete End-to-End Workflow"""
    print("\n" + "="*60)
    print("EXAMPLE 7: End-to-End Workflow")
    print("="*60)
    
    # Simulate data from ingestion layer
    from data.ingestion import DataSample, DataModality
    
    samples = [
        DataSample(
            id='text_001',
            modality=DataModality.TEXT,
            data="Natural Language Processing is powerful!",
            metadata={'source': 'dataset.txt'},
            source='./data/text/sample1.txt'
        ),
        DataSample(
            id='image_001',
            modality=DataModality.IMAGE,
            data=Image.new('RGB', (512, 512), color=(100, 150, 200)),
            metadata={'source': 'image.jpg'},
            source='./data/images/sample1.jpg'
        )
    ]
    
    # Create preprocessing pipeline
    pipeline = PreprocessingPipeline(PipelineConfig())
    
    # Process each sample
    processed_samples = []
    for sample in samples:
        modality_str = sample.modality.value
        processed = pipeline.process(sample.data, modality_str)
        processed_samples.append(processed)
        
        print(f"Processed {sample.id} ({modality_str})")
        if modality_str == 'text':
            print(f"  Tokens: {processed.processed_data[:5]}...")
        elif modality_str == 'image':
            print(f"  Shape: {processed.processed_data.shape}")
    
    print(f"\nTotal processed: {len(processed_samples)} samples")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("PREPROCESSING LAYER - EXAMPLES")
    print("="*60)
    
    try:
        example_1_text_processing()
    except Exception as e:
        print(f"Example 1 error: {e}")
    
    try:
        example_2_image_processing()
    except Exception as e:
        print(f"Example 2 error: {e}")
    
    try:
        example_3_audio_processing()
    except Exception as e:
        print(f"Example 3 error: {e}")
    
    try:
        example_4_video_processing()
    except Exception as e:
        print(f"Example 4 error: {e}")
    
    try:
        example_5_multimodal_pipeline()
    except Exception as e:
        print(f"Example 5 error: {e}")
    
    try:
        example_6_custom_chain()
    except Exception as e:
        print(f"Example 6 error: {e}")
    
    try:
        example_7_end_to_end_workflow()
    except Exception as e:
        print(f"Example 7 error: {e}")
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
