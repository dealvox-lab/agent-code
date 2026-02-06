# Data Ingestion Layer - Multi-Modal LLM Training

Complete data ingestion system for loading multi-modal training data from various sources.

## Features

✅ **Multiple Data Sources**
- AWS S3 buckets
- Local file systems
- Streaming data (coming soon)

✅ **Multi-Modal Support**
- Text (.txt, .json, .csv, .md)
- Images (.jpg, .png, .gif, .webp)
- Audio (.mp3, .wav, .flac, .ogg)
- Video (.mp4, .avi, .mov, .mkv)

✅ **Efficient Loading**
- Batch processing
- Memory-efficient generators
- Parallel loading support
- Configurable caching

✅ **Data Management**
- Source validation
- Metadata extraction
- Modality balancing
- Quality filtering

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For audio support
pip install librosa soundfile

# For video support
pip install opencv-python moviepy
```

## Quick Start

### 1. Basic Usage - Local Loader

```python
from data.ingestion import LocalLoader, LoaderConfig, DataModality

# Configure loader
config = LoaderConfig(
    batch_size=32,
    modalities=[DataModality.TEXT, DataModality.IMAGE],
    max_samples=1000
)

# Create loader
loader = LocalLoader(config=config)

# Load data
samples = loader.load('./training_data')
print(f"Loaded {len(samples)} samples")

# Load in batches
for batch in loader.load_batch('./training_data'):
    print(f"Batch size: {len(batch)}")
```

### 2. S3 Loader

```python
from data.ingestion import S3Loader
import os

# Create S3 loader
loader = S3Loader(
    config=config,
    aws_access_key=os.getenv('AWS_ACCESS_KEY'),
    aws_secret_key=os.getenv('AWS_SECRET_KEY'),
    region='us-east-1'
)

# Load from S3
samples = loader.load('s3://my-bucket/training-data/')

# Get metadata
metadata = loader.get_metadata('my-bucket/training-data/')
print(metadata)
```

### 3. Multi-Modal Orchestrator (Recommended)

```python
from data.ingestion import MultiModalLoader, LoaderConfig, DataModality
import os

# Configure
config = LoaderConfig(
    batch_size=64,
    modalities=[DataModality.TEXT, DataModality.IMAGE]
)

# Create orchestrator
orchestrator = MultiModalLoader(
    config=config,
    aws_credentials={
        'access_key': os.getenv('AWS_ACCESS_KEY'),
        'secret_key': os.getenv('AWS_SECRET_KEY'),
        'region': 'us-east-1'
    }
)

# Add multiple sources
orchestrator.add_local_source(
    name='local_images',
    local_path='./data/images',
    modalities=[DataModality.IMAGE]
)

orchestrator.add_s3_source(
    name='s3_text',
    s3_path='my-bucket/text-data/',
    modalities=[DataModality.TEXT]
)

# Validate all sources
validation = orchestrator.validate_all_sources()

# Get statistics
stats = orchestrator.get_statistics()

# Load combined data
samples = orchestrator.load_combined(balance_by_modality=True)
```

## Configuration

Edit `config/data_config.yaml`:

```yaml
loader:
  batch_size: 32
  num_workers: 4
  max_samples: null

modalities:
  - text
  - image

sources:
  - name: local_data
    type: local
    path: ./data
    weight: 1.0
```

## Data Structure

Each data sample is returned as a `DataSample` object:

```python
@dataclass
class DataSample:
    id: str                    # Unique identifier
    modality: DataModality     # TEXT, IMAGE, AUDIO, VIDEO
    data: Any                  # Raw data (varies by modality)
    metadata: Dict[str, Any]   # File metadata
    source: str                # Source path/URL
```

## File Structure

```
data/
└── ingestion/
    ├── __init__.py
    ├── base_loader.py          # Abstract base class
    ├── s3_loader.py            # S3 data loading
    ├── local_loader.py         # Local file system
    └── multi_modal_loader.py   # Orchestrator

config/
└── data_config.yaml            # Configuration

requirements.txt                # Dependencies
```

## GitHub Repository Structure

```
training_service/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── config/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
├── data/
│   ├── __init__.py
│   └── ingestion/
│       ├── __init__.py
│       ├── base_loader.py
│       ├── s3_loader.py
│       ├── local_loader.py
│       └── multi_modal_loader.py
├── models/
│   └── (to be added)
├── training/
│   └── (to be added)
├── tests/
│   └── test_ingestion.py
└── examples/
    └── basic_usage.py
```

## Advanced Usage

### Custom Filtering

```python
# Filter by file size
config = LoaderConfig(
    batch_size=32,
    max_samples=None
)

loader = LocalLoader(config)
samples = loader.load('./data')

# Filter large images
filtered = [s for s in samples 
            if s.modality == DataModality.IMAGE 
            and s.metadata['size'] < 5_000_000]  # 5MB
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor

def load_source(source):
    loader = LocalLoader(config)
    return loader.load(source)

sources = ['./data1', './data2', './data3']

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(load_source, sources))
```

## Environment Variables

Create a `.env` file:

```bash
# AWS Credentials
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
```

## Next Steps

After completing data ingestion:

1. **Preprocessing Layer** - Clean and transform data
2. **Validation Layer** - Quality checks and filtering
3. **Dataset Preparation** - Create training batches
4. **Model Training** - Feed data to GPT 5.2

# Preprocessing Layer - Multi-Modal LLM Training

Comprehensive preprocessing system for multi-modal data (text, images, audio, video).

## Features

✅ **Multi-Modal Processing**
- Text: Cleaning, tokenization, normalization, vocabulary building
- Images: Resizing, normalization, augmentation (ImageNet-compatible)
- Audio: Spectrograms, MFCC, mel-spectrograms, resampling
- Video: Frame extraction, temporal processing, keyframe detection

✅ **Flexible Configuration**
- YAML-based configuration
- Per-modality customization
- Augmentation support

✅ **Production-Ready**
- Batch processing
- Parallel processing
- Error handling
- Caching support

## Installation

### Basic (Text + Images only)
```bash
pip install numpy pillow pyyaml
```

### Full (All modalities)
```bash
# Install all dependencies
pip install numpy pillow librosa soundfile opencv-python pyyaml

# Or use the requirements file
pip install -r requirements_preprocessing.txt
```

### Optional Components
```bash
# Audio only
pip install librosa soundfile

# Video only
pip install opencv-python

# For servers (headless OpenCV)
pip install opencv-python-headless
```

## Quick Start

### 1. Text Processing

```python
from data.preprocessing import TextProcessor, TextProcessorConfig

# Configure
config = TextProcessorConfig(
    lowercase=True,
    remove_urls=True,
    tokenize=True,
    tokenizer_type='word',
    max_length=512
)

# Create processor
processor = TextProcessor(config)

# Process text
text = "Check out https://example.com for more!"
processed = processor.process(text)
print(processed)  # ['check', 'out', 'for', 'more']

# Build vocabulary
texts = ["Hello world", "Natural language processing"]
vocab = processor.build_vocabulary(texts)
print(f"Vocabulary size: {len(vocab)}")
```

### 2. Image Processing

```python
from data.preprocessing import ImageProcessor, ImageProcessorConfig
from PIL import Image

# Configure (ImageNet normalization)
config = ImageProcessorConfig(
    resize=True,
    target_size=(224, 224),
    normalize_pixels=True,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    output_format='numpy'
)

# Create processor
processor = ImageProcessor(config)

# Process image
img = Image.open('photo.jpg')
processed = processor.process(img)
print(processed.shape)  # (224, 224, 3)
print(processed.dtype)  # float32
```

### 3. Audio Processing

```python
from data.preprocessing import AudioProcessor, AudioProcessorConfig

# Configure
config = AudioProcessorConfig(
    sample_rate=16000,
    trim_silence=True,
    extract_features=True,
    feature_type='mel_spectrogram',
    n_mels=80
)

# Create processor
processor = AudioProcessor(config)

# Process audio
processed = processor.process('audio.wav')
print(processed.shape)  # (80, time_steps)
```

### 4. Video Processing

```python
from data.preprocessing import VideoProcessor, VideoProcessorConfig

# Configure
config = VideoProcessorConfig(
    max_frames=16,
    frame_sampling='uniform',
    target_size=(224, 224),
    normalize_pixels=True,
    output_format='tensor'
)

# Create processor
processor = VideoProcessor(config)

# Process video
processed = processor.process('video.mp4')
print(processed.shape)  # (3, 16, 224, 224) for RGB
```

### 5. Multi-Modal Pipeline (Recommended)

```python
from data.preprocessing import (
    PreprocessingPipeline,
    PipelineConfig,
    TextProcessorConfig,
    ImageProcessorConfig
)

# Configure pipeline
config = PipelineConfig(
    text_config=TextProcessorConfig(lowercase=True, tokenize=True),
    image_config=ImageProcessorConfig(resize=True, target_size=(224, 224)),
    parallel_processing=True,
    num_workers=4
)

# Create pipeline
pipeline = PreprocessingPipeline(config)

# Process single modality
text_result = pipeline.process("Sample text", 'text')
image_result = pipeline.process(image, 'image')

# Process multi-modal data
multimodal_data = {
    'text': "A photo of a cat",
    'image': cat_image
}
results = pipeline.process_multimodal(multimodal_data)
```

## Configuration

### Using YAML Configuration

```python
import yaml
from data.preprocessing import PreprocessingPipeline, PipelineConfig

# Load config
with open('config/preprocessing_config.yaml') as f:
    config_dict = yaml.safe_load(f)

# Create pipeline
# (You'll need to convert dict to config objects)
```

### Configuration Options

#### Text Processing
- **Cleaning**: URLs, emails, phone numbers, punctuation
- **Normalization**: Unicode, case, accents
- **Tokenization**: Whitespace, word, sentence-level
- **Vocabulary**: Custom vocabulary building

#### Image Processing
- **Resize**: Multiple modes (bicubic, bilinear, etc.)
- **Normalization**: ImageNet or custom mean/std
- **Augmentation**: Flips, rotation, brightness, contrast
- **Format**: NumPy, PIL, or PyTorch tensor

#### Audio Processing
- **Resampling**: Target sample rate
- **Features**: Raw, spectrogram, mel-spectrogram, MFCC
- **Normalization**: Peak or RMS
- **Augmentation**: Time stretch, pitch shift, noise

#### Video Processing
- **Frame Sampling**: Uniform, random, or all frames
- **Temporal**: Clip duration, keyframe extraction
- **Processing**: Per-frame image processing
- **Format**: Stacked arrays or frame lists

## Advanced Usage

### Batch Processing

```python
# Process batch of images
images = [Image.open(f'img{i}.jpg') for i in range(10)]
batch_results = processor.process_batch(images)
```

### Custom Pipeline Chain

```python
from data.preprocessing import ChainedPipeline

# Create custom chain
chain = ChainedPipeline()
chain.add_operation(preprocess_func1, 'step1', param=value)
chain.add_operation(preprocess_func2, 'step2')

# Process
result = chain.process(data)
```

### Multiple Feature Extraction (Audio)

```python
# Extract all features at once
features = processor.extract_multiple_features(audio)
# Returns: spectrogram, mel_spectrogram, mfcc, zcr, spectral_centroid, rms
```

### Keyframe Extraction (Video)

```python
# Extract scene change keyframes
keyframes = processor.extract_keyframes('video.mp4', threshold=0.3)
```

## Integration with Data Ingestion

```python
from data.ingestion import LocalLoader, LoaderConfig, DataModality
from data.preprocessing import PreprocessingPipeline, PipelineConfig

# Load data
loader = LocalLoader(LoaderConfig())
samples = loader.load('./data/images')

# Preprocess
pipeline = PreprocessingPipeline(PipelineConfig())

for sample in samples:
    if sample.modality == DataModality.IMAGE:
        processed = pipeline.process(sample.data, 'image')
        print(f"Processed: {processed.processed_data.shape}")
```

## File Structure

```
data/preprocessing/
├── __init__.py
├── base_processor.py          # Abstract base class
├── text_processor.py          # Text preprocessing
├── image_processor.py         # Image preprocessing
├── audio_processor.py         # Audio preprocessing
├── video_processor.py         # Video preprocessing
└── pipeline.py                # Pipeline orchestrator

config/
└── preprocessing_config.yaml  # Configuration

requirements_preprocessing.txt  # Dependencies
```

## Output Formats

### Text
- **String**: Cleaned text
- **List[str]**: Tokens
- **List[int]**: Token IDs (after vocabulary building)

### Image
- **NumPy**: (H, W, C) array
- **PIL**: PIL Image object
- **Tensor**: (C, H, W) PyTorch-compatible

### Audio
- **NumPy**: Feature arrays (1D waveform or 2D spectrogram)
- **Tensor**: With channel dimension

### Video
- **NumPy**: (T, H, W, C) frames
- **Tensor**: (C, T, H, W) for 3D CNNs
- **List**: Individual frame arrays

## Statistics & Monitoring

```python
# Get processor statistics
stats = processor.get_statistics(data_samples)
print(stats)

# Get pipeline statistics
pipeline_stats = pipeline.get_statistics()
print(pipeline_stats)
```

## Error Handling

```python
# Configure error handling
config = PipelineConfig(
    error_handling='log'  # 'skip', 'raise', or 'log'
)

# Invalid samples will be logged but won't stop processing
```

## Performance Tips

1. **Enable caching** for repeated processing
2. **Use batch processing** when possible
3. **Enable parallel processing** for large datasets
4. **Choose appropriate output format** (NumPy vs Tensor)
5. **Disable unnecessary features** (e.g., augmentation during inference)

## Next Steps

After preprocessing:
1. ✅ Data Ingestion (COMPLETE)
2. ✅ Preprocessing (COMPLETE)
3. ⏭️ Data Validation (NEXT)
4. ⏭️ Dataset Preparation
5. ⏭️ Model Training

## Troubleshooting

### Audio Issues
```
ImportError: Audio processing requires librosa and soundfile
```
**Solution**: `pip install librosa soundfile`

### Video Issues
```
ImportError: Video processing requires opencv
```
**Solution**: `pip install opencv-python`

### Memory Issues
- Use batch processing with smaller batches
- Enable parallel processing
- Disable caching if not needed

## Contributing

See the main project README for contribution guidelines.

## License

MIT License
