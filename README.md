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

## Contributing

See main project README for contribution guidelines.

## License

MIT License
