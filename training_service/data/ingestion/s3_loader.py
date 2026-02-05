"""
S3 Data Loader - Load multi-modal data from AWS S3
Integrates with S3DatasetOrchestrator
"""

import os
import json
from typing import List, Dict, Any, Generator, Optional
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from PIL import Image
import io

from .base_loader import BaseLoader, DataSample, DataModality, LoaderConfig


class S3Loader(BaseLoader):
    """
    Load multi-modal data from AWS S3 buckets
    Supports: text, images, audio, video files
    """
    
    # File extension to modality mapping
    EXTENSION_MAP = {
        # Text
        '.txt': DataModality.TEXT,
        '.json': DataModality.TEXT,
        '.csv': DataModality.TEXT,
        '.md': DataModality.TEXT,
        
        # Images
        '.jpg': DataModality.IMAGE,
        '.jpeg': DataModality.IMAGE,
        '.png': DataModality.IMAGE,
        '.gif': DataModality.IMAGE,
        '.bmp': DataModality.IMAGE,
        '.webp': DataModality.IMAGE,
        
        # Audio
        '.mp3': DataModality.AUDIO,
        '.wav': DataModality.AUDIO,
        '.flac': DataModality.AUDIO,
        '.ogg': DataModality.AUDIO,
        '.m4a': DataModality.AUDIO,
        
        # Video
        '.mp4': DataModality.VIDEO,
        '.avi': DataModality.VIDEO,
        '.mov': DataModality.VIDEO,
        '.mkv': DataModality.VIDEO,
        '.webm': DataModality.VIDEO,
    }
    
    def __init__(self, config: LoaderConfig, aws_access_key: str, 
                 aws_secret_key: str, region: str = 'us-east-1'):
        """
        Initialize S3 Loader
        
        Args:
            config: LoaderConfig object
            aws_access_key: AWS Access Key ID
            aws_secret_key: AWS Secret Access Key
            region: AWS region
        """
        super().__init__(config)
        
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        self.region = region
        self.logger.info(f"S3 client initialized for region: {region}")
    
    def validate_source(self, source: str) -> bool:
        """
        Validate S3 bucket/path exists
        
        Args:
            source: S3 path in format 's3://bucket/prefix' or 'bucket/prefix'
            
        Returns:
            True if accessible, False otherwise
        """
        try:
            bucket, prefix = self._parse_s3_path(source)
            self.s3_client.head_bucket(Bucket=bucket)
            self.logger.info(f"Validated S3 source: {bucket}/{prefix}")
            return True
        except ClientError as e:
            self.logger.error(f"Failed to validate S3 source: {e}")
            return False
    
    def _parse_s3_path(self, source: str) -> tuple:
        """
        Parse S3 path into bucket and prefix
        
        Args:
            source: S3 path (s3://bucket/prefix or bucket/prefix)
            
        Returns:
            Tuple of (bucket, prefix)
        """
        source = source.replace('s3://', '')
        parts = source.split('/', 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ''
        return bucket, prefix
    
    def _get_modality_from_key(self, key: str) -> Optional[DataModality]:
        """
        Determine data modality from file extension
        
        Args:
            key: S3 object key (file path)
            
        Returns:
            DataModality or None
        """
        ext = Path(key).suffix.lower()
        return self.EXTENSION_MAP.get(ext)
    
    def _list_objects(self, bucket: str, prefix: str) -> List[Dict[str, Any]]:
        """
        List all objects in S3 bucket with prefix
        
        Args:
            bucket: S3 bucket name
            prefix: Key prefix to filter
            
        Returns:
            List of object metadata dictionaries
        """
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            
            objects = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        modality = self._get_modality_from_key(obj['Key'])
                        if modality and (not self.config.modalities or modality in self.config.modalities):
                            objects.append({
                                'key': obj['Key'],
                                'size': obj['Size'],
                                'last_modified': obj['LastModified'],
                                'modality': modality
                            })
            
            self.logger.info(f"Found {len(objects)} objects in {bucket}/{prefix}")
            return objects
        except ClientError as e:
            self.logger.error(f"Error listing objects: {e}")
            raise
    
    def _load_text(self, bucket: str, key: str) -> str:
        """Load text file from S3"""
        try:
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            return obj['Body'].read().decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error loading text {key}: {e}")
            raise
    
    def _load_image(self, bucket: str, key: str) -> Image.Image:
        """Load image file from S3"""
        try:
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            image_data = obj['Body'].read()
            return Image.open(io.BytesIO(image_data))
        except Exception as e:
            self.logger.error(f"Error loading image {key}: {e}")
            raise
    
    def _load_audio(self, bucket: str, key: str) -> bytes:
        """Load audio file from S3 (returns raw bytes)"""
        try:
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            return obj['Body'].read()
        except Exception as e:
            self.logger.error(f"Error loading audio {key}: {e}")
            raise
    
    def _load_video(self, bucket: str, key: str) -> bytes:
        """Load video file from S3 (returns raw bytes)"""
        try:
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            return obj['Body'].read()
        except Exception as e:
            self.logger.error(f"Error loading video {key}: {e}")
            raise
    
    def _load_single_object(self, bucket: str, obj_meta: Dict[str, Any]) -> DataSample:
        """
        Load a single object from S3
        
        Args:
            bucket: S3 bucket name
            obj_meta: Object metadata dictionary
            
        Returns:
            DataSample object
        """
        key = obj_meta['key']
        modality = obj_meta['modality']
        
        # Load data based on modality
        if modality == DataModality.TEXT:
            data = self._load_text(bucket, key)
        elif modality == DataModality.IMAGE:
            data = self._load_image(bucket, key)
        elif modality == DataModality.AUDIO:
            data = self._load_audio(bucket, key)
        elif modality == DataModality.VIDEO:
            data = self._load_video(bucket, key)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        
        return DataSample(
            id=key,
            modality=modality,
            data=data,
            metadata={
                'size': obj_meta['size'],
                'last_modified': str(obj_meta['last_modified']),
                'bucket': bucket,
                'extension': Path(key).suffix
            },
            source=f"s3://{bucket}/{key}"
        )
    
    def load(self, source: str, **kwargs) -> List[DataSample]:
        """
        Load all data from S3 source
        
        Args:
            source: S3 path (s3://bucket/prefix or bucket/prefix)
            **kwargs: Additional arguments
            
        Returns:
            List of DataSample objects
        """
        bucket, prefix = self._parse_s3_path(source)
        
        # List all objects
        objects = self._list_objects(bucket, prefix)
        
        # Apply max_samples limit
        if self.config.max_samples:
            objects = objects[:self.config.max_samples]
        
        # Load each object
        samples = []
        for i, obj_meta in enumerate(objects):
            try:
                sample = self._load_single_object(bucket, obj_meta)
                samples.append(sample)
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Loaded {i + 1}/{len(objects)} samples")
            except Exception as e:
                self.logger.warning(f"Failed to load {obj_meta['key']}: {e}")
        
        self.logger.info(f"Successfully loaded {len(samples)}/{len(objects)} samples")
        return samples
    
    def load_batch(self, source: str, **kwargs) -> Generator[List[DataSample], None, None]:
        """
        Load data in batches from S3
        
        Args:
            source: S3 path
            **kwargs: Additional arguments
            
        Yields:
            Batches of DataSample objects
        """
        bucket, prefix = self._parse_s3_path(source)
        objects = self._list_objects(bucket, prefix)
        
        # Apply max_samples limit
        if self.config.max_samples:
            objects = objects[:self.config.max_samples]
        
        # Yield batches
        batch = []
        for i, obj_meta in enumerate(objects):
            try:
                sample = self._load_single_object(bucket, obj_meta)
                batch.append(sample)
                
                if len(batch) >= self.config.batch_size:
                    yield batch
                    batch = []
            except Exception as e:
                self.logger.warning(f"Failed to load {obj_meta['key']}: {e}")
        
        # Yield remaining samples
        if batch:
            yield batch
    
    def get_metadata(self, source: str) -> Dict[str, Any]:
        """
        Get metadata about S3 data source
        
        Args:
            source: S3 path
            
        Returns:
            Dictionary with metadata
        """
        bucket, prefix = self._parse_s3_path(source)
        objects = self._list_objects(bucket, prefix)
        
        # Count by modality
        modality_counts = {}
        total_size = 0
        for obj in objects:
            modality_key = obj['modality'].value
            modality_counts[modality_key] = modality_counts.get(modality_key, 0) + 1
            total_size += obj['size']
        
        metadata = super().get_metadata(source)
        metadata.update({
            'bucket': bucket,
            'prefix': prefix,
            'total_objects': len(objects),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'modality_counts': modality_counts,
            'region': self.region
        })
        
        return metadata


# Example usage
if __name__ == "__main__":
    config = LoaderConfig(
        batch_size=16,
        max_samples=100,
        modalities=[DataModality.TEXT, DataModality.IMAGE]
    )
    
    loader = S3Loader(
        config=config,
        aws_access_key=os.getenv('AWS_ACCESS_KEY'),
        aws_secret_key=os.getenv('AWS_SECRET_KEY')
    )
    
    # Get metadata
    metadata = loader.get_metadata('my-bucket/training-data/')
    print(f"Metadata: {json.dumps(metadata, indent=2)}")
    
    # Load data in batches
    for batch in loader.load_batch('my-bucket/training-data/'):
        print(f"Batch size: {len(batch)}")
        for sample in batch[:2]:  # Print first 2
            print(f"  {sample}")
