"""
Multi-Modal Data Loader Orchestrator
Coordinates loading from multiple sources and modalities
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

from .base_loader import BaseLoader, DataSample, DataModality, LoaderConfig
from .s3_loader import S3Loader
from .local_loader import LocalLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Configuration for a single data source"""
    name: str
    type: str  # 's3' or 'local'
    path: str
    modalities: Optional[List[DataModality]] = None
    weight: float = 1.0  # For sampling from multiple sources
    
    def __repr__(self):
        return f"DataSource(name={self.name}, type={self.type}, path={self.path})"


class MultiModalLoader:
    """
    Orchestrates loading from multiple data sources
    Combines S3, local, and other loaders
    Handles multi-modal data alignment and balancing
    """
    
    def __init__(self, config: LoaderConfig, aws_credentials: Optional[Dict[str, str]] = None):
        """
        Initialize Multi-Modal Loader
        
        Args:
            config: LoaderConfig object
            aws_credentials: Dict with 'access_key', 'secret_key', 'region' for S3
        """
        self.config = config
        self.aws_credentials = aws_credentials
        self.sources: List[DataSource] = []
        self.loaders: Dict[str, BaseLoader] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_source(self, source: DataSource) -> None:
        """
        Add a data source to load from
        
        Args:
            source: DataSource configuration
        """
        self.sources.append(source)
        self.logger.info(f"Added data source: {source.name} ({source.type})")
    
    def add_s3_source(self, name: str, s3_path: str, 
                      modalities: Optional[List[DataModality]] = None,
                      weight: float = 1.0) -> None:
        """
        Convenience method to add S3 source
        
        Args:
            name: Source name
            s3_path: S3 path (s3://bucket/prefix or bucket/prefix)
            modalities: List of modalities to load
            weight: Sampling weight
        """
        source = DataSource(
            name=name,
            type='s3',
            path=s3_path,
            modalities=modalities,
            weight=weight
        )
        self.add_source(source)
    
    def add_local_source(self, name: str, local_path: str,
                        modalities: Optional[List[DataModality]] = None,
                        weight: float = 1.0) -> None:
        """
        Convenience method to add local source
        
        Args:
            name: Source name
            local_path: Local directory path
            modalities: List of modalities to load
            weight: Sampling weight
        """
        source = DataSource(
            name=name,
            type='local',
            path=local_path,
            modalities=modalities,
            weight=weight
        )
        self.add_source(source)
    
    def _get_loader(self, source: DataSource) -> BaseLoader:
        """
        Get or create loader for a data source
        
        Args:
            source: DataSource configuration
            
        Returns:
            Appropriate loader instance
        """
        # Check cache
        if source.name in self.loaders:
            return self.loaders[source.name]
        
        # Create loader based on type
        if source.type == 's3':
            if not self.aws_credentials:
                raise ValueError("AWS credentials required for S3 sources")
            
            loader = S3Loader(
                config=LoaderConfig(
                    batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers,
                    modalities=source.modalities or self.config.modalities,
                    max_samples=self.config.max_samples
                ),
                aws_access_key=self.aws_credentials['access_key'],
                aws_secret_key=self.aws_credentials['secret_key'],
                region=self.aws_credentials.get('region', 'us-east-1')
            )
        elif source.type == 'local':
            loader = LocalLoader(
                config=LoaderConfig(
                    batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers,
                    modalities=source.modalities or self.config.modalities,
                    max_samples=self.config.max_samples
                ),
                recursive=True
            )
        else:
            raise ValueError(f"Unsupported source type: {source.type}")
        
        # Cache loader
        self.loaders[source.name] = loader
        return loader
    
    def validate_all_sources(self) -> Dict[str, bool]:
        """
        Validate all configured data sources
        
        Returns:
            Dictionary mapping source names to validation status
        """
        results = {}
        for source in self.sources:
            try:
                loader = self._get_loader(source)
                is_valid = loader.validate_source(source.path)
                results[source.name] = is_valid
            except Exception as e:
                self.logger.error(f"Error validating {source.name}: {e}")
                results[source.name] = False
        
        valid_count = sum(results.values())
        self.logger.info(f"Validated {valid_count}/{len(results)} sources")
        return results
    
    def load_all(self) -> Dict[str, List[DataSample]]:
        """
        Load data from all sources
        
        Returns:
            Dictionary mapping source names to their data samples
        """
        all_data = {}
        
        for source in self.sources:
            try:
                loader = self._get_loader(source)
                samples = loader.load(source.path)
                all_data[source.name] = samples
                self.logger.info(f"Loaded {len(samples)} samples from {source.name}")
            except Exception as e:
                self.logger.error(f"Error loading from {source.name}: {e}")
                all_data[source.name] = []
        
        return all_data
    
    def load_combined(self, balance_by_modality: bool = False) -> List[DataSample]:
        """
        Load and combine data from all sources
        
        Args:
            balance_by_modality: Whether to balance samples across modalities
            
        Returns:
            Combined list of DataSample objects
        """
        all_data = self.load_all()
        combined = []
        
        for source_name, samples in all_data.items():
            combined.extend(samples)
        
        if balance_by_modality and combined:
            combined = self._balance_modalities(combined)
        
        self.logger.info(f"Combined total: {len(combined)} samples")
        return combined
    
    def _balance_modalities(self, samples: List[DataSample]) -> List[DataSample]:
        """
        Balance samples across modalities
        
        Args:
            samples: List of DataSample objects
            
        Returns:
            Balanced list of samples
        """
        # Group by modality
        by_modality = {}
        for sample in samples:
            modality_key = sample.modality.value
            if modality_key not in by_modality:
                by_modality[modality_key] = []
            by_modality[modality_key].append(sample)
        
        # Find minimum count
        min_count = min(len(samples) for samples in by_modality.values())
        
        # Balance
        balanced = []
        for modality, modality_samples in by_modality.items():
            balanced.extend(modality_samples[:min_count])
        
        self.logger.info(f"Balanced from {len(samples)} to {len(balanced)} samples")
        return balanced
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about all data sources
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_sources': len(self.sources),
            'sources': {},
            'combined_stats': {
                'total_samples': 0,
                'modality_counts': {},
                'total_size_mb': 0
            }
        }
        
        for source in self.sources:
            try:
                loader = self._get_loader(source)
                metadata = loader.get_metadata(source.path)
                stats['sources'][source.name] = metadata
                
                # Update combined stats
                if 'total_files' in metadata:
                    stats['combined_stats']['total_samples'] += metadata.get('total_files', 0)
                elif 'total_objects' in metadata:
                    stats['combined_stats']['total_samples'] += metadata.get('total_objects', 0)
                
                stats['combined_stats']['total_size_mb'] += metadata.get('total_size_mb', 0)
                
                # Merge modality counts
                for modality, count in metadata.get('modality_counts', {}).items():
                    if modality not in stats['combined_stats']['modality_counts']:
                        stats['combined_stats']['modality_counts'][modality] = 0
                    stats['combined_stats']['modality_counts'][modality] += count
                    
            except Exception as e:
                self.logger.error(f"Error getting stats for {source.name}: {e}")
                stats['sources'][source.name] = {'error': str(e)}
        
        return stats
    
    def get_sample_distribution(self) -> Dict[str, Dict[str, int]]:
        """
        Get sample count distribution across sources and modalities
        
        Returns:
            Nested dictionary with distribution info
        """
        distribution = {}
        
        for source in self.sources:
            try:
                loader = self._get_loader(source)
                metadata = loader.get_metadata(source.path)
                distribution[source.name] = metadata.get('modality_counts', {})
            except Exception as e:
                self.logger.error(f"Error getting distribution for {source.name}: {e}")
                distribution[source.name] = {}
        
        return distribution


# Example usage
if __name__ == "__main__":
    import os
    import json
    
    # Initialize orchestrator
    config = LoaderConfig(
        batch_size=32,
        modalities=[DataModality.TEXT, DataModality.IMAGE],
        max_samples=1000
    )
    
    orchestrator = MultiModalLoader(
        config=config,
        aws_credentials={
            'access_key': os.getenv('AWS_ACCESS_KEY'),
            'secret_key': os.getenv('AWS_SECRET_KEY'),
            'region': 'us-east-1'
        }
    )
    
    # Add data sources
    orchestrator.add_local_source(
        name='local_images',
        local_path='./data/images',
        modalities=[DataModality.IMAGE],
        weight=1.0
    )
    
    orchestrator.add_local_source(
        name='local_text',
        local_path='./data/text',
        modalities=[DataModality.TEXT],
        weight=1.0
    )
    
    orchestrator.add_s3_source(
        name='s3_training_data',
        s3_path='my-bucket/training-data/',
        weight=2.0
    )
    
    # Validate sources
    validation = orchestrator.validate_all_sources()
    print(f"Validation: {validation}")
    
    # Get statistics
    stats = orchestrator.get_statistics()
    print(f"Statistics:\n{json.dumps(stats, indent=2)}")
    
    # Load combined data
    samples = orchestrator.load_combined(balance_by_modality=True)
    print(f"Loaded {len(samples)} samples total")
