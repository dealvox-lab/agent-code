"""
Local File System Data Loader
Load multi-modal data from local directories
"""

import os
from typing import List, Dict, Any, Generator, Optional
from pathlib import Path
from PIL import Image
import json

from .base_loader import BaseLoader, DataSample, DataModality, LoaderConfig


class LocalLoader(BaseLoader):
    """
    Load multi-modal data from local file system
    Supports recursive directory scanning
    """
    
    # File extension to modality mapping (same as S3Loader)
    EXTENSION_MAP = {
        # Text
        '.txt': DataModality.TEXT,
        '.json': DataModality.TEXT,
        '.csv': DataModality.TEXT,
        '.md': DataModality.TEXT,
        '.xml': DataModality.TEXT,
        '.html': DataModality.TEXT,
        
        # Images
        '.jpg': DataModality.IMAGE,
        '.jpeg': DataModality.IMAGE,
        '.png': DataModality.IMAGE,
        '.gif': DataModality.IMAGE,
        '.bmp': DataModality.IMAGE,
        '.webp': DataModality.IMAGE,
        '.tiff': DataModality.IMAGE,
        
        # Audio
        '.mp3': DataModality.AUDIO,
        '.wav': DataModality.AUDIO,
        '.flac': DataModality.AUDIO,
        '.ogg': DataModality.AUDIO,
        '.m4a': DataModality.AUDIO,
        '.aac': DataModality.AUDIO,
        
        # Video
        '.mp4': DataModality.VIDEO,
        '.avi': DataModality.VIDEO,
        '.mov': DataModality.VIDEO,
        '.mkv': DataModality.VIDEO,
        '.webm': DataModality.VIDEO,
        '.flv': DataModality.VIDEO,
    }
    
    def __init__(self, config: LoaderConfig, recursive: bool = True):
        """
        Initialize Local Loader
        
        Args:
            config: LoaderConfig object
            recursive: Whether to scan subdirectories recursively
        """
        super().__init__(config)
        self.recursive = recursive
    
    def validate_source(self, source: str) -> bool:
        """
        Validate local path exists
        
        Args:
            source: Local directory path
            
        Returns:
            True if path exists and is accessible
        """
        path = Path(source)
        is_valid = path.exists() and (path.is_dir() or path.is_file())
        
        if is_valid:
            self.logger.info(f"Validated local source: {source}")
        else:
            self.logger.error(f"Invalid local source: {source}")
        
        return is_valid
    
    def _get_modality_from_path(self, file_path: Path) -> Optional[DataModality]:
        """
        Determine data modality from file extension
        
        Args:
            file_path: Path object
            
        Returns:
            DataModality or None
        """
        ext = file_path.suffix.lower()
        return self.EXTENSION_MAP.get(ext)
    
    def _scan_directory(self, directory: str) -> List[Path]:
        """
        Scan directory for files
        
        Args:
            directory: Directory path
            
        Returns:
            List of file paths
        """
        path = Path(directory)
        
        if path.is_file():
            return [path]
        
        if self.recursive:
            files = [f for f in path.rglob('*') if f.is_file()]
        else:
            files = [f for f in path.glob('*') if f.is_file()]
        
        # Filter by modality
        filtered_files = []
        for file in files:
            modality = self._get_modality_from_path(file)
            if modality and (not self.config.modalities or modality in self.config.modalities):
                filtered_files.append(file)
        
        self.logger.info(f"Found {len(filtered_files)} files in {directory}")
        return filtered_files
    
    def _load_text(self, file_path: Path) -> str:
        """Load text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def _load_image(self, file_path: Path) -> Image.Image:
        """Load image file"""
        return Image.open(file_path)
    
    def _load_audio(self, file_path: Path) -> bytes:
        """Load audio file (returns raw bytes)"""
        with open(file_path, 'rb') as f:
            return f.read()
    
    def _load_video(self, file_path: Path) -> bytes:
        """Load video file (returns raw bytes)"""
        with open(file_path, 'rb') as f:
            return f.read()
    
    def _load_single_file(self, file_path: Path) -> DataSample:
        """
        Load a single file
        
        Args:
            file_path: Path to file
            
        Returns:
            DataSample object
        """
        modality = self._get_modality_from_path(file_path)
        
        if modality is None:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Load data based on modality
        if modality == DataModality.TEXT:
            data = self._load_text(file_path)
        elif modality == DataModality.IMAGE:
            data = self._load_image(file_path)
        elif modality == DataModality.AUDIO:
            data = self._load_audio(file_path)
        elif modality == DataModality.VIDEO:
            data = self._load_video(file_path)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        
        # Get file stats
        stats = file_path.stat()
        
        return DataSample(
            id=str(file_path),
            modality=modality,
            data=data,
            metadata={
                'filename': file_path.name,
                'extension': file_path.suffix,
                'size': stats.st_size,
                'created': stats.st_ctime,
                'modified': stats.st_mtime,
                'absolute_path': str(file_path.absolute())
            },
            source=str(file_path)
        )
    
    def load(self, source: str, **kwargs) -> List[DataSample]:
        """
        Load all data from local directory
        
        Args:
            source: Local directory or file path
            **kwargs: Additional arguments
            
        Returns:
            List of DataSample objects
        """
        files = self._scan_directory(source)
        
        # Apply max_samples limit
        if self.config.max_samples:
            files = files[:self.config.max_samples]
        
        # Load each file
        samples = []
        for i, file_path in enumerate(files):
            try:
                sample = self._load_single_file(file_path)
                samples.append(sample)
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Loaded {i + 1}/{len(files)} samples")
            except Exception as e:
                self.logger.warning(f"Failed to load {file_path}: {e}")
        
        self.logger.info(f"Successfully loaded {len(samples)}/{len(files)} samples")
        return self._filter_by_modality(samples)
    
    def load_batch(self, source: str, **kwargs) -> Generator[List[DataSample], None, None]:
        """
        Load data in batches from local directory
        
        Args:
            source: Local directory or file path
            **kwargs: Additional arguments
            
        Yields:
            Batches of DataSample objects
        """
        files = self._scan_directory(source)
        
        # Apply max_samples limit
        if self.config.max_samples:
            files = files[:self.config.max_samples]
        
        # Yield batches
        batch = []
        for i, file_path in enumerate(files):
            try:
                sample = self._load_single_file(file_path)
                batch.append(sample)
                
                if len(batch) >= self.config.batch_size:
                    yield self._filter_by_modality(batch)
                    batch = []
            except Exception as e:
                self.logger.warning(f"Failed to load {file_path}: {e}")
        
        # Yield remaining samples
        if batch:
            yield self._filter_by_modality(batch)
    
    def get_metadata(self, source: str) -> Dict[str, Any]:
        """
        Get metadata about local data source
        
        Args:
            source: Local directory path
            
        Returns:
            Dictionary with metadata
        """
        files = self._scan_directory(source)
        
        # Count by modality
        modality_counts = {}
        total_size = 0
        for file_path in files:
            modality = self._get_modality_from_path(file_path)
            if modality:
                modality_key = modality.value
                modality_counts[modality_key] = modality_counts.get(modality_key, 0) + 1
                total_size += file_path.stat().st_size
        
        metadata = super().get_metadata(source)
        metadata.update({
            'directory': source,
            'recursive': self.recursive,
            'total_files': len(files),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'modality_counts': modality_counts
        })
        
        return metadata
    
    def get_file_tree(self, source: str, max_depth: int = 3) -> Dict[str, Any]:
        """
        Get directory tree structure
        
        Args:
            source: Local directory path
            max_depth: Maximum depth to traverse
            
        Returns:
            Nested dictionary representing directory structure
        """
        def build_tree(path: Path, depth: int = 0) -> Dict[str, Any]:
            if depth > max_depth or not path.is_dir():
                return {}
            
            tree = {}
            for item in path.iterdir():
                if item.is_dir():
                    tree[item.name] = build_tree(item, depth + 1)
                else:
                    modality = self._get_modality_from_path(item)
                    if modality:
                        tree[item.name] = {
                            'type': 'file',
                            'modality': modality.value,
                            'size': item.stat().st_size
                        }
            return tree
        
        return build_tree(Path(source))


# Example usage
if __name__ == "__main__":
    config = LoaderConfig(
        batch_size=32,
        max_samples=100,
        modalities=[DataModality.TEXT, DataModality.IMAGE]
    )
    
    loader = LocalLoader(config=config, recursive=True)
    
    # Get metadata
    metadata = loader.get_metadata('./training_data')
    print(f"Metadata: {json.dumps(metadata, indent=2)}")
    
    # Get file tree
    tree = loader.get_file_tree('./training_data', max_depth=2)
    print(f"File tree: {json.dumps(tree, indent=2)}")
    
    # Load data in batches
    for i, batch in enumerate(loader.load_batch('./training_data')):
        print(f"Batch {i+1}: {len(batch)} samples")
        if i >= 2:  # Just show first 3 batches
            break
