"""
Video Processor for Video Preprocessing
Handles frame extraction, temporal processing, and video feature extraction
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import warnings

from .base_processor import BaseProcessor, ProcessorConfig

# Optional imports - will be checked at runtime
try:
    import cv2
    VIDEO_LIBS_AVAILABLE = True
except ImportError:
    VIDEO_LIBS_AVAILABLE = False
    warnings.warn(
        "Video libraries not available. Install with: pip install opencv-python"
    )


@dataclass
class VideoProcessorConfig(ProcessorConfig):
    """Configuration for video preprocessing"""
    # Frame extraction options
    fps: Optional[int] = None  # Target FPS (None = use original)
    max_frames: Optional[int] = None  # Maximum number of frames to extract
    frame_sampling: str = 'uniform'  # 'uniform', 'random', 'all'
    
    # Frame processing options
    resize_frames: bool = True
    target_size: Tuple[int, int] = (224, 224)  # (width, height)
    maintain_aspect_ratio: bool = False
    
    # Color options
    convert_to_rgb: bool = True
    grayscale: bool = False
    
    # Normalization options
    normalize_pixels: bool = True
    pixel_range: str = '0-1'  # '0-1' or '-1-1'
    mean: Optional[Tuple[float, float, float]] = None
    std: Optional[Tuple[float, float, float]] = None
    
    # Temporal processing
    temporal_sampling: str = 'all'  # 'all', 'keyframes', 'uniform'
    clip_duration: Optional[float] = None  # Duration in seconds
    
    # Output options
    output_format: str = 'numpy'  # 'numpy', 'tensor', 'frames_list'
    stack_frames: bool = True  # Stack frames into single array
    dtype: str = 'float32'


class VideoProcessor(BaseProcessor):
    """
    Comprehensive video preprocessing for video understanding tasks
    Supports frame extraction, temporal processing, and various output formats
    """
    
    def __init__(self, config: VideoProcessorConfig):
        """
        Initialize video processor
        
        Args:
            config: VideoProcessorConfig object
        """
        if not VIDEO_LIBS_AVAILABLE:
            raise ImportError(
                "Video processing requires opencv. "
                "Install with: pip install opencv-python"
            )
        
        super().__init__(config)
        self.config: VideoProcessorConfig = config
    
    def _get_modality(self) -> str:
        return "video"
    
    def validate(self, data: Any) -> bool:
        """
        Validate video input
        
        Args:
            data: Input data (file path or video capture)
            
        Returns:
            True if valid, False otherwise
        """
        try:
            cap = self._get_video_capture(data)
            
            if not cap.isOpened():
                self.logger.warning("Failed to open video")
                return False
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                self.logger.warning("Video has no frames")
                cap.release()
                return False
            
            cap.release()
            return True
        except Exception as e:
            self.logger.warning(f"Invalid video data: {e}")
            return False
    
    def process(self, data: Any, **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Process a single video
        
        Args:
            data: Input video (file path or bytes)
            **kwargs: Additional parameters
            
        Returns:
            Processed video frames
        """
        # Step 1: Extract frames
        frames = self._extract_frames(data)
        
        if len(frames) == 0:
            raise ValueError("No frames extracted from video")
        
        # Step 2: Process each frame
        processed_frames = []
        for frame in frames:
            processed_frame = self._process_frame(frame)
            processed_frames.append(processed_frame)
        
        # Step 3: Convert to output format
        return self._to_output_format(processed_frames)
    
    def process_batch(self, data_batch: List[Any], **kwargs) -> List[Union[np.ndarray, List[np.ndarray]]]:
        """
        Process a batch of videos
        
        Args:
            data_batch: List of input videos
            **kwargs: Additional parameters
            
        Returns:
            List of processed videos
        """
        return [self.process(video, **kwargs) for video in data_batch]
    
    def _get_video_capture(self, data: Any) -> cv2.VideoCapture:
        """
        Get video capture object from input data
        
        Args:
            data: Input data (file path or bytes)
            
        Returns:
            cv2.VideoCapture object
        """
        if isinstance(data, str):
            # File path
            return cv2.VideoCapture(data)
        elif isinstance(data, bytes):
            # Bytes - save temporarily and load
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            
            cap = cv2.VideoCapture(tmp_path)
            os.unlink(tmp_path)
            return cap
        else:
            raise ValueError(f"Unsupported video input type: {type(data)}")
    
    def _extract_frames(self, data: Any) -> List[np.ndarray]:
        """
        Extract frames from video
        
        Args:
            data: Input video
            
        Returns:
            List of frame arrays
        """
        cap = self._get_video_capture(data)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Determine which frames to extract
        frame_indices = self._get_frame_indices(total_frames, original_fps)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frames.append(frame)
            else:
                self.logger.warning(f"Failed to read frame {idx}")
        
        cap.release()
        
        self.logger.debug(f"Extracted {len(frames)}/{total_frames} frames")
        return frames
    
    def _get_frame_indices(self, total_frames: int, fps: float) -> List[int]:
        """
        Determine which frame indices to extract
        
        Args:
            total_frames: Total number of frames in video
            fps: Frames per second
            
        Returns:
            List of frame indices to extract
        """
        # Apply clip duration if specified
        if self.config.clip_duration:
            max_frames_for_duration = int(self.config.clip_duration * fps)
            total_frames = min(total_frames, max_frames_for_duration)
        
        # Apply max_frames limit
        if self.config.max_frames:
            num_frames = min(total_frames, self.config.max_frames)
        else:
            num_frames = total_frames
        
        # Sample frames based on strategy
        if self.config.frame_sampling == 'all':
            return list(range(total_frames))
        
        elif self.config.frame_sampling == 'uniform':
            # Uniformly sample frames
            if num_frames >= total_frames:
                return list(range(total_frames))
            else:
                step = total_frames / num_frames
                indices = [int(i * step) for i in range(num_frames)]
                return indices
        
        elif self.config.frame_sampling == 'random':
            # Randomly sample frames
            import random
            return sorted(random.sample(range(total_frames), num_frames))
        
        else:
            raise ValueError(f"Unknown frame sampling: {self.config.frame_sampling}")
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame
        
        Args:
            frame: Frame array (BGR format from OpenCV)
            
        Returns:
            Processed frame
        """
        # Step 1: Convert color space
        if self.config.convert_to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self.config.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Resize
        if self.config.resize_frames:
            frame = self._resize_frame(frame)
        
        # Step 3: Convert to float and normalize
        frame = frame.astype(self.config.dtype)
        
        if self.config.normalize_pixels:
            frame = self._normalize_frame(frame)
        
        return frame
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame to target size
        
        Args:
            frame: Input frame
            
        Returns:
            Resized frame
        """
        target_width, target_height = self.config.target_size
        
        if self.config.maintain_aspect_ratio:
            # Calculate aspect ratio
            h, w = frame.shape[:2]
            aspect_ratio = w / h
            target_aspect = target_width / target_height
            
            if aspect_ratio > target_aspect:
                # Width is limiting
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                # Height is limiting
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
            
            # Resize
            resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Pad to target size
            if len(frame.shape) == 3:
                padded = np.zeros((target_height, target_width, frame.shape[2]), dtype=frame.dtype)
            else:
                padded = np.zeros((target_height, target_width), dtype=frame.dtype)
            
            paste_y = (target_height - new_height) // 2
            paste_x = (target_width - new_width) // 2
            
            if len(frame.shape) == 3:
                padded[paste_y:paste_y+new_height, paste_x:paste_x+new_width, :] = resized
            else:
                padded[paste_y:paste_y+new_height, paste_x:paste_x+new_width] = resized
            
            return padded
        else:
            # Direct resize
            return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame pixel values
        
        Args:
            frame: Input frame
            
        Returns:
            Normalized frame
        """
        # Scale to 0-1 or -1-1
        if self.config.pixel_range == '0-1':
            frame = frame / 255.0
        elif self.config.pixel_range == '-1-1':
            frame = (frame / 127.5) - 1.0
        
        # Apply mean and std normalization
        if self.config.mean is not None and self.config.std is not None:
            mean = np.array(self.config.mean, dtype=self.config.dtype)
            std = np.array(self.config.std, dtype=self.config.dtype)
            
            if frame.ndim == 3:  # (H, W, C)
                mean = mean.reshape(1, 1, -1)
                std = std.reshape(1, 1, -1)
            
            frame = (frame - mean) / std
        
        return frame
    
    def _to_output_format(self, frames: List[np.ndarray]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Convert frames to output format
        
        Args:
            frames: List of processed frames
            
        Returns:
            Frames in configured output format
        """
        if not self.config.stack_frames:
            return frames
        
        # Stack frames
        stacked = np.stack(frames, axis=0)  # (T, H, W, C) or (T, H, W)
        
        if self.config.output_format == 'numpy':
            return stacked
        
        elif self.config.output_format == 'tensor':
            # For PyTorch: (T, H, W, C) -> (C, T, H, W) or (T, H, W) -> (1, T, H, W)
            if stacked.ndim == 4:  # (T, H, W, C)
                stacked = np.transpose(stacked, (3, 0, 1, 2))  # (C, T, H, W)
            elif stacked.ndim == 3:  # (T, H, W)
                stacked = stacked[np.newaxis, :, :, :]  # (1, T, H, W)
            return stacked
        
        elif self.config.output_format == 'frames_list':
            return frames
        
        else:
            raise ValueError(f"Invalid output format: {self.config.output_format}")
    
    def extract_keyframes(self, data: Any, threshold: float = 0.3) -> List[np.ndarray]:
        """
        Extract keyframes using scene change detection
        
        Args:
            data: Input video
            threshold: Scene change threshold (0-1)
            
        Returns:
            List of keyframes
        """
        cap = self._get_video_capture(data)
        
        frames = []
        prev_frame = None
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale for comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is None:
                frames.append(frame)
            else:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, gray)
                diff_ratio = np.sum(diff) / (gray.shape[0] * gray.shape[1] * 255.0)
                
                if diff_ratio > threshold:
                    frames.append(frame)
            
            prev_frame = gray
            frame_idx += 1
        
        cap.release()
        
        self.logger.info(f"Extracted {len(frames)} keyframes from {frame_idx} total frames")
        return frames
    
    def get_statistics(self, videos: List[Any]) -> Dict[str, Any]:
        """
        Get statistics about video data
        
        Args:
            videos: List of videos
            
        Returns:
            Dictionary with statistics
        """
        durations = []
        frame_counts = []
        fps_values = []
        resolutions = []
        
        for video_data in videos:
            try:
                cap = self._get_video_capture(video_data)
                
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                duration = frame_count / fps if fps > 0 else 0
                
                durations.append(duration)
                frame_counts.append(frame_count)
                fps_values.append(fps)
                resolutions.append((width, height))
                
                cap.release()
            except Exception as e:
                self.logger.warning(f"Failed to analyze video: {e}")
        
        return {
            'total_videos': len(videos),
            'avg_duration': sum(durations) / len(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'avg_frame_count': sum(frame_counts) / len(frame_counts) if frame_counts else 0,
            'avg_fps': sum(fps_values) / len(fps_values) if fps_values else 0,
            'resolutions': list(set(resolutions))
        }


# Example usage
if __name__ == "__main__":
    # Create processor for action recognition
    config = VideoProcessorConfig(
        max_frames=16,
        frame_sampling='uniform',
        resize_frames=True,
        target_size=(224, 224),
        normalize_pixels=True,
        pixel_range='0-1',
        output_format='tensor',
        stack_frames=True
    )
    
    processor = VideoProcessor(config)
    
    # Note: This example requires an actual video file
    # video_path = 'sample_video.mp4'
    # processed = processor.process(video_path)
    # print(f"Processed video shape: {processed.shape}")
    # print(f"Output format: {config.output_format}")
    
    print("Video processor initialized successfully")
    print(f"Config: {config}")
