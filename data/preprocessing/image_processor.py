"""
Image Processor for Computer Vision Preprocessing
Handles resizing, normalization, augmentation, and format conversion
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from PIL import Image
import io

from .base_processor import BaseProcessor, ProcessorConfig


@dataclass
class ImageProcessorConfig(ProcessorConfig):
    """Configuration for image preprocessing"""
    # Resize options
    resize: bool = True
    target_size: Tuple[int, int] = (224, 224)  # (width, height)
    resize_mode: str = 'bicubic'  # nearest, bilinear, bicubic, lanczos
    maintain_aspect_ratio: bool = False
    
    # Normalization options
    normalize_pixels: bool = True
    pixel_range: str = '0-1'  # '0-1' or '-1-1'
    mean: Optional[Tuple[float, float, float]] = None  # ImageNet: (0.485, 0.456, 0.406)
    std: Optional[Tuple[float, float, float]] = None   # ImageNet: (0.229, 0.224, 0.225)
    
    # Color options
    convert_to_rgb: bool = True
    grayscale: bool = False
    
    # Augmentation options (for training)
    augment: bool = False
    horizontal_flip: bool = False
    vertical_flip: bool = False
    rotation_range: int = 0  # degrees
    brightness_range: Optional[Tuple[float, float]] = None  # e.g., (0.8, 1.2)
    contrast_range: Optional[Tuple[float, float]] = None
    
    # Output options
    output_format: str = 'numpy'  # 'numpy', 'pil', 'tensor'
    dtype: str = 'float32'  # Data type for numpy arrays


class ImageProcessor(BaseProcessor):
    """
    Comprehensive image preprocessing for computer vision tasks
    Supports multiple formats, augmentation, and normalization schemes
    """
    
    # Resize mode mapping
    RESIZE_MODES = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS
    }
    
    def __init__(self, config: ImageProcessorConfig):
        """
        Initialize image processor
        
        Args:
            config: ImageProcessorConfig object
        """
        super().__init__(config)
        self.config: ImageProcessorConfig = config
        
        # Validate resize mode
        if config.resize_mode not in self.RESIZE_MODES:
            raise ValueError(f"Invalid resize mode: {config.resize_mode}")
    
    def _get_modality(self) -> str:
        return "image"
    
    def validate(self, data: Any) -> bool:
        """
        Validate image input
        
        Args:
            data: Input data (PIL Image, numpy array, or bytes)
            
        Returns:
            True if valid, False otherwise
        """
        try:
            img = self._to_pil(data)
            
            if img.size[0] == 0 or img.size[1] == 0:
                self.logger.warning("Image has zero dimension")
                return False
            
            return True
        except Exception as e:
            self.logger.warning(f"Invalid image data: {e}")
            return False
    
    def process(self, data: Any, **kwargs) -> Union[np.ndarray, Image.Image]:
        """
        Process a single image
        
        Args:
            data: Input image (PIL Image, numpy array, or bytes)
            **kwargs: Additional parameters
            
        Returns:
            Processed image in configured format
        """
        # Convert to PIL Image
        img = self._to_pil(data)
        
        # Step 1: Convert color mode
        if self.config.convert_to_rgb and img.mode != 'RGB':
            img = img.convert('RGB')
        elif self.config.grayscale:
            img = img.convert('L')
        
        # Step 2: Resize
        if self.config.resize:
            img = self._resize(img)
        
        # Step 3: Augmentation (if enabled)
        if self.config.augment:
            img = self._augment(img)
        
        # Step 4: Convert to numpy for normalization
        img_array = np.array(img, dtype=self.config.dtype)
        
        # Step 5: Normalize pixels
        if self.config.normalize_pixels:
            img_array = self._normalize(img_array)
        
        # Step 6: Convert to output format
        return self._to_output_format(img_array, img)
    
    def process_batch(self, data_batch: List[Any], **kwargs) -> List[Union[np.ndarray, Image.Image]]:
        """
        Process a batch of images
        
        Args:
            data_batch: List of input images
            **kwargs: Additional parameters
            
        Returns:
            List of processed images
        """
        return [self.process(img, **kwargs) for img in data_batch]
    
    def _to_pil(self, data: Any) -> Image.Image:
        """
        Convert input data to PIL Image
        
        Args:
            data: Input data (PIL Image, numpy array, or bytes)
            
        Returns:
            PIL Image
        """
        if isinstance(data, Image.Image):
            return data
        elif isinstance(data, np.ndarray):
            return Image.fromarray(data)
        elif isinstance(data, bytes):
            return Image.open(io.BytesIO(data))
        else:
            raise ValueError(f"Unsupported input type: {type(data)}")
    
    def _resize(self, img: Image.Image) -> Image.Image:
        """
        Resize image to target size
        
        Args:
            img: PIL Image
            
        Returns:
            Resized PIL Image
        """
        target_width, target_height = self.config.target_size
        resize_mode = self.RESIZE_MODES[self.config.resize_mode]
        
        if self.config.maintain_aspect_ratio:
            # Calculate aspect ratio
            img_width, img_height = img.size
            aspect_ratio = img_width / img_height
            target_aspect = target_width / target_height
            
            if aspect_ratio > target_aspect:
                # Width is the limiting factor
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                # Height is the limiting factor
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
            
            # Resize
            img = img.resize((new_width, new_height), resize_mode)
            
            # Pad to target size
            padded = Image.new(img.mode, (target_width, target_height), color=0)
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            padded.paste(img, (paste_x, paste_y))
            return padded
        else:
            # Direct resize
            return img.resize((target_width, target_height), resize_mode)
    
    def _normalize(self, img_array: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values
        
        Args:
            img_array: Image as numpy array
            
        Returns:
            Normalized image array
        """
        # Scale to 0-1 or -1-1
        if self.config.pixel_range == '0-1':
            img_array = img_array / 255.0
        elif self.config.pixel_range == '-1-1':
            img_array = (img_array / 127.5) - 1.0
        else:
            raise ValueError(f"Invalid pixel range: {self.config.pixel_range}")
        
        # Apply mean and std normalization
        if self.config.mean is not None and self.config.std is not None:
            mean = np.array(self.config.mean, dtype=self.config.dtype)
            std = np.array(self.config.std, dtype=self.config.dtype)
            
            # Reshape for broadcasting
            if img_array.ndim == 3:  # (H, W, C)
                mean = mean.reshape(1, 1, -1)
                std = std.reshape(1, 1, -1)
            
            img_array = (img_array - mean) / std
        
        return img_array
    
    def _augment(self, img: Image.Image) -> Image.Image:
        """
        Apply data augmentation
        
        Args:
            img: PIL Image
            
        Returns:
            Augmented PIL Image
        """
        import random
        
        # Horizontal flip
        if self.config.horizontal_flip and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Vertical flip
        if self.config.vertical_flip and random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Rotation
        if self.config.rotation_range > 0:
            angle = random.uniform(-self.config.rotation_range, self.config.rotation_range)
            img = img.rotate(angle, resample=Image.BICUBIC)
        
        # Brightness
        if self.config.brightness_range:
            from PIL import ImageEnhance
            factor = random.uniform(*self.config.brightness_range)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
        
        # Contrast
        if self.config.contrast_range:
            from PIL import ImageEnhance
            factor = random.uniform(*self.config.contrast_range)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)
        
        return img
    
    def _to_output_format(self, img_array: np.ndarray, pil_img: Image.Image) -> Union[np.ndarray, Image.Image]:
        """
        Convert to output format
        
        Args:
            img_array: Image as numpy array
            pil_img: Original PIL Image
            
        Returns:
            Image in configured output format
        """
        if self.config.output_format == 'numpy':
            return img_array
        elif self.config.output_format == 'pil':
            # Convert back to PIL (denormalize if needed)
            if self.config.normalize_pixels:
                if self.config.pixel_range == '0-1':
                    img_array = (img_array * 255).astype(np.uint8)
                elif self.config.pixel_range == '-1-1':
                    img_array = ((img_array + 1.0) * 127.5).astype(np.uint8)
            return Image.fromarray(img_array)
        elif self.config.output_format == 'tensor':
            # For PyTorch: (H, W, C) -> (C, H, W)
            if img_array.ndim == 3:
                img_array = np.transpose(img_array, (2, 0, 1))
            return img_array
        else:
            raise ValueError(f"Invalid output format: {self.config.output_format}")
    
    def get_statistics(self, images: List[Any]) -> Dict[str, Any]:
        """
        Get statistics about image data
        
        Args:
            images: List of images
            
        Returns:
            Dictionary with statistics
        """
        sizes = []
        modes = []
        
        for img_data in images:
            try:
                img = self._to_pil(img_data)
                sizes.append(img.size)
                modes.append(img.mode)
            except Exception as e:
                self.logger.warning(f"Failed to analyze image: {e}")
        
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        
        return {
            'total_images': len(images),
            'avg_width': sum(widths) / len(widths) if widths else 0,
            'avg_height': sum(heights) / len(heights) if heights else 0,
            'min_width': min(widths) if widths else 0,
            'max_width': max(widths) if widths else 0,
            'min_height': min(heights) if heights else 0,
            'max_height': max(heights) if heights else 0,
            'color_modes': list(set(modes))
        }


# Example usage
if __name__ == "__main__":
    # Create processor with ImageNet normalization
    config = ImageProcessorConfig(
        resize=True,
        target_size=(224, 224),
        normalize_pixels=True,
        pixel_range='0-1',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        output_format='numpy'
    )
    
    processor = ImageProcessor(config)
    
    # Create a sample image
    sample_img = Image.new('RGB', (512, 512), color=(255, 128, 0))
    
    # Process single image
    processed = processor.process(sample_img)
    print(f"Original shape: {sample_img.size}")
    print(f"Processed shape: {processed.shape}")
    print(f"Processed dtype: {processed.dtype}")
    print(f"Processed range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Get statistics
    stats = processor.get_statistics([sample_img])
    print(f"\nStatistics: {stats}")
