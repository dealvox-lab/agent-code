"""
Audio Processor for Audio/Speech Preprocessing
Handles resampling, normalization, feature extraction (spectrograms, MFCC, mel-spectrograms)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import warnings

from .base_processor import BaseProcessor, ProcessorConfig

# Optional imports - will be checked at runtime
try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    warnings.warn(
        "Audio libraries not available. Install with: pip install librosa soundfile"
    )


@dataclass
class AudioProcessorConfig(ProcessorConfig):
    """Configuration for audio preprocessing"""
    # Loading options
    sample_rate: int = 16000  # Target sample rate in Hz
    mono: bool = True  # Convert to mono
    
    # Duration options
    duration: Optional[float] = None  # Target duration in seconds
    trim_silence: bool = True
    silence_threshold: float = 0.01  # Amplitude threshold for silence
    
    # Normalization options
    normalize_amplitude: bool = True
    normalization_type: str = 'peak'  # 'peak' or 'rms'
    target_level: float = 1.0  # Target amplitude level
    
    # Resampling options
    resample_quality: str = 'kaiser_best'  # 'kaiser_best', 'kaiser_fast', 'scipy'
    
    # Feature extraction
    extract_features: bool = True
    feature_type: str = 'mel_spectrogram'  # 'spectrogram', 'mel_spectrogram', 'mfcc', 'raw'
    
    # Spectrogram options
    n_fft: int = 2048  # FFT window size
    hop_length: int = 512  # Number of samples between frames
    win_length: Optional[int] = None  # Window length (defaults to n_fft)
    window: str = 'hann'  # Window function
    
    # Mel-spectrogram options
    n_mels: int = 128  # Number of mel bands
    fmin: float = 0.0  # Minimum frequency
    fmax: Optional[float] = None  # Maximum frequency (defaults to sample_rate/2)
    
    # MFCC options
    n_mfcc: int = 40  # Number of MFCCs
    dct_type: int = 2  # DCT type
    
    # Augmentation options (for training)
    augment: bool = False
    time_stretch_range: Optional[Tuple[float, float]] = None  # e.g., (0.8, 1.2)
    pitch_shift_range: Optional[Tuple[int, int]] = None  # semitones, e.g., (-2, 2)
    add_noise: bool = False
    noise_factor: float = 0.005
    
    # Output options
    output_format: str = 'numpy'  # 'numpy' or 'tensor'
    dtype: str = 'float32'


class AudioProcessor(BaseProcessor):
    """
    Comprehensive audio preprocessing for speech and audio tasks
    Supports multiple feature types and augmentation techniques
    """
    
    def __init__(self, config: AudioProcessorConfig):
        """
        Initialize audio processor
        
        Args:
            config: AudioProcessorConfig object
        """
        if not AUDIO_LIBS_AVAILABLE:
            raise ImportError(
                "Audio processing requires librosa and soundfile. "
                "Install with: pip install librosa soundfile"
            )
        
        super().__init__(config)
        self.config: AudioProcessorConfig = config
        
        # Set default fmax if not specified
        if self.config.fmax is None:
            self.config.fmax = self.config.sample_rate / 2.0
    
    def _get_modality(self) -> str:
        return "audio"
    
    def validate(self, data: Any) -> bool:
        """
        Validate audio input
        
        Args:
            data: Input data (file path, bytes, or numpy array)
            
        Returns:
            True if valid, False otherwise
        """
        try:
            audio, sr = self._load_audio(data)
            
            if len(audio) == 0:
                self.logger.warning("Empty audio array")
                return False
            
            if sr <= 0:
                self.logger.warning(f"Invalid sample rate: {sr}")
                return False
            
            return True
        except Exception as e:
            self.logger.warning(f"Invalid audio data: {e}")
            return False
    
    def process(self, data: Any, **kwargs) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Process a single audio sample
        
        Args:
            data: Input audio (file path, bytes, or numpy array)
            **kwargs: Additional parameters
            
        Returns:
            Processed audio features or waveform
        """
        # Step 1: Load audio
        audio, original_sr = self._load_audio(data)
        
        # Step 2: Resample if needed
        if original_sr != self.config.sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=original_sr,
                target_sr=self.config.sample_rate,
                res_type=self.config.resample_quality
            )
        
        # Step 3: Convert to mono if needed
        if self.config.mono and audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        # Step 4: Trim silence
        if self.config.trim_silence:
            audio = self._trim_silence(audio)
        
        # Step 5: Adjust duration
        if self.config.duration:
            audio = self._adjust_duration(audio)
        
        # Step 6: Normalize amplitude
        if self.config.normalize_amplitude:
            audio = self._normalize_amplitude(audio)
        
        # Step 7: Augmentation (if enabled)
        if self.config.augment:
            audio = self._augment(audio)
        
        # Step 8: Extract features
        if self.config.extract_features:
            features = self._extract_features(audio)
            return self._to_output_format(features)
        
        # Return raw waveform
        return self._to_output_format(audio)
    
    def process_batch(self, data_batch: List[Any], **kwargs) -> List[Union[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Process a batch of audio samples
        
        Args:
            data_batch: List of input audio samples
            **kwargs: Additional parameters
            
        Returns:
            List of processed audio features or waveforms
        """
        return [self.process(audio, **kwargs) for audio in data_batch]
    
    def _load_audio(self, data: Any) -> Tuple[np.ndarray, int]:
        """
        Load audio from various input formats
        
        Args:
            data: Input data (file path, bytes, or numpy array)
            
        Returns:
            Tuple of (audio array, sample rate)
        """
        if isinstance(data, str):
            # File path
            audio, sr = librosa.load(data, sr=None, mono=False)
            return audio, sr
        
        elif isinstance(data, bytes):
            # Bytes data
            import io
            audio, sr = sf.read(io.BytesIO(data))
            # librosa format: samples x channels -> channels x samples
            if audio.ndim > 1:
                audio = audio.T
            return audio, sr
        
        elif isinstance(data, np.ndarray):
            # Already loaded audio (assume target sample rate)
            return data, self.config.sample_rate
        
        elif isinstance(data, tuple) and len(data) == 2:
            # Tuple of (audio, sample_rate)
            return data[0], data[1]
        
        else:
            raise ValueError(f"Unsupported audio input type: {type(data)}")
    
    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """
        Trim leading and trailing silence
        
        Args:
            audio: Audio waveform
            
        Returns:
            Trimmed audio
        """
        trimmed, _ = librosa.effects.trim(
            audio,
            top_db=-librosa.amplitude_to_db(self.config.silence_threshold)
        )
        return trimmed
    
    def _adjust_duration(self, audio: np.ndarray) -> np.ndarray:
        """
        Adjust audio to target duration
        
        Args:
            audio: Audio waveform
            
        Returns:
            Duration-adjusted audio
        """
        target_length = int(self.config.duration * self.config.sample_rate)
        current_length = len(audio)
        
        if current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            audio = np.pad(audio, (0, padding), mode='constant')
        elif current_length > target_length:
            # Truncate
            audio = audio[:target_length]
        
        return audio
    
    def _normalize_amplitude(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio amplitude
        
        Args:
            audio: Audio waveform
            
        Returns:
            Normalized audio
        """
        if self.config.normalization_type == 'peak':
            # Peak normalization
            peak = np.abs(audio).max()
            if peak > 0:
                audio = audio * (self.config.target_level / peak)
        
        elif self.config.normalization_type == 'rms':
            # RMS normalization
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                audio = audio * (self.config.target_level / rms)
        
        return audio
    
    def _augment(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply audio augmentation
        
        Args:
            audio: Audio waveform
            
        Returns:
            Augmented audio
        """
        import random
        
        # Time stretching
        if self.config.time_stretch_range:
            rate = random.uniform(*self.config.time_stretch_range)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        
        # Pitch shifting
        if self.config.pitch_shift_range:
            n_steps = random.uniform(*self.config.pitch_shift_range)
            audio = librosa.effects.pitch_shift(
                audio,
                sr=self.config.sample_rate,
                n_steps=n_steps
            )
        
        # Add noise
        if self.config.add_noise:
            noise = np.random.randn(len(audio))
            audio = audio + self.config.noise_factor * noise
        
        return audio
    
    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract audio features
        
        Args:
            audio: Audio waveform
            
        Returns:
            Extracted features
        """
        if self.config.feature_type == 'raw':
            return audio
        
        elif self.config.feature_type == 'spectrogram':
            # Short-time Fourier Transform
            stft = librosa.stft(
                audio,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                window=self.config.window
            )
            # Convert to magnitude
            spectrogram = np.abs(stft)
            # Convert to dB scale
            spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
            return spectrogram_db
        
        elif self.config.feature_type == 'mel_spectrogram':
            # Mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                window=self.config.window,
                n_mels=self.config.n_mels,
                fmin=self.config.fmin,
                fmax=self.config.fmax
            )
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            return mel_spec_db
        
        elif self.config.feature_type == 'mfcc':
            # Mel-frequency cepstral coefficients
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.config.sample_rate,
                n_mfcc=self.config.n_mfcc,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                win_length=self.config.win_length,
                n_mels=self.config.n_mels,
                fmin=self.config.fmin,
                fmax=self.config.fmax,
                dct_type=self.config.dct_type
            )
            return mfcc
        
        else:
            raise ValueError(f"Unknown feature type: {self.config.feature_type}")
    
    def _to_output_format(self, data: np.ndarray) -> np.ndarray:
        """
        Convert to output format
        
        Args:
            data: Input data
            
        Returns:
            Data in configured output format
        """
        # Ensure correct dtype
        data = data.astype(self.config.dtype)
        
        if self.config.output_format == 'numpy':
            return data
        elif self.config.output_format == 'tensor':
            # For PyTorch: add channel dimension if needed
            if data.ndim == 1:
                data = data[np.newaxis, :]  # (C, T)
            return data
        else:
            raise ValueError(f"Invalid output format: {self.config.output_format}")
    
    def extract_multiple_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract multiple feature types from audio
        
        Args:
            audio: Audio waveform
            
        Returns:
            Dictionary with multiple features
        """
        features = {}
        
        # Spectrogram
        stft = librosa.stft(
            audio,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        features['spectrogram'] = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        
        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.sample_rate,
            n_mels=self.config.n_mels
        )
        features['mel_spectrogram'] = librosa.power_to_db(mel_spec, ref=np.max)
        
        # MFCC
        features['mfcc'] = librosa.feature.mfcc(
            y=audio,
            sr=self.config.sample_rate,
            n_mfcc=self.config.n_mfcc
        )
        
        # Zero-crossing rate
        features['zcr'] = librosa.feature.zero_crossing_rate(audio)
        
        # Spectral centroid
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.config.sample_rate
        )
        
        # RMS energy
        features['rms'] = librosa.feature.rms(y=audio)
        
        return features
    
    def get_statistics(self, audio_samples: List[Any]) -> Dict[str, Any]:
        """
        Get statistics about audio data
        
        Args:
            audio_samples: List of audio samples
            
        Returns:
            Dictionary with statistics
        """
        durations = []
        sample_rates = []
        
        for audio_data in audio_samples:
            try:
                audio, sr = self._load_audio(audio_data)
                duration = len(audio) / sr
                durations.append(duration)
                sample_rates.append(sr)
            except Exception as e:
                self.logger.warning(f"Failed to analyze audio: {e}")
        
        return {
            'total_samples': len(audio_samples),
            'avg_duration': sum(durations) / len(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'total_duration': sum(durations),
            'sample_rates': list(set(sample_rates)),
            'target_sample_rate': self.config.sample_rate
        }


# Example usage
if __name__ == "__main__":
    # Create processor for speech recognition
    config = AudioProcessorConfig(
        sample_rate=16000,
        mono=True,
        trim_silence=True,
        normalize_amplitude=True,
        extract_features=True,
        feature_type='mel_spectrogram',
        n_mels=80,
        output_format='numpy'
    )
    
    processor = AudioProcessor(config)
    
    # Create a sample audio (sine wave)
    import numpy as np
    duration = 2.0
    sample_rate = 16000
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Process audio
    processed = processor.process((audio, sample_rate))
    print(f"Original audio shape: {audio.shape}")
    print(f"Processed feature shape: {processed.shape}")
    print(f"Feature type: {config.feature_type}")
    print(f"Feature dtype: {processed.dtype}")
    
    # Extract multiple features
    multiple_features = processor.extract_multiple_features(audio)
    print(f"\nExtracted features:")
    for name, feature in multiple_features.items():
        print(f"  {name}: {feature.shape}")
    
    # Get statistics
    stats = processor.get_statistics([(audio, sample_rate)])
    print(f"\nStatistics: {stats}")
