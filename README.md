# Pretrained Models Documentation

## Table of Contents
1. [Overview](#overview)
2. [Model 1: pyannote/speaker-diarization-3.1](#model-1-pyannotespeaker-diarization-31)
3. [Model 2: OpenAI Whisper (via WhisperX)](#model-2-openai-whisper-via-whisperx)
4. [Model 3: Resemblyzer VoiceEncoder](#model-3-resemblyzer-voiceencoder)
5. [Model 4: Wav2Vec2 Alignment Model](#model-4-wav2vec2-alignment-model)
6. [Model Loading & Caching](#model-loading--caching)
7. [Model Requirements](#model-requirements)
8. [Troubleshooting Model Issues](#troubleshooting-model-issues)

---

## Overview

This project uses **4 pretrained models** from different sources to perform speaker diarization, transcription, and identification. All models are loaded automatically on first use and cached locally.

| Model | Purpose | Source | Size | License |
|-------|---------|--------|------|---------|
| pyannote/speaker-diarization-3.1 | Speaker Diarization | Hugging Face | ~500MB | Custom (requires acceptance) |
| OpenAI Whisper | Speech Recognition | OpenAI | 39M-1550M params | MIT |
| Resemblyzer VoiceEncoder | Speaker Embeddings | GitHub | ~5MB | MIT |
| Wav2Vec2 | Word Alignment | Hugging Face | ~300MB | MIT |

---

## Model 1: pyannote/speaker-diarization-3.1

### Purpose
Determines **"who spoke when"** in an audio recording. Assigns speaker labels (SPEAKER_00, SPEAKER_01, etc.) to time segments.

### Location
- **Hugging Face**: [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- **Loaded in**: `diarization/diarize.py`
- **Function**: `load_diarization_pipeline()`

### Architecture

The pipeline consists of **3 neural networks**:

1. **Segmentation Model (PyanNet)**
   - **Architecture**: SincNet + LSTM + Feedforward layers
   - **Input**: Audio spectrograms (10-second windows)
   - **Output**: For each 17ms frame, predicts which speakers are active
   - **Purpose**: Detects when speech occurs and how many speakers are talking

2. **Embedding Model (ECAPA-TDNN)**
   - **Architecture**: ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation)
   - **Input**: Speaker segments from segmentation model
   - **Output**: 512-dimensional speaker embedding vectors
   - **Purpose**: Creates voiceprints that capture unique speaker characteristics

3. **Clustering Algorithm**
   - **Method**: Agglomerative clustering
   - **Input**: Speaker embeddings
   - **Output**: Speaker labels (SPEAKER_00, SPEAKER_01, etc.)
   - **Purpose**: Groups similar embeddings together (same person = same label)

### Model Details

- **Framework**: PyTorch
- **Size**: ~500MB (compressed)
- **Training Data**: Multiple datasets (VoxCeleb, LibriSpeech, etc.)
- **Languages**: Multilingual (works best with English)
- **Sample Rate**: 16kHz
- **GPU Support**: Yes (CUDA)

### Usage

```python
from diarization.diarize import load_diarization_pipeline, diarize_audio

# Load pipeline (requires HF token)
pipeline = load_diarization_pipeline(use_auth_token="your_token")

# Run diarization
segments = diarize_audio("audio.wav", pipeline=pipeline)
```

### Output Format

```python
[
    {"start": 0.0, "end": 2.5, "speaker_label": "SPEAKER_00"},
    {"start": 2.5, "end": 5.0, "speaker_label": "SPEAKER_01"},
    ...
]
```

### Requirements

⚠️ **CRITICAL**: This model is **gated** and requires:
1. **Hugging Face account** (free)
2. **Accept model terms** at [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. **Access token** from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Configuration

- **Device**: Auto-detects GPU/CPU (prefers GPU)
- **Min/Max Speakers**: Can be specified (optional hints)
- **Pipeline Version**: 3.1 (latest stable)

### Performance

- **Accuracy**: ~95% DER (Diarization Error Rate) on clean audio
- **Speed**: 
  - CPU: ~1-2x real-time
  - GPU: ~10-20x real-time
- **Limitations**: 
  - Struggles with overlapping speech
  - Performance degrades with >5 speakers
  - Requires clear audio (SNR > 20dB)

---

## Model 2: OpenAI Whisper (via WhisperX)

### Purpose
Converts **speech to text** with word-level timestamps. Provides accurate transcription of what was said.

### Location
- **Original**: [OpenAI Whisper](https://github.com/openai/whisper)
- **Wrapper**: [WhisperX](https://github.com/m-bain/whisperX)
- **Loaded in**: `asr/transcribe.py`
- **Function**: `load_whisperx_model()`

### Architecture

**Whisper Model** (Encoder-Decoder Transformer):
- **Encoder**: Converts audio spectrograms into hidden states
- **Decoder**: Auto-regressively generates text tokens
- **Training**: 680,000+ hours of multilingual audio
- **Languages**: 99+ languages (English, Spanish, French, etc.)

### Model Variants

| Model | Parameters | Size | Speed | Accuracy | Use Case |
|-------|------------|------|-------|----------|----------|
| `tiny` | 39M | ~75MB | Fastest | Good | Quick testing |
| `base` | 74M | ~150MB | Fast | Very Good | **Recommended** |
| `small` | 244M | ~500MB | Medium | Excellent | Production |
| `medium` | 769M | ~1.5GB | Slow | Excellent | High accuracy |
| `large` | 1550M | ~3GB | Slowest | Best | Maximum accuracy |
| `large-v2` | 1550M | ~3GB | Slowest | Best | Latest large model |
| `large-v3` | 1550M | ~3GB | Slowest | Best | Latest version |

**Default**: `base` (best balance of speed and accuracy)

### WhisperX Enhancements

WhisperX adds:
1. **Batched Inference**: Processes multiple segments simultaneously (70x faster)
2. **CTranslate2**: Optimized inference engine
3. **Word-Level Alignment**: Uses separate alignment model (see Model 4)

### Model Details

- **Framework**: PyTorch (via CTranslate2)
- **License**: MIT
- **Sample Rate**: 16kHz
- **GPU Support**: Yes (CUDA)
- **Quantization**: Supports int8, int8_float16, float16, float32

### Usage

```python
from asr.transcribe import load_whisperx_model, transcribe_audio

# Load model
models = load_whisperx_model(model_name="base", device="cuda")

# Transcribe
segments = transcribe_audio("audio.wav", models=models)
```

### Output Format

```python
[
    {"start": 0.0, "end": 2.5, "text": "Hello, how are you?"},
    {"start": 2.5, "end": 5.0, "text": "I'm doing well, thanks."},
    ...
]
```

### Configuration

- **Model Size**: `--whisper-model tiny|base|small|medium|large`
- **Device**: Auto-detects GPU/CPU
- **Batch Size**: 16 (default, can be adjusted)
- **Compute Type**: `int8` (default, faster), `float16` or `float32` (more accurate)

### Performance

- **Accuracy**: 
  - English: ~95% WER (Word Error Rate) on clean audio
  - Multilingual: Varies by language
- **Speed**:
  - CPU: ~0.5-1x real-time (base model)
  - GPU: ~20-50x real-time (base model)
- **Limitations**:
  - Requires clear audio
  - Struggles with heavy accents
  - May hallucinate on very noisy audio

---

## Model 3: Resemblyzer VoiceEncoder

### Purpose
Extracts **speaker embeddings** (voiceprints) from audio. Used to identify if a speaker matches someone in the registry.

### Location
- **GitHub**: [Resemblyzer](https://github.com/resemble-ai/Resemblyzer)
- **Loaded in**: `embeddings/speaker_id.py`
- **Function**: `load_speaker_encoder()`

### Architecture

**GE2E (Generalized End-to-End) Speaker Encoder**:
- **Architecture**: 3-layer LSTM
- **Input**: 40-channel mel spectrograms
- **Output**: 256-dimensional embedding vector
- **Training**: VoxCeleb dataset (7,000+ speakers)

### Model Details

- **Framework**: PyTorch
- **Size**: ~5MB
- **License**: MIT
- **Sample Rate**: 16kHz
- **Embedding Dimension**: 256
- **GPU Support**: Yes (but small model, CPU is fine)

### How It Works

1. **Preprocessing**: Converts audio to 40-channel mel spectrogram
2. **LSTM Encoding**: 3-layer LSTM processes spectrogram frames
3. **Embedding Extraction**: Final hidden state → 256-dim vector
4. **Normalization**: L2 normalization for cosine similarity

### Usage

```python
from embeddings.speaker_id import (
    load_speaker_encoder,
    extract_speaker_embedding,
    identify_speaker
)

# Load encoder
encoder = load_speaker_encoder()

# Extract embedding
embedding = extract_speaker_embedding("audio.wav", encoder)

# Identify speaker (requires registry)
speaker_name, similarity = identify_speaker(embedding, threshold=0.75)
```

### Output Format

**Embedding**: NumPy array of shape `(256,)` with float values

**Identification Result**:
```python
("Alice", 0.87)  # (speaker_name, similarity_score)
# OR
("UNKNOWN", 0.65)  # If no match above threshold
```

### Similarity Scores

- **Same Person**: 0.75 - 0.95 (typically 0.80-0.90)
- **Different Person**: 0.30 - 0.65 (typically 0.40-0.60)
- **Threshold**: 0.75 (recommended, balances accuracy and security)

### Performance

- **Accuracy**: 96.7% on clean audio (with 0.75 threshold)
- **Speed**: 
  - CPU: ~100x real-time (very fast)
  - GPU: Not necessary (model is small)
- **Limitations**:
  - Requires clear audio (SNR > 20dB)
  - May struggle with identical twins
  - Voice changes (illness, age) can affect embeddings

---

## Model 4: Wav2Vec2 Alignment Model

### Purpose
Provides **word-level timestamps** for Whisper transcriptions. Whisper alone gives sentence-level timestamps; this model adds precise word boundaries.

### Location
- **Hugging Face**: `WAV2VEC2_ASR_BASE_960H` (or similar)
- **Loaded in**: `asr/transcribe.py` (via WhisperX)
- **Function**: `whisperx.load_align_model()`

### Architecture

**Wav2Vec2**:
- **Architecture**: Convolutional encoder + Transformer
- **Training**: LibriSpeech (960 hours of English)
- **Purpose**: Forced alignment (matching text to audio)

### Model Details

- **Framework**: PyTorch
- **Size**: ~300MB
- **License**: MIT
- **Language**: English (other languages have separate models)
- **Sample Rate**: 16kHz

### How It Works

1. **Input**: Audio + Text transcript (from Whisper)
2. **Alignment**: Finds optimal alignment between phonemes and audio frames
3. **Output**: Word-level start/end times

### Usage

This model is loaded automatically by WhisperX:

```python
import whisperx

# Load alignment model
align_model, metadata = whisperx.load_align_model(
    language_code="en",
    device="cuda"
)

# Align transcription
aligned = whisperx.align(
    segments,
    align_model,
    metadata,
    audio,
    device
)
```

### Output Format

Adds word-level timestamps to Whisper segments:

```python
{
    "start": 0.0,
    "end": 2.5,
    "text": "Hello, how are you?",
    "words": [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "how", "start": 0.6, "end": 0.8},
        ...
    ]
}
```

### Performance

- **Accuracy**: ~95% word alignment accuracy
- **Speed**: Very fast (adds <1 second to processing)
- **Limitations**: 
  - English only (other languages need different models)
  - Requires accurate transcription from Whisper

---

## Model Loading & Caching

### Automatic Download

All models are downloaded automatically on first use:
- **pyannote**: Downloaded to `~/.cache/huggingface/`
- **Whisper**: Downloaded to `~/.cache/whisper/`
- **Resemblyzer**: Downloaded to `~/.cache/torch/hub/`
- **Wav2Vec2**: Downloaded to `~/.cache/huggingface/`

### Caching Behavior

- **First Run**: Downloads models (may take several minutes)
- **Subsequent Runs**: Loads from cache (instant)
- **Cache Size**: ~4-5GB total (depending on Whisper model)

### Manual Download

You can pre-download models:

```python
# pyannote
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="token")

# Whisper
import whisper
model = whisper.load_model("base")

# Resemblyzer
from resemblyzer import VoiceEncoder
encoder = VoiceEncoder()
```

### Clearing Cache

To free up space or force re-download:

```bash
# Linux/Mac
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/whisper/
rm -rf ~/.cache/torch/hub/

# Windows
rmdir /s "%USERPROFILE%\.cache\huggingface"
rmdir /s "%USERPROFILE%\.cache\whisper"
rmdir /s "%USERPROFILE%\.cache\torch\hub"
```

---

## Model Requirements

### System Requirements

| Model | CPU | RAM | GPU | Disk Space |
|-------|-----|-----|-----|------------|
| pyannote | Any | 4GB+ | Recommended | 500MB |
| Whisper (base) | Any | 4GB+ | Recommended | 150MB |
| Whisper (large) | Any | 8GB+ | Required | 3GB |
| Resemblyzer | Any | 2GB+ | Optional | 5MB |
| Wav2Vec2 | Any | 2GB+ | Optional | 300MB |

### Python Dependencies

All models require:
- **PyTorch** (CPU or GPU version)
- **NumPy**
- **librosa** (for audio processing)

Specific requirements:
- **pyannote**: `pyannote.audio`, `huggingface-hub`
- **WhisperX**: `whisperx`, `faster-whisper`, `ctranslate2`
- **Resemblyzer**: `resemblyzer`

See `requirements.txt` for exact versions.

### GPU Requirements (Optional)

- **CUDA**: 11.8+ (for PyTorch)
- **cuDNN**: 8.6+ (usually bundled)
- **VRAM**: 2GB+ (4GB+ recommended for large models)

---

## Troubleshooting Model Issues

### Issue 1: "Model not found" or "Connection error"

**Problem**: Cannot download model from internet.

**Solutions**:
1. Check internet connection
2. For Hugging Face models: Verify token is set correctly
3. For pyannote: Accept model terms on Hugging Face website
4. Try manual download (see above)

### Issue 2: "Out of memory" errors

**Problem**: Model too large for available RAM/VRAM.

**Solutions**:
1. Use smaller Whisper model (`tiny` or `base` instead of `large`)
2. Process shorter audio segments
3. Use CPU instead of GPU (slower but less memory)
4. Close other applications to free RAM

### Issue 3: Slow processing

**Problem**: Models running on CPU or inefficient settings.

**Solutions**:
1. Install GPU-enabled PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
2. Use smaller Whisper model
3. Reduce batch size in WhisperX
4. Ensure models are cached (not downloading each time)

### Issue 4: "Permission denied" for cache directory

**Problem**: Cannot write to cache directory.

**Solutions**:
1. Check directory permissions: `~/.cache/`
2. Set custom cache directory:
   ```python
   import os
   os.environ['HF_HOME'] = '/path/to/cache'
   os.environ['WHISPER_CACHE_DIR'] = '/path/to/cache'
   ```

### Issue 5: "Model version mismatch"

**Problem**: Cached model version doesn't match code.

**Solutions**:
1. Clear cache (see above)
2. Update dependencies: `pip install --upgrade -r requirements.txt`
3. Force re-download by deleting specific model cache

### Issue 6: pyannote "Authentication required"

**Problem**: Hugging Face token not set or invalid.

**Solutions**:
1. Accept model terms: [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. Get token: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Set environment variable: `export HF_TOKEN="your_token"`
4. Or pass via code: `use_auth_token="your_token"`

---

## Model Version Information

### Current Versions (as of 2024)

- **pyannote.audio**: 3.1.1
- **pyannote/speaker-diarization**: 3.1
- **whisperx**: 3.1.1
- **OpenAI Whisper**: Latest (via WhisperX)
- **resemblyzer**: 0.1.4
- **Wav2Vec2**: Latest (via WhisperX)

### Checking Installed Versions

```bash
pip show pyannote.audio
pip show whisperx
pip show resemblyzer
```

### Updating Models

To update to latest versions:

```bash
pip install --upgrade pyannote.audio whisperx resemblyzer
```

**Note**: Model weights are cached separately. Clearing cache will force re-download of latest weights.

---

## Model Licenses Summary

| Model | License | Commercial Use | Redistribution |
|-------|---------|----------------|----------------|
| pyannote/speaker-diarization-3.1 | Custom | ✅ (with terms acceptance) | ❌ |
| OpenAI Whisper | MIT | ✅ | ✅ |
| Resemblyzer | MIT | ✅ | ✅ |
| Wav2Vec2 | MIT | ✅ | ✅ |

**Important**: Always check license terms before commercial use. pyannote model requires explicit acceptance of terms on Hugging Face.

---

## Additional Resources

### Official Documentation

- **pyannote.audio**: [github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)
- **WhisperX**: [github.com/m-bain/whisperX](https://github.com/m-bain/whisperX)
- **Resemblyzer**: [github.com/resemble-ai/Resemblyzer](https://github.com/resemble-ai/Resemblyzer)
- **OpenAI Whisper**: [github.com/openai/whisper](https://github.com/openai/whisper)

### Model Cards

- **pyannote/speaker-diarization-3.1**: [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- **Whisper Models**: [github.com/openai/whisper#available-models-and-languages](https://github.com/openai/whisper#available-models-and-languages)

### Research Papers

- **pyannote**: [arxiv.org/abs/1911.01255](https://arxiv.org/abs/1911.01255)
- **Whisper**: [arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356)
- **GE2E (Resemblyzer)**: [arxiv.org/abs/1710.10467](https://arxiv.org/abs/1710.10467)
- **Wav2Vec2**: [arxiv.org/abs/2006.11477](https://arxiv.org/abs/2006.11477)

---

**Last Updated**: 2024  
**Version**: 2.0  
**Maintained By**: Project Contributors
