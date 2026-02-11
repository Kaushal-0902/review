# Pretrained Models Documentation

This document provides a comprehensive explanation of all pretrained models used in the Speaker Diarization and Recognition system, including their versions, internal workings, and how they are integrated into the pipeline.

---

## Table of Contents

1. [pyannote.audio Speaker Diarization Model](#1-pyannoteaudio-speaker-diarization-model)
2. [WhisperX Speech Recognition Models](#2-whisperx-speech-recognition-models)
3. [Wav2Vec2 Alignment Model](#3-wav2vec2-alignment-model)
4. [Resemblyzer VoiceEncoder](#4-resemblyzer-voiceencoder)
5. [Model Integration Flow](#model-integration-flow)
6. [Version Specifications](#version-specifications)
7. [Model Loading and Initialization](#model-loading-and-initialization)

---

## 1. pyannote.audio Speaker Diarization Model

### Model Information
- **Model Name:** `pyannote/speaker-diarization-3.1`
- **Library Version:** `pyannote.audio==3.1.1`
- **Source:** Hugging Face Hub
- **License:** MIT License
- **Access:** Requires Hugging Face account and token (free after accepting terms)

### What It Does
Identifies **who spoke when** in an audio recording by:
- Detecting speech activity (speech vs. silence)
- Identifying speaker change points
- Detecting overlapping speech (multiple speakers talking simultaneously)
- Assigning speaker labels (SPEAKER_00, SPEAKER_01, etc.) with timestamps

### Internal Architecture

The `pyannote/speaker-diarization-3.1` pipeline is a **multi-stage neural network system** that consists of three main components:

#### Stage 1: Local Neural Speaker Segmentation
- **Input:** Raw audio waveform
- **Processing:** 
  - Uses a **5-second sliding window** with 500ms step overlap
  - Applies a neural network to each window to detect:
    - Speech activity (is someone speaking?)
    - Speaker change points (did the speaker change?)
  - Performs test-time augmentation through overlapping windows
- **Output:** Local speaker segments with timestamps

#### Stage 2: Neural Speaker Embedding Extraction
- **Input:** Audio segments from Stage 1
- **Processing:**
  - Uses a separate neural network (different from segmentation)
  - Extracts **voice embeddings** (mathematical representations of voice characteristics)
  - Creates **overlap-aware embeddings** that can handle simultaneous speakers
  - Uses longer audio context (not just 5-second windows) for better embeddings
- **Output:** 256-dimensional embedding vectors for each speaker segment

#### Stage 3: Global Agglomerative Clustering
- **Input:** Speaker embeddings from Stage 2
- **Processing:**
  - Groups similar embeddings together using clustering algorithms
  - Connects local segments into global speaker identities
  - Resolves speaker labels across the entire audio file
- **Output:** Final speaker labels (SPEAKER_00, SPEAKER_01, etc.) with global timestamps

### How We Use It

**Location:** `diarization/diarize.py`

```python
from pyannote.audio import Pipeline

# Load the pretrained pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=hf_token
)

# Move to GPU if available
if torch.cuda.is_available():
    pipeline = pipeline.to(torch.device("cuda"))

# Run diarization
diarization = pipeline(audio_file)

# Extract segments
segments = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    segments.append({
        'start': turn.start,
        'end': turn.end,
        'speaker_label': speaker  # e.g., "SPEAKER_00"
    })
```

### Performance Metrics
- **Diarization Error Rate (DER):** ~11.7% on AISHELL-4 benchmark
- **Processing Speed:** ~1-2x real-time on GPU (depends on audio length)
- **Memory Usage:** ~2-4GB GPU memory

### Key Technical Details
- **Window Size:** 5 seconds (optimal balance between accuracy and computational cost)
- **Overlap:** 500ms step (provides test-time augmentation)
- **Embedding Dimension:** 256 floats per speaker
- **Clustering Method:** Agglomerative hierarchical clustering
- **Overlap Handling:** Can detect and label overlapping speech segments

---

## 2. WhisperX Speech Recognition Models

### Model Information
- **Library Version:** `whisperx==3.1.1`
- **Base Model:** OpenAI Whisper
- **Model Sizes Available:** tiny, base, small, medium, large
- **Default Used:** `base` (can be changed via `--whisper-model` flag)
- **Source:** OpenAI (via WhisperX wrapper)
- **License:** MIT License

### What It Does
Converts **speech to text** with precise word-level timestamps:
- Transcribes spoken words into text
- Provides segment-level timestamps (start/end times for each sentence/phrase)
- Supports multiple languages (English by default in this project)
- Handles various accents and speaking styles

### Available Model Sizes

| Model Size | Parameters | Memory (GPU) | Speed | Accuracy |
|------------|-----------|--------------|-------|----------|
| tiny       | 39M       | ~1GB         | Fastest| Lowest   |
| base       | 74M       | ~1GB         | Fast   | Good     |
| small      | 244M      | ~2GB         | Medium | Better   |
| medium     | 769M      | ~5GB         | Slow   | High     |
| large      | 1550M     | ~10GB        | Slowest| Highest  |

**Default:** `base` (good balance of speed and accuracy)

### Internal Architecture

WhisperX uses OpenAI's Whisper architecture, which is a **transformer-based encoder-decoder model**:

#### Encoder (Audio Processing)
- **Input:** Audio waveform (16kHz, mono)
- **Processing:**
  - Converts audio to mel-spectrogram (visual representation of sound frequencies)
  - Processes through convolutional layers
  - Creates audio feature representations
- **Output:** Encoded audio features

#### Decoder (Text Generation)
- **Input:** Encoded audio features
- **Processing:**
  - Uses transformer decoder architecture
  - Generates text tokens autoregressively (word by word)
  - Predicts next word based on audio context and previous words
- **Output:** Text transcription

#### Word-Level Alignment
- **Input:** Transcription + audio
- **Processing:**
  - Uses separate alignment model (Wav2Vec2 - see Section 3)
  - Aligns each word to precise timestamps in audio
- **Output:** Word-level timestamps

### How We Use It

**Location:** `asr/transcribe.py`

```python
import whisperx

# Load WhisperX model
model = whisperx.load_model(
    model_name="base",  # or tiny, small, medium, large
    device="cuda",       # or "cpu"
    compute_type="int8" # quantization for faster inference
)

# Load alignment model (for word-level timestamps)
align_model, metadata = whisperx.load_align_model(
    language_code="en",
    device="cuda"
)

# Load audio
audio = whisperx.load_audio(audio_file)

# Transcribe
result = model.transcribe(audio, batch_size=16)

# Align for word-level timestamps
result = whisperx.align(
    result["segments"],
    align_model,
    metadata,
    audio,
    device,
    return_char_alignments=False
)

# Extract segments
segments = []
for segment in result["segments"]:
    segments.append({
        'start': segment['start'],  # Start time in seconds
        'end': segment['end'],      # End time in seconds
        'text': segment['text'].strip()  # Transcribed text
    })
```

### Performance Metrics
- **Word Error Rate (WER):** ~5-10% on clean English audio (varies by model size)
- **Processing Speed:** 
  - Base model: ~10-20x real-time on GPU
  - Large model: ~2-5x real-time on GPU
- **Memory Usage:** Varies by model size (see table above)

### Key Technical Details
- **Audio Format:** 16kHz, mono (automatically resampled if needed)
- **Batch Processing:** Processes audio in batches (default batch_size=16)
- **Quantization:** Uses int8 quantization by default for faster inference
- **Language Support:** Supports 99+ languages (English used in this project)
- **Timestamp Precision:** Millisecond-level accuracy for word timestamps

---

## 3. Wav2Vec2 Alignment Model

### Model Information
- **Model Type:** Wav2Vec2-based alignment model
- **Library:** WhisperX (bundled with alignment functionality)
- **Purpose:** Word-level timestamp alignment
- **Language:** English (language-specific models available)
- **Source:** WhisperX library

### What It Does
Aligns transcribed words to **precise timestamps** in the audio:
- Takes transcription from Whisper model
- Matches each word to exact time positions in audio
- Provides word-level start/end times (not just segment-level)

### Internal Architecture

Wav2Vec2 is a **self-supervised learning model** for speech:

#### Feature Extraction
- **Input:** Raw audio waveform
- **Processing:**
  - Uses convolutional layers to extract features
  - Creates frame-level representations (one per ~20ms of audio)
  - Learns representations through self-supervised pretraining
- **Output:** Frame-level feature vectors

#### Alignment Process
- **Input:** 
  - Transcription text (from Whisper)
  - Audio features (from Wav2Vec2)
- **Processing:**
  - Uses forced alignment algorithm
  - Matches phonemes/words to audio frames
  - Uses dynamic time warping (DTW) or similar alignment techniques
- **Output:** Word-level timestamps

### How We Use It

**Location:** `asr/transcribe.py` (integrated with WhisperX)

```python
# Loaded automatically with WhisperX
align_model, metadata = whisperx.load_align_model(
    language_code="en",  # English alignment model
    device="cuda"
)

# Used during transcription alignment
result = whisperx.align(
    result["segments"],      # Whisper transcription
    align_model,             # Wav2Vec2 alignment model
    metadata,                 # Model metadata
    audio,                   # Original audio
    device,                  # GPU/CPU
    return_char_alignments=False  # Word-level only
)
```

### Key Technical Details
- **Frame Rate:** ~50 frames per second (20ms per frame)
- **Alignment Method:** Forced alignment with dynamic programming
- **Precision:** Millisecond-level word timestamps
- **Language Models:** Separate models for different languages

---

## 4. Resemblyzer VoiceEncoder

### Model Information
- **Library Version:** `resemblyzer==0.1.4`
- **Model Type:** Speaker embedding encoder
- **Embedding Dimension:** 256 floats
- **Source:** GitHub (resemblyzer library)
- **License:** MIT License

### What It Does
Extracts **speaker voice embeddings** (voiceprints) from audio:
- Converts audio segments into fixed-size numerical vectors
- Each unique voice produces a distinct embedding
- Enables speaker comparison through cosine similarity
- Used for speaker identification and verification

### Internal Architecture

Resemblyzer uses a **deep neural network** based on ResNet architecture:

#### Audio Preprocessing
- **Input:** Raw audio (any format)
- **Processing:**
  - Resamples to 16kHz mono
  - Normalizes audio levels
  - Converts to mel-spectrogram (frequency representation)
- **Output:** Preprocessed audio features

#### Neural Network (VoiceEncoder)
- **Architecture:** ResNet-style convolutional neural network
- **Input:** Mel-spectrogram features
- **Processing:**
  - Passes through multiple convolutional layers
  - Uses residual connections (ResNet blocks)
  - Applies global average pooling
  - Final fully connected layer produces embedding
- **Output:** 256-dimensional embedding vector

#### Embedding Properties
- **Dimension:** 256 float values
- **Normalization:** Embeddings are L2-normalized
- **Invariance:** Robust to:
  - Different recording conditions
  - Background noise (to some extent)
  - Speaking style variations
- **Uniqueness:** Different speakers produce different embeddings

### How We Use It

**Location:** `embeddings/speaker_id.py`

```python
from resemblyzer import VoiceEncoder, preprocess_wav

# Load encoder (loads pretrained weights automatically)
encoder = VoiceEncoder()

# Extract embedding from audio file
wav = preprocess_wav(audio_file)  # Preprocesses audio
embedding = encoder.embed_utterance(wav)  # Returns 256-dim numpy array

# Extract embedding from audio array
embedding = encoder.embed_utterance(audio_array)  # Direct from array

# Compare embeddings using cosine similarity
similarity = cosine_similarity(embedding1, embedding2)
# Returns value between -1 and 1 (higher = more similar)
```

### Speaker Identification Process

1. **Enrollment (Adding Known Speakers):**
   ```python
   # Extract embedding from reference audio
   embedding = extract_speaker_embedding("alice_sample.wav")
   
   # Save to registry
   save_speaker_to_registry("Alice", embedding, metadata={...})
   ```

2. **Identification (Matching Unknown Speaker):**
   ```python
   # Extract embedding from unknown audio segment
   unknown_embedding = extract_embedding_from_segment(segment_audio, sr)
   
   # Compare with all known speakers
   best_match = None
   best_similarity = -1.0
   
   for speaker_name, data in registry.items():
       known_embedding = data['embedding']
       similarity = cosine_similarity(unknown_embedding, known_embedding)
       
       if similarity > best_similarity:
           best_similarity = similarity
           best_match = speaker_name
   
   # If similarity >= threshold, speaker is identified
   if best_similarity >= threshold:  # default: 0.75
       identified_speaker = best_match
   else:
       identified_speaker = "UNKNOWN"
   ```

### Performance Metrics
- **Embedding Extraction Speed:** ~100-200x real-time on GPU
- **Similarity Computation:** Very fast (simple cosine similarity)
- **Accuracy:** ~95-98% speaker verification accuracy on clean audio
- **Robustness:** Works well with 0.5+ second audio segments

### Key Technical Details
- **Minimum Audio Length:** ~0.5 seconds recommended
- **Optimal Audio Length:** 1-5 seconds for best embeddings
- **Sample Rate:** 16kHz (automatically resampled if needed)
- **Similarity Threshold:** Default 0.75 (configurable)
- **Embedding Storage:** 256 floats = 1KB per speaker (very efficient)

---

## Model Integration Flow

### Complete Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Raw Audio File (WAV/MP3)                  │
└───────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  [librosa] Audio Preprocessing                               │
│  - Load audio                                                │
│  - Resample to 16kHz mono                                    │
│  - Normalize audio levels                                    │
└───────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  [pyannote.audio] Speaker Diarization                       │
│  Model: pyannote/speaker-diarization-3.1                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Stage 1: Local Segmentation (5s windows)              │   │
│  │ Stage 2: Embedding Extraction (256-dim vectors)     │   │
│  │ Stage 3: Global Clustering (speaker grouping)        │   │
│  └─────────────────────────────────────────────────────┘   │
│  Output: [(start, end, SPEAKER_00), ...]                   │
└───────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  [WhisperX] Speech Recognition                              │
│  Model: OpenAI Whisper (base/small/medium/large)            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Encoder: Audio → Features                            │   │
│  │ Decoder: Features → Text                             │   │
│  └─────────────────────────────────────────────────────┘   │
│  Output: [(start, end, "transcribed text"), ...]            │
└───────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  [Wav2Vec2] Word-Level Alignment                            │
│  Model: Wav2Vec2 alignment (via WhisperX)                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Forced Alignment: Words → Precise Timestamps        │   │
│  └─────────────────────────────────────────────────────┘   │
│  Output: Word-level timestamps                               │
└───────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  [Alignment Module] Fuse Diarization + ASR                 │
│  - Match timestamps between diarization and transcription   │
│  - Assign speaker labels to text segments                   │
│  Output: [(start, end, text, SPEAKER_00), ...]             │
└───────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  [Resemblyzer] Speaker Embedding Extraction                 │
│  Model: VoiceEncoder (ResNet-based)                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Audio → Mel-spectrogram → ResNet → 256-dim vector   │   │
│  └─────────────────────────────────────────────────────┘   │
│  Output: Embedding vectors for each segment                 │
└───────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  [Speaker Identification] Compare Embeddings                 │
│  - Load speaker registry (known speakers)                   │
│  - Compute cosine similarity                                │
│  - Match if similarity >= threshold (0.75)                 │
│  Output: [(start, end, text, "Alice", similarity), ...]      │
└───────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  [Authorization Filter] Filter Unauthorized Speakers         │
│  - Remove segments from UNKNOWN speakers                    │
│  - Keep only authorized speakers                            │
│  Output: Final segments with authorization status           │
└───────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
                    JSON Output / CLI Display
```

---

## Version Specifications

### Exact Versions Used

| Component | Version | Source | Purpose |
|-----------|---------|--------|---------|
| **pyannote.audio** | 3.1.1 | Hugging Face | Diarization pipeline |
| **pyannote.core** | 5.0.0 | PyPI | Core utilities |
| **pyannote.database** | 5.0.1 | PyPI | Database utilities |
| **pyannote.pipeline** | 3.0.1 | PyPI | Pipeline framework |
| **pyannote.metrics** | 3.2.1 | PyPI | Evaluation metrics |
| **whisperx** | 3.1.1 | PyPI | ASR wrapper |
| **faster-whisper** | 1.0.3 | PyPI | Fast Whisper inference |
| **ctranslate2** | 4.3.1 | PyPI | Translation engine |
| **resemblyzer** | 0.1.4 | PyPI | Speaker embeddings |
| **transformers** | 4.38.2 | Hugging Face | Model loading |
| **huggingface-hub** | 0.23.4 | PyPI | Model downloads |
| **torch** | 2.1.2 | PyTorch | Deep learning framework |
| **torchaudio** | 2.1.2 | PyTorch | Audio processing |
| **librosa** | 0.10.1 | PyPI | Audio loading |

### Model Checkpoints (Downloaded Automatically)

| Model | Hugging Face ID | Size | Download Location |
|-------|----------------|------|-------------------|
| pyannote diarization | `pyannote/speaker-diarization-3.1` | ~80MB | `~/.cache/huggingface/` |
| Whisper base | `openai/whisper-base` | ~150MB | WhisperX cache |
| Whisper large | `openai/whisper-large-v2` | ~3GB | WhisperX cache |
| Wav2Vec2 align | `WAV2VEC2_ASR_BASE_960H` | ~300MB | WhisperX cache |
| Resemblyzer | Bundled with library | ~60MB | Library installation |

---

## Model Loading and Initialization

### Loading Sequence

1. **First Run (Models Not Cached):**
   ```
   User runs: python main.py audio.wav
   
   Step 1: Check cache for pyannote model
   → Not found → Download from Hugging Face (~80MB)
   → Save to ~/.cache/huggingface/
   
   Step 2: Check cache for Whisper model
   → Not found → Download from OpenAI (~150MB for base)
   → Save to WhisperX cache directory
   
   Step 3: Check cache for alignment model
   → Not found → Download Wav2Vec2 (~300MB)
   → Save to WhisperX cache directory
   
   Step 4: Load Resemblyzer
   → Already installed with library
   → Loads pretrained weights automatically
   
   Step 5: Move models to GPU (if available)
   → Transfer model weights to CUDA memory
   ```

2. **Subsequent Runs (Models Cached):**
   ```
   Models loaded from cache (much faster)
   → No downloads needed
   → Direct loading from disk
   ```

### Memory Management

**GPU Memory Usage (Base Whisper Model):**
- pyannote.audio: ~2GB
- Whisper base: ~1GB
- Wav2Vec2 alignment: ~500MB
- Resemblyzer: ~200MB
- **Total:** ~3.7GB GPU memory

**CPU Mode:**
- Same models, but slower processing
- Uses system RAM instead of GPU memory
- ~2-5x slower than GPU

### Model Initialization Code

**Diarization Model:**
```python
# Location: diarization/diarize.py
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=hf_token  # Required for first access
)
if torch.cuda.is_available():
    pipeline = pipeline.to(torch.device("cuda"))
```

**WhisperX Model:**
```python
# Location: asr/transcribe.py
model = whisperx.load_model(
    model_name="base",      # Model size
    device="cuda",          # GPU/CPU
    compute_type="int8"     # Quantization
)
```

**Alignment Model:**
```python
# Location: asr/transcribe.py
align_model, metadata = whisperx.load_align_model(
    language_code="en",
    device="cuda"
)
```

**Resemblyzer Encoder:**
```python
# Location: embeddings/speaker_id.py
from resemblyzer import VoiceEncoder
encoder = VoiceEncoder()  # Automatically loads pretrained weights
```

---

## Summary

This project uses **4 main pretrained models**:

1. **pyannote/speaker-diarization-3.1** - Identifies who spoke when
2. **OpenAI Whisper (via WhisperX)** - Transcribes speech to text
3. **Wav2Vec2 Alignment (via WhisperX)** - Provides word-level timestamps
4. **Resemblyzer VoiceEncoder** - Extracts speaker voice embeddings

All models are:
- **Pretrained** (no training required)
- **Downloaded automatically** on first use
- **Cached locally** for subsequent runs
- **GPU-accelerated** when available
- **Open-source** (MIT or similar licenses)

The models work together in a pipeline to provide complete speaker-attributed transcription with authorization capabilities.
