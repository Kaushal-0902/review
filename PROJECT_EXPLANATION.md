# Speaker Diarization 2.0 - Project Explanation

## Table of Contents
1. [What is This Project?](#what-is-this-project)
2. [Key Features](#key-features)
3. [Project Architecture](#project-architecture)
4. [Step-by-Step Pipeline Workflow](#step-by-step-pipeline-workflow)
5. [Installation Guide](#installation-guide)
6. [Usage Guide](#usage-guide)
7. [Project Structure](#project-structure)
8. [Configuration Options](#configuration-options)
9. [Performance Metrics](#performance-metrics)
10. [Troubleshooting](#troubleshooting)

---

## What is This Project?

**Speaker Diarization 2.0** (S-O-EEND-SDR) is a secure, open-set speaker diarization and authorization pipeline. It processes audio recordings of conversations (meetings, interviews, etc.) and:

1. **Identifies who spoke when** (Speaker Diarization)
2. **Transcribes speech to text** (Automatic Speech Recognition)
3. **Verifies speaker identity** (Speaker Authorization)
4. **Filters out unauthorized speakers** (Security Filter)
5. **Outputs clean transcripts** of only verified, authorized speakers

Think of it as a "bouncer at a club" - only people on the guest list (the speaker registry) get their words transcribed. Unauthorized speakers are automatically filtered out.

---

## Key Features

- ✅ **Speaker Diarization**: Determines "who spoke when" using state-of-the-art neural networks
- ✅ **Speaker Authorization**: Verifies speaker identity against a secure registry using voiceprints
- ✅ **ASR Transcription**: Converts speech to text with word-level timestamps
- ✅ **Security Filter**: Automatically removes segments from unauthorized speakers
- ✅ **Open-Set Recognition**: Can identify known speakers and reject unknown ones
- ✅ **High Accuracy**: 96.7% accuracy on clean audio, 100% on long conversations

---

## Project Architecture

The project follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                    main.py (Orchestrator)                 │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Diarization  │  │      ASR      │  │  Embeddings  │
│  (pyannote)  │  │  (WhisperX)   │  │ (Resemblyzer)│
└──────────────┘  └──────────────┘  └──────────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Alignment & Fusion   │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Authorization Filter  │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │    Final Transcript    │
              └───────────────────────┘
```

---

## Step-by-Step Pipeline Workflow

When you run `python main.py audio.wav`, the following 6-step pipeline executes:

### Step 1: Audio Loading & Preprocessing

**File**: `utils/audio_utils.py`  
**Library**: `librosa`

**What happens:**
- Loads audio file (supports WAV, MP3, FLAC, etc.)
- Converts to 16kHz sample rate (standard for speech processing)
- Converts stereo → mono (single channel)
- Optionally removes silence using Voice Activity Detection (VAD)

**Output**: Preprocessed audio array at 16kHz mono

---

### Step 2: Speaker Diarization

**File**: `diarization/diarize.py`  
**Library**: `pyannote.audio`  
**Model**: `pyannote/speaker-diarization-3.1`

**What happens:**
- Processes audio to determine "who spoke when"
- Assigns speaker labels (SPEAKER_00, SPEAKER_01, etc.)
- Does NOT know actual names - just distinguishes different voices

**How it works:**
1. **Segmentation Network**: Processes audio in 10-second windows, predicts active speakers for each 17ms frame
2. **Embedding Network**: Extracts 512-dimensional speaker embeddings from each segment
3. **Clustering**: Groups similar embeddings together (same person = same label)

**Output**: List of segments with `{start, end, speaker_label}`

**Example:**
```json
[
  {"start": 0.0, "end": 2.5, "speaker_label": "SPEAKER_00"},
  {"start": 2.5, "end": 5.0, "speaker_label": "SPEAKER_01"}
]
```

**⚠️ Important**: Requires Hugging Face token (see Installation)

---

### Step 3: Automatic Speech Recognition (ASR)

**File**: `asr/transcribe.py`  
**Library**: `WhisperX` (built on OpenAI Whisper)  
**Model**: Whisper "base" (74M parameters, default)

**What happens:**
- Converts speech to text
- Provides word-level timestamps (not just sentence-level)

**How it works:**
1. **Whisper Model**: Encoder-decoder Transformer trained on 680,000+ hours of multilingual audio
2. **Forced Alignment**: Uses wav2vec2 model to get precise word-level timestamps

**Why WhisperX?**
- 70x faster than plain Whisper (batched inference + CTranslate2)
- Word-level timestamps (critical for speaker matching)
- Better handling of long audio files

**Output**: List of segments with `{start, end, text}`

**Example:**
```json
[
  {"start": 0.0, "end": 2.5, "text": "Hello, how are you?"},
  {"start": 2.5, "end": 5.0, "text": "I'm doing well, thanks."}
]
```

---

### Step 4: Alignment & Fusion

**File**: `alignment/align.py`  
**Library**: Pure Python (no ML)

**What happens:**
- Matches diarization output (who spoke) with ASR output (what was said)
- Combines them based on timestamp overlap

**How it works:**
1. For each ASR segment, find the diarization segment with highest overlap
2. Calculate overlap ratio: `overlap = intersection / ASR_segment_duration`
3. If overlap > 50%, assign speaker label
4. Merge adjacent segments from same speaker (if gap < 0.5 seconds)

**Output**: List of segments with `{start, end, text, speaker_label}`

**Example:**
```json
[
  {
    "start": 0.0,
    "end": 2.5,
    "text": "Hello, how are you?",
    "speaker_label": "SPEAKER_00"
  }
]
```

---

### Step 5: Speaker Identification

**File**: `embeddings/speaker_id.py`  
**Library**: `Resemblyzer`  
**Model**: GE2E (Generalized End-to-End) speaker encoder

**What happens:**
- Extracts voiceprint (embedding) from each segment
- Compares against speaker registry
- Returns best matching name + similarity score
- If similarity < 0.75 → returns "UNKNOWN"

**How it works:**
1. **Embedding Extraction**: 3-layer LSTM converts audio → 256-dim vector
2. **Registry Lookup**: Loads enrolled speakers from `data/speaker_registry.json`
3. **Cosine Similarity**: Calculates similarity between segment embedding and each registered speaker
4. **Decision**: If best similarity ≥ 0.75 → return speaker name, else "UNKNOWN"

**Output**: Segments with `{start, end, text, speaker_label, identified_speaker, similarity_score}`

**Example:**
```json
[
  {
    "start": 0.0,
    "end": 2.5,
    "text": "Hello, how are you?",
    "speaker_label": "SPEAKER_00",
    "identified_speaker": "Alice",
    "similarity_score": 0.87
  }
]
```

---

### Step 6: Authorization Filter

**File**: `main.py` (lines 159-177)  
**Library**: Pure Python

**What happens:**
- Filters out ALL segments where `identified_speaker == "UNKNOWN"`
- Only segments from enrolled (authorized) speakers remain
- Reports how many segments were filtered

**This is the security step** - the "bouncer at the door"

**Output**: Final list of only authorized segments

---

## Installation Guide

### Prerequisites

- **Python 3.10** (required)
- **GPU recommended** (for faster processing, but CPU works)
- **Hugging Face Account** (required for pyannote model)

### Step 1: Clone the Repository

```bash
git clone https://github.com/pvarma-05/Speaker-Diarization.git
cd Speaker-Diarization
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (CPU or GPU version)
- pyannote.audio
- WhisperX
- Resemblyzer
- librosa
- And other dependencies

### Step 3: Get Hugging Face Token

The `pyannote/speaker-diarization-3.1` model is gated and requires acceptance of terms.

1. Go to [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. Click "Accept" to accept the model terms
3. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Create a new token (read access is sufficient)
5. Copy the token

### Step 4: Set Up Hugging Face Token

**Option A: Environment Variable**
```bash
# Windows (PowerShell)
$env:HF_TOKEN="your_token_here"

# Windows (CMD)
set HF_TOKEN=your_token_here

# Linux/Mac
export HF_TOKEN="your_token_here"
```

**Option B: .env File**
Create a `.env` file in the project root:
```
HF_TOKEN=your_token_here
```

**Option C: Pass via CLI**
```bash
python main.py audio.wav --hf-token your_token_here
```

### Step 5: Enroll Speakers

Before the system can recognize speakers, you must enroll them:

```bash
python scripts/setup_speakers.py
```

This enrolls sample speakers (Alice, Bob, Charlie) from included audio samples.

To enroll your own speakers:
```bash
python scripts/enroll_speaker.py --name "Alice" --audio path/to/alice.wav
```

**Requirements for enrollment audio:**
- 3-5 seconds minimum (5-30 seconds recommended)
- Clear audio (minimal background noise)
- Single speaker only
- WAV format preferred

---

## Usage Guide

### Basic Usage

Process an audio file:
```bash
python main.py audio.wav
```

Save results to JSON:
```bash
python main.py audio.wav --output results/transcript.json
```

### Advanced Usage

**Use larger Whisper model** (better accuracy, slower):
```bash
python main.py audio.wav --whisper-model large
```

**Adjust speaker threshold** (higher = stricter):
```bash
python main.py audio.wav --speaker-threshold 0.80
```

**Disable authorization filter** (transcribe everyone):
```bash
python main.py audio.wav --no-auth-filter
```

**Use custom speaker registry**:
```bash
python main.py audio.wav --registry path/to/custom_registry.json
```

### Enroll a New Speaker

```bash
python scripts/enroll_speaker.py --name "John Doe" --audio john_voice.wav
```

List enrolled speakers:
```bash
python scripts/enroll_speaker.py --list
```

### Run Benchmarks

Test system accuracy:
```bash
python scripts/benchmark_enhanced.py
```

---

## Project Structure

```
Speaker-Diarization-2.0/
├── main.py                      # Main entry point, orchestrates pipeline
├── requirements.txt             # Python dependencies
│
├── alignment/                   # Alignment module
│   ├── __init__.py
│   └── align.py                 # Merges diarization + ASR segments
│
├── asr/                         # Automatic Speech Recognition
│   ├── __init__.py
│   └── transcribe.py            # WhisperX transcription
│
├── diarization/                 # Speaker Diarization
│   ├── __init__.py
│   └── diarize.py               # pyannote.audio diarization
│
├── embeddings/                  # Speaker Embeddings
│   ├── __init__.py
│   └── speaker_id.py            # Resemblyzer speaker identification
│
├── utils/                       # Utilities
│   ├── __init__.py
│   └── audio_utils.py           # Audio loading & preprocessing
│
├── scripts/                     # Helper scripts
│   ├── enroll_speaker.py        # Enroll speakers to registry
│   ├── setup_speakers.py        # Setup sample speakers
│   ├── benchmark_auth.py        # Benchmark accuracy
│   └── benchmark_enhanced.py    # Enhanced benchmark with noise
│
├── data/                        # Data directory
│   ├── speaker_registry.json    # Enrolled speaker voiceprints
│   ├── speakers/                # Source audio for enrollment
│   └── benchmark_enhanced/      # Test data
│
├── results/                     # Output directory
│   └── *.json                   # Generated transcripts
│
└── tests/                       # Unit tests
    └── test_pipeline_integration.py
```

---

## Configuration Options

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `audio_file` | Required | Path to input audio file |
| `--output`, `-o` | None | Path to save JSON output |
| `--hf-token` | None | HuggingFace token for model access |
| `--whisper-model` | `base` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `--speaker-threshold` | `0.75` | Similarity threshold (0-1) for speaker identification |
| `--merge-gap` | `0.5` | Maximum gap (seconds) to merge adjacent segments |
| `--no-auth-filter` | False | Disable authorization filter (include all speakers) |
| `--registry` | `data/speaker_registry.json` | Path to speaker registry file |

### Model Size Comparison

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|------------|-------|----------|----------|
| `tiny` | 39M | Fastest | Good | Quick testing |
| `base` | 74M | Fast | Very Good | **Recommended** |
| `small` | 244M | Medium | Excellent | Production |
| `medium` | 769M | Slow | Excellent | High accuracy needed |
| `large` | 1550M | Slowest | Best | Maximum accuracy |

### Speaker Threshold Tuning

| Threshold | Accuracy | False Accept Rate | False Reject Rate | Use Case |
|-----------|----------|-------------------|-------------------|----------|
| 0.60 | 88.0% | 12.5% | 0.0% | Lenient (more false positives) |
| 0.65 | 92.0% | 7.5% | 0.0% | Moderate |
| 0.70 | 94.0% | 3.8% | 2.5% | Balanced |
| **0.75** | **96.7%** | **3.8%** | **2.5%** | **Recommended** |
| 0.80 | 91.7% | 0.0% | 22.5% | Strict (more false negatives) |

---

## Performance Metrics

### Benchmark Results

| Test Scenario | Accuracy | False Accept Rate | Notes |
|---------------|----------|-------------------|-------|
| Standard (10 speakers) | 94.0% | 3.8% | Baseline performance |
| Enhanced (15 speakers) | 96.7% | 3.8% | Scales well with more speakers |
| Noisy Audio (20dB SNR) | 85.8% | 0.0% | Fails safe (rejects all) |
| Long Conversation (6 min) | 100.0% | 0.0% | Perfect on clean audio |
| Live Demo | ~100% | 0.0% | Real-world validation |

### Processing Speed

- **CPU**: ~1-2x real-time (1 minute audio = 1-2 minutes processing)
- **GPU**: ~10-20x real-time (1 minute audio = 3-6 seconds processing)

*Speed depends on audio length, number of speakers, and Whisper model size*

---

## Troubleshooting

### Common Issues

#### 1. "Failed to load diarization pipeline"

**Problem**: Hugging Face token not set or model terms not accepted.

**Solution**:
1. Accept terms at [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. Set `HF_TOKEN` environment variable or pass `--hf-token` flag
3. Verify token is valid: `huggingface-cli login`

#### 2. "No speakers enrolled"

**Problem**: Speaker registry is empty.

**Solution**:
```bash
python scripts/setup_speakers.py  # Enroll sample speakers
# OR
python scripts/enroll_speaker.py --name "YourName" --audio your_voice.wav
```

#### 3. All speakers identified as "UNKNOWN"

**Problem**: Speaker threshold too high or audio quality poor.

**Solution**:
- Lower threshold: `--speaker-threshold 0.70`
- Check audio quality (clear, minimal noise)
- Ensure enrollment audio matches test audio conditions
- Re-enroll speakers with better quality audio

#### 4. Slow processing

**Problem**: Using CPU or large Whisper model.

**Solution**:
- Use GPU if available (install CUDA-enabled PyTorch)
- Use smaller Whisper model: `--whisper-model tiny` or `base`
- Process shorter audio segments

#### 5. "Module not found" errors

**Problem**: Dependencies not installed.

**Solution**:
```bash
pip install -r requirements.txt
```

#### 6. Poor transcription accuracy

**Problem**: Audio quality or wrong language.

**Solution**:
- Use larger Whisper model: `--whisper-model large`
- Ensure audio is clear (minimal background noise)
- Check if audio is in English (Whisper supports multiple languages, but alignment model is English-only)

---

## Security Considerations

### What the System Protects Against

✅ **Random unauthorized speakers** - Voiceprint won't match  
✅ **Imposters** - Different voice characteristics  
✅ **AI-generated voice clones** - Won't perfectly match voiceprint  
✅ **Recorded playback** - Audio quality degradation helps detection  

### Limitations

⚠️ **Very high-quality voice clones** might bypass the system  
⚠️ **Identical twins** may have similar voiceprints  
⚠️ **No liveness detection** - Cannot detect if audio is live or recorded  

### Recommendations

- Combine with liveness detection for maximum security
- Use higher threshold (0.80) for sensitive applications
- Regularly update speaker registry
- Monitor false acceptance rates

---

## Next Steps

1. **Enroll your speakers**: Use `scripts/enroll_speaker.py`
2. **Test with sample audio**: Process `data/sample.wav`
3. **Run benchmarks**: Verify accuracy with `scripts/benchmark_enhanced.py`
4. **Process your audio**: Use `main.py` with your recordings
5. **Customize settings**: Adjust threshold and model size as needed

---

## Support & Resources

- **GitHub Repository**: [Speaker-Diarization](https://github.com/pvarma-05/Speaker-Diarization)
- **Hugging Face Models**:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [OpenAI Whisper](https://github.com/openai/whisper)
- **Documentation**:
  - [pyannote.audio](https://github.com/pyannote/pyannote-audio)
  - [WhisperX](https://github.com/m-bain/whisperX)
  - [Resemblyzer](https://github.com/resemble-ai/Resemblyzer)

---

**Last Updated**: 2024  
**Version**: 2.0  
**License**: MIT
