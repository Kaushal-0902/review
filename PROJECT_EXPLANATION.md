# Speaker Diarization and Recognition System - Complete Explanation

## **Project Overview: Speaker Diarization and Recognition System**

This is a **Secure, Open-Set, Near End-to-End Speaker Diarization and Recognition (S-O-EEND-SDR)** system. It's a CLI-based pipeline that processes audio files to answer four key questions:

- **Who spoke?** (Speaker identification)
- **When did they speak?** (Timestamps)
- **What did they say?** (Transcription)
- **Is the speaker authorized?** (Open-set identification)

---

## **How It Works: The Pipeline**

The system runs in **6 sequential steps**:

### **Step 1: Audio Loading & Preprocessing**
- Loads the audio file using `librosa`
- Converts to 16kHz mono (required format for models)
- Basic audio format validation

**File:** `utils/audio_utils.py`

### **Step 2: Speaker Diarization** (`diarization/diarize.py`)
- Uses **pyannote.audio** (pretrained model from HuggingFace)
- Identifies **when** different speakers are talking
- Outputs: "SPEAKER_00 spoke from 0.0-2.5 seconds", "SPEAKER_01 spoke from 2.5-5.0 seconds", etc.
- **Important:** This only labels speakers as SPEAKER_00, SPEAKER_01, etc. - it doesn't identify WHO they are

**Technology:** pyannote/speaker-diarization-3.1 model

### **Step 3: Speech Recognition** (`asr/transcribe.py`)
- Uses **WhisperX** (OpenAI Whisper with word-level timestamps)
- Transcribes what was said
- Outputs: "Hello, how are you?" [0.0-2.5s], "I'm doing well" [2.5-5.0s]
- Provides text but **not** which speaker said it

**Technology:** OpenAI Whisper models (tiny/base/small/medium/large)

### **Step 4: Alignment** (`alignment/align.py`)
- **Critical step:** Combines diarization and ASR outputs
- Matches text segments to speakers based on timestamp overlap
- Merges adjacent segments from the same speaker
- Result: "SPEAKER_00: Hello, how are you? [0.0-2.5s]"

**Logic:** Calculates overlap ratios between ASR segments and diarization segments, assigns speaker labels based on best match

### **Step 5: Speaker Identification** (`embeddings/speaker_id.py`)
- Uses **Resemblyzer** to extract voice embeddings (voiceprints)
- Compares each segment's embedding with a registry of known speakers
- Uses cosine similarity (default threshold: 0.75)
- If match found: "Alice: Hello, how are you?"
- If no match: "UNKNOWN: Hello, how are you?"

**Technology:** Resemblyzer speaker encoder model

### **Step 6: Authorization Filter**
- Filters out segments from unauthorized (UNKNOWN) speakers
- Only keeps segments from registered speakers
- Can be disabled with `--no-auth-filter` flag

---

## **Key Components**

### **Speaker Registry** (`data/speaker_registry.json`)
- Stores voiceprints (embeddings) of known speakers
- JSON format with speaker names, embeddings, and metadata
- To add a speaker: `python scripts/enroll_speaker.py --name "Alice" --audio alice_sample.wav`
- The system compares new audio against this registry

### **Output Format**

**CLI Output:**
```
[0.00-2.50] Alice: Hello, how are you? (✓ 85%)
[2.50-5.00] Bob: I am doing well, thank you. (✓ 92%)
[5.00-7.50] UNKNOWN: Can you help me? (✗ UNAUTHORIZED 45%)
```

**JSON Output:**
```json
{
  "audio_file": "conversation.wav",
  "num_segments": 3,
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, how are you?",
      "speaker_label": "SPEAKER_00",
      "identified_speaker": "Alice",
      "similarity_score": 0.85,
      "spoof_check": {
        "is_genuine": true,
        "confidence": 1.0,
        "method": "placeholder"
      }
    }
  ]
}
```

---

## **Project Structure**

```
Speaker-Diarization-main/
├── main.py                    # Main CLI entry point - orchestrates entire pipeline
├── diarization/               # Who spoke when?
│   ├── __init__.py
│   └── diarize.py            # pyannote.audio integration
├── asr/                       # What did they say?
│   ├── __init__.py
│   └── transcribe.py         # WhisperX integration
├── alignment/                 # Combine diarization + ASR
│   ├── __init__.py
│   └── align.py              # Timestamp overlap matching
├── embeddings/                # Speaker identification
│   ├── __init__.py
│   └── speaker_id.py         # Resemblyzer + registry matching
├── utils/                     # Audio utilities
│   ├── __init__.py
│   └── audio_utils.py        # Audio loading, VAD, preprocessing
├── scripts/                   # Helper scripts
│   ├── enroll_speaker.py      # Add speakers to registry
│   ├── benchmark_auth.py       # Authorization testing
│   ├── benchmark_enhanced.py  # Enhanced benchmarking
│   ├── build_long_conversation.py  # Test data generation
│   └── setup_test_data.py    # Test data setup
├── tests/                     # Unit tests
│   ├── __init__.py
│   └── test_pipeline_integration.py
├── data/                      # Audio files & registry
│   ├── speaker_registry.json  # Known speakers database
│   ├── sample.wav             # Test audio
│   ├── long_conversation.wav  # Long test audio
│   ├── speakers/              # Reference speaker samples
│   │   ├── Alice/
│   │   ├── Bob/
│   │   └── ...
│   └── benchmark_enhanced/    # Benchmark test data
├── results/                   # Output JSON files
│   ├── test_results.json
│   └── benchmark_*.json
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## **Use Cases**

1. **Meeting Transcription**: Identify who said what in meetings
2. **Security/Access Control**: Only process audio from authorized speakers
3. **Call Center Analysis**: Track agent vs customer speech
4. **Podcast/Video Processing**: Generate speaker-attributed transcripts

---

## **Example Workflow**

### **1. Enroll a Known Speaker**
```bash
python scripts/enroll_speaker.py --name "Alice" --audio alice_voice.wav
```

This extracts Alice's voiceprint and saves it to `data/speaker_registry.json`.

### **2. Process an Audio File**
```bash
python main.py conversation.wav --output results.json
```

The pipeline will:
- Load and preprocess audio
- Run diarization (find speaker segments)
- Run ASR (transcribe speech)
- Align diarization with transcription
- Identify speakers using registry
- Filter unauthorized speakers
- Output results

### **3. View Results**
The output shows:
- Timestamps for each segment
- Speaker names (or UNKNOWN)
- Transcribed text
- Authorization status and similarity scores

---

## **Command-Line Usage**

### **Basic Usage**
```bash
python main.py path/to/audio.wav
```

### **Advanced Options**
```bash
# Save results to JSON
python main.py audio.wav --output results.json

# Use larger Whisper model for better accuracy
python main.py audio.wav --whisper-model large

# Adjust speaker identification threshold
python main.py audio.wav --speaker-threshold 0.8

# Provide HuggingFace token
python main.py audio.wav --hf-token YOUR_HF_TOKEN

# Disable authorization filter (include all speakers)
python main.py audio.wav --no-auth-filter

# Combine options
python main.py audio.wav --output results.json --whisper-model large --speaker-threshold 0.75
```

### **Command-Line Arguments**
- `audio_file`: Path to input audio file (required)
- `--output, -o`: Path to save JSON output file (optional)
- `--hf-token`: HuggingFace token for model access (optional, may be required)
- `--whisper-model`: Whisper model size: tiny, base, small, medium, large (default: base)
- `--speaker-threshold`: Similarity threshold for speaker identification, 0.0-1.0 (default: 0.75)
- `--merge-gap`: Maximum gap in seconds to merge adjacent segments (default: 0.5)
- `--no-auth-filter`: Disable speaker authorization filter (include all speakers)

---

## **Technical Stack**

- **PyTorch**: Deep learning framework
- **pyannote.audio**: Speaker diarization models
- **WhisperX**: Speech recognition with word-level timestamps
- **Resemblyzer**: Speaker embedding extraction
- **librosa**: Audio processing and loading
- **HuggingFace Hub**: Model hosting and access

---

## **Module Details**

### **Diarization Module** (`diarization/diarize.py`)
- Loads pretrained pyannote.audio pipeline
- Processes audio to find speaker segments
- Outputs: `[start, end, speaker_label]` tuples
- Requires HuggingFace token (user must accept model terms)

### **ASR Module** (`asr/transcribe.py`)
- Loads WhisperX model (configurable size)
- Transcribes audio with word-level timestamps
- Uses alignment model for precise timing
- Outputs: `[start, end, text]` tuples

### **Alignment Module** (`alignment/align.py`)
- Calculates overlap between ASR and diarization segments
- Assigns speaker labels to text segments
- Merges adjacent segments from same speaker
- Handles edge cases (no overlap, multiple overlaps)

### **Speaker ID Module** (`embeddings/speaker_id.py`)
- Extracts speaker embeddings using Resemblyzer
- Loads speaker registry from JSON
- Compares embeddings using cosine similarity
- Identifies speakers or marks as UNKNOWN
- Supports adding new speakers to registry

### **Audio Utils** (`utils/audio_utils.py`)
- Audio loading and format conversion
- Sample rate conversion (to 16kHz)
- Mono conversion
- Simple VAD (Voice Activity Detection) for silence removal

---

## **Data Flow**

```
Audio File (WAV/MP3)
    ↓
[Step 1] Load & Preprocess (16kHz mono)
    ↓
[Step 2] Diarization → Speaker Segments
    ↓                    ↓
[Step 3] ASR         Transcription Segments
    ↓                    ↓
[Step 4] Alignment → Aligned Segments (speaker + text)
    ↓
[Step 5] Speaker ID → Identified Segments (name + similarity)
    ↓
[Step 6] Auth Filter → Final Segments (authorized only)
    ↓
JSON Output / CLI Display
```

---

## **Speaker Registry Format**

The `data/speaker_registry.json` file stores known speakers:

```json
{
  "Alice": {
    "embedding": [0.123, -0.456, 0.789, ...],
    "metadata": {
      "source_audio": "alice_sample.wav"
    }
  },
  "Bob": {
    "embedding": [0.234, -0.567, 0.890, ...],
    "metadata": {
      "source_audio": "bob_sample.wav"
    }
  }
}
```

**To enroll a speaker:**
```bash
python scripts/enroll_speaker.py --name "Alice" --audio alice_sample.wav
```

**To list enrolled speakers:**
```bash
python scripts/enroll_speaker.py --list
```

---

## **Current Limitations**

1. **Anti-Spoofing**: Currently a placeholder that always returns "genuine". Future work needed to integrate pretrained anti-spoofing models.

2. **Speaker Registry**: Manual process to add known speakers. No automatic enrollment from diarization output.

3. **Overlapping Speech**: Basic handling of overlapping speech segments. May not perfectly handle complex overlaps.

4. **Language Support**: Currently configured for English. WhisperX supports multiple languages but alignment model is English-only.

5. **Model Access**: Requires HuggingFace account and token acceptance for pyannote.audio models.

6. **GPU Memory**: Large Whisper models require significant GPU memory. Use smaller models (base/small) if memory is limited.

---

## **Installation Requirements**

### **Prerequisites**
- Python 3.9 or 3.10
- CUDA-capable GPU (recommended, but CPU will work)
- Git

### **Setup Steps**
1. Clone or navigate to the project directory
2. Create virtual environment: `python -m venv venv`
3. Activate virtual environment: `source venv/bin/activate` (Windows: `venv\Scripts\activate`)
4. Install dependencies: `pip install -r requirements.txt`
5. Accept model terms on HuggingFace: https://huggingface.co/pyannote/speaker-diarization-3.1
6. Create HuggingFace token if needed: https://huggingface.co/settings/tokens

---

## **Troubleshooting**

### **"Failed to load diarization pipeline"**
- Ensure you've accepted model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
- Provide HuggingFace token with `--hf-token` flag

### **"CUDA out of memory"**
- Use smaller Whisper model: `--whisper-model tiny` or `--whisper-model base`
- Process shorter audio segments
- Use CPU mode (slower but uses less memory)

### **"No module named 'resemblyzer'"**
- Install dependencies: `pip install -r requirements.txt`
- If issues persist, resemblyzer may need additional setup

### **Poor transcription accuracy**
- Use larger Whisper model: `--whisper-model large`
- Ensure audio quality is good (16kHz, mono recommended)
- Check audio file format (WAV, MP3, etc.)

### **Speaker identification always returns "UNKNOWN"**
- Check that speaker registry exists and contains known speakers
- Lower the similarity threshold: `--speaker-threshold 0.5`
- Ensure audio segments are long enough (>0.5 seconds)

---

## **Future Work**

1. **Anti-Spoofing Integration**:
   - Integrate ASVspoof or AASIST pretrained models
   - Add confidence scores and detailed spoofing analysis

2. **Automatic Speaker Enrollment**:
   - Automatically add speakers to registry after manual verification
   - Support for speaker clustering and naming

3. **Improved Overlap Handling**:
   - Better handling of simultaneous speech
   - Multi-speaker transcription for overlapping segments

4. **Multi-Language Support**:
   - Language detection
   - Language-specific alignment models

5. **Performance Optimization**:
   - Batch processing for multiple files
   - Streaming support for real-time processing
   - Model quantization for faster inference

6. **Enhanced Output Formats**:
   - SRT subtitle file generation
   - RTTM format export
   - WebVTT format support

---

## **License**

This project is for educational/capstone purposes. Please check individual model licenses:
- pyannote.audio: MIT License
- WhisperX: MIT License
- Resemblyzer: MIT License

---

## **Acknowledgments**

- pyannote.audio team for speaker diarization models
- OpenAI for Whisper ASR models
- Resemblyzer for speaker embedding models
- Inspired by SpeakerLM and related research
