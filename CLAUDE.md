# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wyoming OpenAI is a proxy middleware that bridges the Wyoming protocol with OpenAI-compatible endpoints for ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) services. It enables Wyoming clients like Home Assistant to use various OpenAI-compatible STT/TTS services.

## Development Commands

### Testing
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run tests with coverage
pytest --cov=wyoming_openai

# Run specific test file
pytest tests/test_handler.py
```

### Code Quality
```bash
# Run linting with Ruff
ruff check .

# Auto-fix linting issues
ruff check . --fix
```

### Local Development Setup
```bash
# Install in editable mode
pip install -e .

# Run the server locally
python -m wyoming_openai --uri tcp://0.0.0.0:10300 --stt-models whisper-1 --tts-models tts-1
```

### Docker Development
```bash
# Build and run development container
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build

# With local services (e.g., Speaches)
docker compose -f docker-compose.speaches.yml -f docker-compose.dev.yml up -d --build
```

## Architecture

### Core Components

- **`handler.py`**: Contains `OpenAIEventHandler` - the main Wyoming protocol event handler that processes ASR and TTS requests
- **`compatibility.py`**: Provides `CustomAsyncOpenAI` class with backend detection and OpenAI API compatibility layer
- **`__main__.py`**: Entry point with argument parsing and server initialization
- **`utilities.py`**: Helper functions for audio processing and data handling
- **`const.py`**: Version constants and configuration

### Key Architecture Patterns

1. **Async Event Handling**: Uses Wyoming's `AsyncEventHandler` to process incoming protocol events
2. **Backend Abstraction**: `CustomAsyncOpenAI` wraps different backends (OpenAI, Speaches, LocalAI, etc.) with a unified interface
3. **Stream Processing**: Handles both streaming and non-streaming transcription modes
4. **Audio Buffer Management**: Collects audio chunks into complete files for processing

### Wyoming Protocol Flow

The handler processes these Wyoming events:
- `AudioStart/AudioChunk/AudioStop` → STT transcription
- `Transcribe` → Initiate transcription request  
- `Synthesize` → TTS audio generation

### Backend Support

The `OpenAIBackend` enum defines supported backends:
- `OPENAI`: Official OpenAI API
- `SPEACHES`: Local Speaches service
- `LOCALAI`: LocalAI service
- `KOKORO_FASTAPI`: Kokoro TTS service

## Configuration

The server accepts both command-line arguments and environment variables. Key configuration includes:
- STT/TTS API keys and URLs
- Model lists for STT and TTS
- Voice configurations
- Backend-specific settings (temperature, speed, etc.)

## Testing Strategy

Tests are organized by module:
- `test_handler.py`: Event handler logic
- `test_compatibility.py`: Backend compatibility
- `test_utilities.py`: Helper functions
- `test_integration.py`: End-to-end scenarios