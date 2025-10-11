# GitHub Copilot Instructions

## Project Context

Wyoming OpenAI is a proxy middleware that bridges the Wyoming protocol with OpenAI-compatible endpoints for ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) services. It enables Wyoming clients like Home Assistant to use various OpenAI-compatible STT/TTS services.

## Code Style and Conventions

- Use async/await patterns for all I/O operations
- Follow Python type hints for function signatures
- Maintain consistency with existing error handling patterns
- Use logging for debugging and error messages
- Keep functions focused and modular

## Architecture Overview

### Core Components

- **`handler.py`**: Contains `OpenAIEventHandler` - the main Wyoming protocol event handler that processes ASR and TTS requests
- **`compatibility.py`**: Provides `CustomAsyncOpenAI` class with backend detection and OpenAI API compatibility layer
- **`__main__.py`**: Entry point with argument parsing and server initialization
- **`utilities.py`**: Helper functions for audio processing and data handling
- **`const.py`**: Version constants and configuration

### Key Patterns

1. **Async Event Handling**: Uses Wyoming's `AsyncEventHandler` to process incoming protocol events
2. **Backend Abstraction**: `CustomAsyncOpenAI` wraps different backends (OpenAI, Speaches, LocalAI, etc.) with a unified interface
3. **Stream Processing**: Handles both streaming and non-streaming transcription modes
4. **Audio Buffer Management**: Collects audio chunks into complete files for processing

### Wyoming Protocol Events

The handler processes these Wyoming events:
- `AudioStart/AudioChunk/AudioStop` → STT transcription
- `Transcribe` → Initiate transcription request  
- `Synthesize` → TTS audio generation

### Supported Backends

The `OpenAIBackend` enum defines supported backends:
- `OPENAI`: Official OpenAI API
- `SPEACHES`: Local Speaches service
- `LOCALAI`: LocalAI service
- `KOKORO_FASTAPI`: Kokoro TTS service

## Testing Guidelines

When writing tests:
- Use pytest fixtures for common setup
- Mock external API calls
- Test both success and error scenarios
- Include integration tests for end-to-end flows
- Aim for high code coverage

Test files are organized by module:
- `test_handler.py`: Event handler logic
- `test_compatibility.py`: Backend compatibility
- `test_utilities.py`: Helper functions
- `test_integration.py`: End-to-end scenarios

## Common Development Tasks

### Running Tests
```bash
pytest                              # Run all tests
pytest --cov=wyoming_openai        # With coverage
pytest tests/test_handler.py       # Specific test file
```

### Code Quality
```bash
ruff check .                       # Run linting
ruff check . --fix                 # Auto-fix issues
```

### Local Development
```bash
pip install -e ".[dev]"            # Install dev dependencies
python -m wyoming_openai --uri tcp://0.0.0.0:10300 --stt-models whisper-1 --tts-models tts-1
```

### Docker Development
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
```

## Configuration

The server accepts both command-line arguments and environment variables. When suggesting configuration changes, consider:
- STT/TTS API keys and URLs
- Model lists for STT and TTS
- Voice configurations
- Backend-specific settings (temperature, speed, etc.)

## When Making Changes

- Ensure backward compatibility with existing Wyoming clients
- Update tests to reflect new functionality
- Add appropriate logging for debugging
- Document new configuration options
- Consider impact on all supported backends
- Validate audio format conversions maintain quality
