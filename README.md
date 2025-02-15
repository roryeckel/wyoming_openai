# Wyoming OpenAI

OpenAI-Compatible Proxy Middleware for the Wyoming Protocol

**Author:** Rory Eckel

Note: This project is not affiliated with OpenAI or the Wyoming project.

## Overview

This project introduces an OpenAI-compatible proxy server that integrates seamlessly with the [Wyoming](https://github.com/rhasspy/wyoming) framework. It provides transcription (Automatic Speech Recognition - ASR) and text-to-speech synthesis (TTS) capabilities using OpenAI-compatible APIs. By acting as a bridge between the Wyoming protocol and OpenAI's services, this proxy server enables efficient utilization of local ASR and TTS models. This is particularly advantageous for homelab users who aim to consolidate multiple protocols into a single server, thereby addressing resource constraints.

## Objectives

1. **Wyoming Server, OpenAI-compatible Client**: Function as an intermediary between the Wyoming protocol and OpenAI's ASR and TTS services.
2. **Service Consolidation**: Allow users of various programs to run inference on a single server without needing separate instances for each service.
Example: Sharing TTS/STT services between [Open WebUI](#open-webui) and [Home Assistant](#usage-in-home-assistant).
3. **Asynchronous Processing**: Enable efficient handling of multiple requests by supporting asynchronous processing of audio streams.

## Terminology

- **TTS (Text-to-Speech)**: The process of converting text into audible speech output.
- **ASR (Automatic Speech Recognition) / STT (Speech-to-Text)**: Technologies that convert spoken language into written text. ASR and STT are often used interchangeably to describe this function.

## Installation (Local Development)

### Prerequisites

- Tested with Python 3.12 or later
- Optional: OpenAI API key(s) if using proprietary models

### Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/roryeckel/wyoming-openai.git
   cd wyoming-openai
   ```

2. **Create a Virtual Environment** (optional but recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure OpenAI API Keys** (ensure you have your OpenAI API keys ready)

## Command Line Arguments

The proxy server can be configured using several command line arguments to tailor its behavior to your specific needs.

### Example Usage

```bash
python -m wyoming_openai \
    --uri tcp://0.0.0.0:10300 \
    --log-level INFO \
    --languages en \
    --stt-openai-key YOUR_STT_API_KEY_HERE \
    --stt-openai-url https://api.openai.com/v1 \
    --stt-models whisper-1 \
    --tts-openai-key YOUR_TTS_API_KEY_HERE \
    --tts-openai-url https://api.openai.com/v1 \
    --tts-models tts-1 \
    --tts-voices alloy echo fable onyx nova shimmer
```

## Environment Variables

In addition to using command-line arguments, you can configure the Wyoming OpenAI proxy server via environment variables. This is especially useful for containerized deployments.

### Table of Environment Variables for Command Line Arguments

| **Command Line Argument**               | **Environment Variable**                   | **Description**                                              |
|-----------------------------------------|--------------------------------------------|--------------------------------------------------------------|
| `--uri`                                 | `WYOMING_URI`                              | The URI for the Wyoming server to bind to.                   |
| `--log-level`                           | `WYOMING_LOG_LEVEL`                        | Sets the logging level (e.g., INFO, DEBUG).                  |
| `--languages`                           | `WYOMING_LANGUAGES`                        | Space-separated list of supported languages to avertise.     |
| `--stt-openai-key`                      | `STT_OPENAI_KEY`                           | The API key for accessing OpenAI's speech-to-text services.  |
| `--stt-openai-url`                      | `STT_OPENAI_URL`                           | The URL for OpenAI's STT endpoint.                           |
| `--stt-models`                          | `STT_MODELS`                               | Space-separated list of models to use for the STT service.   |
| `--tts-openai-key`                      | `TTS_OPENAI_KEY`                           | The API key for accessing OpenAI's text-to-speech services.  |
| `--tts-openai-url`                      | `TTS_OPENAI_URL`                           | The URL for OpenAI's TTS endpoint.                           |
| `--tts-models`                          | `TTS_MODELS`                               | Space-separated list of models to use for the TTS service.   |
| `--tts-voices`                          | `TTS_VOICES`                               | Space-separated list of voices for TTS, default is automatic |

## Docker (Recommended)

### Prerequisites

- Ensure you have [Docker](https://www.docker.com/products/docker-desktop) and [Docker Compose](https://docs.docker.com/compose/install/) installed on your system.

### Deployment Options

You can deploy the Wyoming OpenAI proxy server in different environments depending on whether you are using official OpenAI services or a local alternative like Speaches. You can even run multiple wyoming_openai instances on different ports for different purposes. Below are example scenarios:

#### 1. Deploying with Official OpenAI Services

To set up the Wyoming OpenAI proxy to work with official OpenAI APIs, follow these steps:

- **Environment Variables**: Create a `.env` file in your project directory that includes necessary environment variables such as `STT_OPENAI_KEY`, `TTS_OPENAI_KEY`.

- **Docker Compose Configuration**: Use the provided `docker-compose.yml` template. This setup binds a Wyoming server to port 10300 and uses environment variables for OpenAI URLs, model configurations, and voices as specified in the compose file.

- **Command**:
  
  ```bash
  docker-compose -f docker-compose.yml up -d
  ```

#### 2. Deploying with Speaches Local Service

If you prefer using a local service like Speaches instead of official OpenAI services, follow these instructions:

- **Docker Compose Configuration**: Use the `docker-compose.speaches.yml` template which includes configuration for both the Wyoming OpenAI proxy and the Speaches service.

- **Speaches Setup**:
  - The Speaches container is configured with specific model settings (`Systran/faster-distil-whisper-large-v3` for STT and `hexgrad/Kokoro-82M` for TTS).
  - It uses a local port (8000) to expose the Speaches service.
  - NVIDIA GPU support is enabled, so ensure your system has an appropriate setup if you plan to utilize GPU resources.

- **Command**:
  
  ```bash
  docker-compose -f docker-compose.speaches.yml up -d
  ```

#### 3. Deploying with FastAPI-Kokoro and Speaches Local Services

For users preferring a setup that leverages FastAPI-Kokoro for TTS and Speaches for STT, follow these instructions:

- **Docker Compose Configuration**: Use the `docker-compose.fastapi-kokoro.yml` template which includes configuration for both the Wyoming OpenAI proxy and FastAPI-Kokoro TTS service (Kokoro).

- **Speaches Setup**:
  - Use it in combination with the Speaches container for access to STT.

- **Kokoro Setup**:
  - The FastAPI-Kokoro container provides TTS capabilities.
  - It uses a local port (8880) to expose the Kokoro service.
  - NVIDIA GPU support is enabled, so ensure your system has an appropriate setup if you plan to utilize GPU resources.

- **Command**:

  ```bash
  docker-compose -f docker-compose.speaches.yml -f docker-compose.fastapi-kokoro.yml up -d
  ```

#### 4. Development with Docker

If you are developing the Wyoming OpenAI proxy server and want to build it from source, use the `docker-compose.dev.yml` file along with the base configuration.

- **Command**:
  
  ```bash
  docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
  ```

#### 5. Example: Development with Additional Local Service

For a development setup using the Speaches local service, combine `docker-compose.speaches.yml` and `docker-compose.dev.yml`. This also works for `docker-compose.fastapi-kokoro.yml`.

- **Command**:
  
  ```bash
  docker-compose -f docker-compose.speaches.yml -f docker-compose.dev.yml up -d --build
  ```

#### 6. Docker Tags

We follow specific tagging conventions for our Docker images. These tags help in identifying the version and branch of the code that a particular Docker image is based on.

- **`latest`**: This tag always points to the latest stable release of the Wyoming OpenAI proxy server. It is recommended for users who want to run the most recent, well-tested version without worrying about specific versions.

- **`main`**: This tag points to the latest commit on the main code branch. It is suitable for users who want to experiment with the most up-to-date features and changes, but may include unstable or experimental code.

- **`version`**: Specific version tags (e.g., `0.1.0`) correspond to stable releases of the Wyoming OpenAI proxy server. These tags are ideal for users who need a consistent, reproducible environment and want to avoid breaking changes introduced in newer versions.

- **`major.minor version`**: Tags that follow the `major.minor` format (e.g., `0.1`) represent a range of patch-level updates within the same minor version series. These tags are useful for users who want to stay updated with bug fixes and minor improvements without upgrading to a new major or minor version.

### General Deployment Steps

1. **Start Services**: Run the appropriate Docker Compose command based on your deployment option.
2. **Verify Deployment**: Ensure that all services are running by checking the logs with `docker-compose logs -f` or accessing the Wyoming OpenAI proxy through its exposed port (e.g., 10300) to ensure it responds as expected.
3. **Configuration Changes**: You can modify environment variables in the `.env` file or directly within your Docker Compose configuration files to adjust settings such as languages, models, and voices without rebuilding containers.

### Usage in Home Assistant

1. Install & set up your Wyoming OpenAI instance using one of the [deployment options](#deployment-options) above.
2. In HA, Go to Settings, Devices & Services, Add Integration, and search for Wyoming Protocol. Add the Wyoming Protocol integration with the URI of your Wyoming OpenAI instance.
3. The hard part is over! Configure your Voice Assistant pipeline to use the STT/TTS services provided by your new Wyoming OpenAI instance.

### Sequence Diagrams

#### Home Assistant

Home Assistant uses the Wyoming Protocol integration to communicate with the Wyoming OpenAI proxy server. The proxy server then communicates with the OpenAI API to perform the requested ASR or TTS tasks. The results are then sent back to Home Assistant.

[![](https://mermaid.ink/img/pako:eNqdk01v2zAMhv8KoVOKJtjdhwLqhiI9bB1mo0EGA4UqM4lQW1QlOf1C__uoKIm9pb3MJ0t8KL18Kb4JTQ2KQgR87NFq_GbU2quutsCfUz4abZyyEeYSVIA5dQgyBBMib55Si2Winl6oM3Z9Rw6tMqfUjfx5nbgbjstr4FVtM_WDIgJt0fN904QVUDpEvZlFmlX4HGFSVtUXWf46g6uWnnLWXM4uLhbLAiqvbNDe3CPgFg8Cj2HZN4bKyErG4ZbI5RCU0aNK0nPkNPnrprcPOTkMzKA6gZf9aoU-wELeQqOiyhza5mM1fPlIzGLJ0X3hnAKaOtciH692AlemxQwmZnYoOxkT97W7CB6DIxvweCRzcznY4_6q_1Pn07HJ99wBmFRV-aHr5YuNGwzmde86TGISdA5bMhrPTgvL54V9WmC9_PjC2IGk9n-6NXYlQ2EHgU6NG7VsMOWTvh779a-eQ7_EVHToO2UaHp-3xNaC6-mwFgX_Nso_1KK278ypPhKbpEURfY9T4alfb0SxUm3gVe_4lRwG77jLg_KbaFhjYyL573lad0P7_gfoyTL1?type=png)](https://mermaid-js.github.io/mermaid-live-editor/edit#pako:eNqdk01v2zAMhv8KoVOKJtjdhwLqhiI9bB1mo0EGA4UqM4lQW1QlOf1C__uoKIm9pb3MJ0t8KL18Kb4JTQ2KQgR87NFq_GbU2quutsCfUz4abZyyEeYSVIA5dQgyBBMib55Si2Winl6oM3Z9Rw6tMqfUjfx5nbgbjstr4FVtM_WDIgJt0fN904QVUDpEvZlFmlX4HGFSVtUXWf46g6uWnnLWXM4uLhbLAiqvbNDe3CPgFg8Cj2HZN4bKyErG4ZbI5RCU0aNK0nPkNPnrprcPOTkMzKA6gZf9aoU-wELeQqOiyhza5mM1fPlIzGLJ0X3hnAKaOtciH692AlemxQwmZnYoOxkT97W7CB6DIxvweCRzcznY4_6q_1Pn07HJ99wBmFRV-aHr5YuNGwzmde86TGISdA5bMhrPTgvL54V9WmC9_PjC2IGk9n-6NXYlQ2EHgU6NG7VsMOWTvh779a-eQ7_EVHToO2UaHp-3xNaC6-mwFgX_Nso_1KK278ypPhKbpEURfY9T4alfb0SxUm3gVe_4lRwG77jLg_KbaFhjYyL573lad0P7_gfoyTL1)

#### Open WebUI

No proxy is needed for Open WebUI, because it has native support for OpenAI-compatible endpoints.

[![](https://mermaid.ink/img/pako:eNp9klFLwzAQx7_KkacJK773YVAYQh90YisF6UtMzzW45uLlqo6x727aOAfizFMu_P7HD_45KEMdqlwFfBvRGVxbvWU9tA7i8ZrFGuu1E9g0oANsPDpo8Pmx_IMo7ssTU5QQp9Yl6o4Egd6R45blhOVQeUTTZ0JZjZ8Ci6qur4vq4QpudvSRUpsmW60SvbaMRkCPnSUQ1i4Ytl4sOeBJPMh3JNLZlGpymPee2QgGTy5gIi-KTbFJKwnCoq6r_6VCAsPeSY_BhstGxawfhFEPv3TUUg3Ig7Zd7OIwvbUqrhuwVXm8dppfW9W6Y-T0KFTtnVG58IhLxTRue5W_6F2I0-g7LacWf15jP09E5xk7K8S3qfr5Bxy_AGmJqvg?type=png)](https://mermaid-js.github.io/mermaid-live-editor/edit#pako:eNp9klFLwzAQx7_KkacJK773YVAYQh90YisF6UtMzzW45uLlqo6x727aOAfizFMu_P7HD_45KEMdqlwFfBvRGVxbvWU9tA7i8ZrFGuu1E9g0oANsPDpo8Pmx_IMo7ssTU5QQp9Yl6o4Egd6R45blhOVQeUTTZ0JZjZ8Ci6qur4vq4QpudvSRUpsmW60SvbaMRkCPnSUQ1i4Ytl4sOeBJPMh3JNLZlGpymPee2QgGTy5gIi-KTbFJKwnCoq6r_6VCAsPeSY_BhstGxawfhFEPv3TUUg3Ig7Zd7OIwvbUqrhuwVXm8dppfW9W6Y-T0KFTtnVG58IhLxTRue5W_6F2I0-g7LacWf15jP09E5xk7K8S3qfr5Bxy_AGmJqvg)

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests. For major changes, please first discuss the proposed changes in an issue.

### Future Plans (Descending Priority)

- Improved streaming support directly to OpenAI APIs
- Reverse direction support (Server for OpenAI compatible endpoints - possibly FastAPI)
- OpenAI Realtime API
