# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (if any)
# build-essential and libssl-dev might be needed for some dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy files required for installation
# We need pyproject.toml for dependencies and package info
# README.md and LICENSE are referenced in pyproject.toml
# src contains the actual code
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Install python dependencies and the project itself using pyproject.toml
RUN pip install --upgrade pip && \
    pip install --no-cache-dir .

# Expose the application port (already correct)
EXPOSE 10300

# Run the application as an installed module (already correct)
CMD ["python", "-m", "wyoming_openai"]