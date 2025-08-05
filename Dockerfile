# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# No system dependencies needed - all Python packages have pre-compiled wheels
# Uncomment the following lines if you need to install system dependencies
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#         build-essential \
#         libssl-dev \
#     && rm -rf /var/lib/apt/lists/*

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

# Expose the application port
EXPOSE 10300

# Run the application as an installed module
CMD ["python", "-m", "wyoming_openai"]