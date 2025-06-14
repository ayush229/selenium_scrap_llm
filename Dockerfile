# Use official Python image
FROM python:3.11-slim

# Avoid prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install system deps & Playwright
RUN apt-get update && apt-get install -y \
    curl unzip gnupg ca-certificates fonts-liberation libnss3 libatk-bridge2.0-0 libgtk-3-0 libxss1 libasound2 \
    && pip install --upgrade pip

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN pip install playwright && python -m playwright install --with-deps

# Copy app code
COPY . /app
WORKDIR /app

# Expose port (must match fly.toml)
EXPOSE 8000

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
