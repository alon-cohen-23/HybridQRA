# Use Python 3.11 as base image
FROM python:3.11

# Set working directory
WORKDIR /app

RUN pip install --upgrade pip 

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

   
# Copy requirements files
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY config.yaml ./
COPY qdrant_collections.json ./

# Create directories for data if they don't exist
RUN mkdir -p data

# Expose the port your application uses
EXPOSE 5002

# Command to run the application
CMD ["python", "src/app.py"]

