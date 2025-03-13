FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone and install SAM2 (adjust the URL if needed)
# Note: If SAM2 is not publicly available, you'll need to copy your local SAM2 directory
# If using a private repository, you might need to handle authentication
RUN git clone https://github.com/your-sam2-repo-url.git sam2_repo || echo "SAM2 repo not available, will use local copy"

# If cloning fails, we'll use a local copy that should be included in the build context
COPY sam2/ /app/sam2/

# Add SAM2 to Python path
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Copy model files
COPY fine_tuned_sam2_2000.torch .
COPY sam2_hiera_small.pt .
COPY sam2_hiera_s.yaml .

# Copy the rest of the application
COPY . .

# Create directories for uploads and results
RUN mkdir -p uploads results

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]