# Use the NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set the working directory in the container
WORKDIR /poket money

# Install Python and necessary tools
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN python3 -m pip install --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt .

# Install required Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code to the working directory
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the Streamlit application
CMD ["streamlit", "run", "app.py"]
