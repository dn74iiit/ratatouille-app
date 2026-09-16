# Use an official Python runtime as a parent image (slim version for smaller size)
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (FAISS often needs some C++ build tools, but faiss-cpu wheels usually work out of the box on Debian. We'll add libgomp1 just in case)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# IMPORTANT: Ensure your data/ directory (containing recipes.index) is copied!
COPY . .

# Expose the port the app runs on (Render provides $PORT)
EXPOSE 10000

# Command to run the application using Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "10000"]
