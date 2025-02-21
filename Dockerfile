FROM python:3.9

# Set the working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Copy requirement files
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose the necessary port
EXPOSE 5000

# Run the application
CMD ["python", "app_yolo_norfair.py"]
