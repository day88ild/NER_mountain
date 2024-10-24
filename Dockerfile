# Start from the official Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the dependencies specified in the requirements file
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --timeout=3000 -r requirements.txt

# Copy the rest of the project files into the container
COPY . .

# Run the inference script when the container starts
CMD ["python", "inference.py"]

