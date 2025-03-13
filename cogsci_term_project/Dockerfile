# Use a base image with build tools
FROM python:3.12.7-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m pip install --upgrade pymupdf

RUN python -m venv venv
RUN . venv/bin/activate

# Copy the rest of the application code into the container
COPY static /app/static
COPY .env /app
COPY mistral_app.py /app

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application using Chainlit
CMD ["chainlit", "run", "mistral_app.py", "-w", "--host", "0.0.0.0", "--port", "8080"]
