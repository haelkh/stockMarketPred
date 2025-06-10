# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Upgrade pip and increase timeout for large packages like TensorFlow
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout=800 -r requirements.txt

# Copy all your application code into the container at /app
# This includes data_collector.py, StockMarketPredictor.py, and run_all.py
COPY . .

# Create base directories for data, models, and logs.
# These will be mapped to your host machine via docker-compose volumes.
RUN mkdir -p data models logs

# Define the command to run your orchestrator script
# This will execute run_all.py, which then calls data_collector.py and StockMarketPredictor.py
CMD ["python", "run_all.py"]

# No need to expose ports for a batch processing script
# EXPOSE 8000