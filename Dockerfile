FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Railway will set the PORT environment variable)
EXPOSE $PORT

# Start command
CMD gunicorn -w 4 -b 0.0.0.0:$PORT app:app