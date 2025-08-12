# Base image
FROM --platform=linux/amd64 python:3.10-slim


# Set working directory
WORKDIR /app

# Copy project files
COPY app/ ./app/
COPY app/requirements.txt  ./

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose port Flask will run on
EXPOSE 5050

# Run the Flask app
CMD ["python", "app/app.py"]
