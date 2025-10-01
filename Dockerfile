# Dockerfile

# 1. Base Image
FROM python:3.10-slim

# Prevent Python from writing .pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 2. Set working directory
WORKDIR /app

# 3. Install dependencies
# First copy only requirements to leverage Docker cache
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy project files
COPY . .

# 5. Expose port
EXPOSE 8000

# 6. Run FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
