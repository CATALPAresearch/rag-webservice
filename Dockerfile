# Use an official Python base image
FROM python:3.11-slim

# Set environment variables
ENV POETRY_VERSION=1.8.2 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /rag-webservice

# Install system dependencies
RUN apt-get update \
    && apt-get install -y curl build-essential \
    && apt-get clean

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s ~/.local/bin/poetry /usr/local/bin/poetry

# Copy project files
COPY pyproject.toml poetry.lock* /rag-webservice/

# Copy rest of the app
COPY . /rag-webservice


# Install project dependencies
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi
#RUN poetry config virtualenvs.create false \
#    && poetry install --no-interaction --no-ansi


# Expose port 5000
EXPOSE 5000

# Start the app (assumes Flask app in app.py or similar)
#CMD ["python", "app.py"]
CMD ["poetry", "run", "python", "src/app.py"]
