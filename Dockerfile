FROM python:3.8.6

RUN apt-get update && apt-get install -y \
	python3-pip software-properties-common wget && \
	rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install production dependencies.
COPY requirements-release.txt requirements.txt
RUN pip install -r requirements.txt

# Copy local code to the container image.
COPY src ./src

# Copy the models folder
COPY release ./release

# Set default port.
ENV PORT 8020

# Run the web service using uvicorn.
CMD uvicorn --host 0.0.0.0 --port $PORT --workers 1 --app-dir src serve:app --reload 
