# Docker Setup

Run SignalForge in a Docker container for consistent, isolated development and deployment.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) (optional, for easier commands)

## Quick Start

### Option 1: Docker Compose (Recommended)

Simplest way to get running:

```bash
docker-compose up
```

The app will be available at **http://localhost:5000**

Stop with `Ctrl+C`, or in another terminal:
```bash
docker-compose down
```

### Option 2: Docker Only

Build the image:
```bash
docker build -t signalforge .
```

Run the container:
```bash
docker run -p 5000:5000 \
  -v $(pwd)/model_weights.pth:/app/model_weights.pth \
  -v $(pwd)/label_encoder.pkl:/app/label_encoder.pkl \
  signalforge
```

## Configuration

### Environment Variables

Pass env vars via docker-compose or docker run:

```bash
# Force model retraining
docker-compose up -e RETRAIN=1

# Or with docker run
docker run -p 5000:5000 -e RETRAIN=1 signalforge
```

Edit `docker-compose.yml` to set defaults:

```yaml
environment:
  - RETRAIN=1
  - FLASK_DEBUG=1
```

## Development

### Live Code Reloading

`docker-compose.yml` mounts the current directory, so code changes reload instantly.

Edit a file, save, and refresh the browser.

### Running Tests

```bash
docker-compose exec signalforge pytest test_arnn_model.py test_app.py -v
```

Or build a test image:

```bash
docker run signalforge pytest test_arnn_model.py test_app.py
```

### Accessing Logs

```bash
docker-compose logs -f signalforge
```

## Production Deployment

### Push to Docker Hub

```bash
docker tag signalforge:latest your-username/signalforge:latest
docker push your-username/signalforge:latest
```

### Deploy to Cloud

**AWS ECS:**
```bash
aws ecs run-task --cluster my-cluster --task-definition signalforge:1
```

**Google Cloud Run:**
```bash
gcloud run deploy signalforge --image gcr.io/my-project/signalforge
```

**Heroku:**
```bash
heroku container:push web
heroku container:release web
```

## Troubleshooting

### Port already in use

Change the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8080:5000"  # Maps container 5000 → host 8080
```

### Model files not persisting

Ensure volumes are mounted correctly. The compose file auto-mounts:
- `model_weights.pth`
- `label_encoder.pkl`

### Memory issues

If training is slow or crashes, increase Docker memory allocation in Docker Desktop settings.

## Image Size

Current image size: ~2.5GB (PyTorch + transformers are large)

To reduce:
- Use `python:3.11-slim` (done) vs `python:3.11` (would be +500MB)
- Exclude test files and docs in `.dockerignore` (done)
