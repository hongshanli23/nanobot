# Local Qdrant Server

This directory contains a minimal Docker Compose setup for running Qdrant locally.

## Files

- `docker-compose.yml` — starts Qdrant and persists data under `./qdrant_storage`

## Start

```bash
cd ~/nanobot/qdrant
docker compose up -d
```

## Verify

```bash
docker ps | grep qdrant
curl -s http://localhost:6333/ | jq
```

## Logs

```bash
cd ~/nanobot/qdrant
docker compose logs -f qdrant
```

## Stop

```bash
cd ~/nanobot/qdrant
docker compose down
```

## Reset local data (optional)

```bash
cd ~/nanobot/qdrant
docker compose down
rm -rf qdrant_storage
```
