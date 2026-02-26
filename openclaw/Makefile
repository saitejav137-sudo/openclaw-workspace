.PHONY: help build up down restart logs clean ps

help:
	@echo "OpenClaw Docker Commands"
	@echo "======================="
	@echo "make build      - Build Docker image"
	@echo "make up         - Start all services"
	@echo "make down       - Stop all services"
	@echo "make restart    - Restart services"
	@echo "make logs       - View logs"
	@echo "make ps         - Show running containers"
	@echo "make clean      - Remove containers and volumes"
	@echo ""
	@echo "Optional profiles:"
	@echo "make up-db      - Start with PostgreSQL"
	@echo "make up-cache   - Start with Redis"
	@echo "make up-monitor - Start with monitoring"

build:
	docker-compose build

up:
	docker-compose up -d

up-db:
	docker-compose --profile database up -d

up-cache:
	docker-compose --profile cache up -d

up-monitor:
	docker-compose --profile monitoring up -d

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

ps:
	docker-compose ps

clean:
	docker-compose down -v
	docker system prune -f

shell:
	docker-compose exec openclaw bash

# Development commands
dev:
	docker-compose up

dev-db:
	docker-compose --profile database up
