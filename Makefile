UID ?= $(shell id -u) # to set UID of container user to match host
GID ?= $(shell id -g) # to set GID of container user to match host
USER_PASSWORD ?= password # container user password (for sudo)

.PHONY: build
build:
	docker-compose stop
	docker-compose build --no-cache --build-arg UID=$(UID) --build-arg GID=$(GID) --build-arg USER_PASSWORD=$(USER_PASSWORD) pytorch-env

.PHONY: run
run:
	docker-compose stop
	docker-compose up -d pytorch-env

.PHONY: terminal
terminal:
	docker-compose run pytorch-env bash