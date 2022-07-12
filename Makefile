help:
	@echo "available commands"
	@echo " - install		: installs default requirements with pip"
	@echo " - install-all	: installs all requirements with pip"
	@echo " - dev			: installs default and development requirements with pip"
	@echo " - dev-all		: installs all and development requirements with pip"
	@echo " - sync			: export poetry dependencies as pip dev, all and default requirements"
	@echo " - update		: update poetry dependencies and sync requirements"

install:
	pip install -r requirements.txt

install-all:
	pip install -r requirements-all.txt

dev: install
	pip install -r requirements-dev.txt

dev-all: install-all
	pip install -r requirements-dev.txt

sync:
	poetry export -f requirements.txt --output requirements.txt --without-hashes
	poetry export -f requirements.txt --output requirements-dev.txt --without-hashes --dev
	poetry export -f requirements.txt --output requirements-all.txt --without-hashes -E all

up:
	poetry update

update: up sync
