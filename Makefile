requirements:
	poetry export -f requirements.txt --output requirements.txt --without-hashes
	poetry export -f requirements.txt --output requirements-dev.txt --without-hashes --dev
	poetry export -f requirements.txt --output requirements-all.txt --without-hashes -E all

update:
	poetry update

sync: update requirements
