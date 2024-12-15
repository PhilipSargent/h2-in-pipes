# 2024-12-15 Philip Sargent
all: run

run:
	git config --global core.quotePath false # enables display of unicode in filenames in git output
	uv run peng.py
	uv run moody.py

peng_z.png: peng.py
	uv run peng.py
  
moody_afzal.png: moody.py
	uv run moody.py
	
ruff: moody.py peng.py
	ruff check moody.py
	ruff peng.py

.PHONY: all run