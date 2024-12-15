# 2024-04-01 Philip Sargent
all: run

run:
	git config --global core.quotePath false # enables display of unicode in filenames in git output
	python peng.py
	python moody.py

peng_z.png: peng.py
	python peng.py
  
moody_afzal.png: moody.py
	python moody.py

.PHONY: all run