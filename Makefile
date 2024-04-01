# 2024-04-01 Philip Sargent
all: run

run:
	python peng.py
	python moody.py

peng_z.png: peng.py
	python peng.py
  
moody_afzal.png: moody.py
	python moody.py

.PHONY: all run