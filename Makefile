SRCS := pd.py

all: mypy

mypy:
	mypy $(SRCS)

flake8:
	flake8 --max-line-length=150 $(SRCS)

.PHONY: all mypy flake8
