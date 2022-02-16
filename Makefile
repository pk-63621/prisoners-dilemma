SRCS := pd.py

all: mypy

mypy:
	mypy $(SRCS)

.PHONY: all mypy
