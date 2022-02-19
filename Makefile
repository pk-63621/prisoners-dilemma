SRCS := pd.py

all: mypy

mypy:
	mypy $(SRCS)

flake8:
	flake8 --max-line-length=150 --ignore=E126,E221,E226,E231,E241 $(SRCS)

.PHONY: all mypy flake8
