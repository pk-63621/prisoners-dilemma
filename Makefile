SRCS := src

RUNNER := time python3

all: mypy flake8

mypy:
	mypy $(SRCS)

flake8:
	flake8 --max-line-length=150 --ignore=E201,E202,E126,E221,E226,E231,E241,E252,F405,F403 $(SRCS)

run:
	$(RUNNER) src/pd_tournament.py -q -i 1000

run_mp:
	$(RUNNER) src/pd_tournament.py -q -i 1000 -j

.PHONY: all mypy flake8 run run_mp
