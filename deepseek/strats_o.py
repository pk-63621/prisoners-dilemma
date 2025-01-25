#!/usr/bin/env python3

import random
import numpy as np
from typing import Dict, List

COOPERATING = 0
DEFECTING = 1

class Strategy:
    __slots__ = ('name', 'action')
    def __init__(self, name: str, action):
        self.name = name
        self.action = action

# Strategy implementations
def strategy_gandhi() -> Strategy:
    return Strategy("gandhi", lambda _, __, ___: COOPERATING)

def strategy_defector() -> Strategy:
    return Strategy("defector", lambda _, __, ___: DEFECTING)

def strategy_alternator() -> Strategy:
    def action(own_decisions: np.ndarray, *_):
        return 1 - own_decisions[-1] if own_decisions.size > 0 else COOPERATING
    return Strategy("alternator", action)

def strategy_grudger() -> Strategy:
    def action(_, opponent_decisions: np.ndarray, ls: Dict):
        if opponent_decisions.size == 0:
            return COOPERATING
        ls['has_def'] = ls.get('has_def', False) or (opponent_decisions[-1] == DEFECTING)
        return DEFECTING if ls['has_def'] else COOPERATING
    return Strategy("grudger", action)

def strategy_tit_for_tat() -> Strategy:
    def action(_, opponent_decisions: np.ndarray, __):
        return opponent_decisions[-1] if opponent_decisions.size > 0 else COOPERATING
    return Strategy("tit-for-tat", action)

def strategy_random() -> Strategy:
    return Strategy("random", lambda _, __, ___: random.choice([COOPERATING, DEFECTING]))

def strategy_pavlov() -> Strategy:
    def action(own_decisions: np.ndarray, opponent_decisions: np.ndarray, __):
        if own_decisions.size == 0:
            return COOPERATING
        return 1 - own_decisions[-1] if opponent_decisions[-1] == DEFECTING else own_decisions[-1]
    return Strategy("pavlov", action)

# ... Add other strategies following similar patterns ...

name2strategy = {
    "gandhi": strategy_gandhi(),
    "defector": strategy_defector(),
    "alternator": strategy_alternator(),
    "grudger": strategy_grudger(),
    "tit-for-tat": strategy_tit_for_tat(),
    "random": strategy_random(),
    "pavlov": strategy_pavlov(),
    # ... add other strategies ...
}

def all_strategies():
    return name2strategy.values()
