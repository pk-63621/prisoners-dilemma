#!/usr/bin/env python3

import random

from enum import Enum
from collections import defaultdict
from typing import Callable, DefaultDict, Dict, List, Tuple, TypeVar

T = TypeVar('T')


class Action(Enum):
    DEFECTING = 'Defecting'
    COOPERATING = 'Cooperating'


def complement_action(a: Action) -> Action:
    if a == Action.DEFECTING:
        return Action.COOPERATING
    else:
        return Action.DEFECTING


class Strategy:
    def __init__(self, name, action):
        self.name: str = name
        self.action: Callable = action

    def action(self,
               own_decisions: List[Action],
               opponent_decisions: List[Action],
               local_state: Dict) -> Action:
        return self.action(own_decisions, opponent_decisions, local_state)


class StrategyResults:
    def __init__(self):
        self.dict: DefaultDict[str,List[int]] = defaultdict(list)
        self.cache_invalidated = False
        self.sorted_items: List[Tuple[str,List[int]]] = []

    def add_score(self, name: str, score: int):
        self.dict[name].append(score)
        self.cache_invalidated = True

    def get_sorted_items(self) -> List[Tuple[str,List[int]]]:
        if self.cache_invalidated:
            self.sorted_items = sorted(self.dict.items(), key=lambda p: sum(p[1]))
            self.cache_invalidated = False
        return self.sorted_items

    def get_best_strategies_and_score(self) -> Tuple[List[str],int]:
        sorted_items = self.get_sorted_items()
        best_strats, best_score = [], 0
        for strat,sl in sorted_items:
            total_score = sum(sl)
            if best_score < total_score:
                best_score = total_score
                best_strats = [strat]
            elif best_score == total_score:
                best_strats.append(strat)
        return best_strats, best_score

    def print(self):
        print()
        print("Strategy wise result")
        print("--------------------")
        for strat,sl in self.get_sorted_items():
            print("Strategy: {0:30} Count: {1:<10} Score: {2:10}".format(strat, len(sl), sum(sl)))
        print("--------------------")
        print()


# strategies -- add to name2strategy if adding new strategy

def strategy_defector() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        return Action.DEFECTING
    return Strategy("defector", action)


def strategy_alternator() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        return complement_action(own_decisions[-1]) if len(own_decisions) > 0 else Action.COOPERATING
    return Strategy("alternator", action)


def strategy_hate_opponent() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        return complement_action(opponent_decisions[-1]) if len(opponent_decisions) > 0 else Action.DEFECTING
    return Strategy("hate opponent", action)


def has_defection(local_state, opponent_decisions):
    has_def = False
    if 'has_def' in local_state:
        has_def = local_state['has_def'] or (len(opponent_decisions) >= 1 and opponent_decisions[-1] == Action.DEFECTING)
    else:
        has_def = Action.DEFECTING in opponent_decisions
    local_state['has_def'] = has_def
    return has_def


def strategy_grudger() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        if has_defection(local_state, opponent_decisions):
            return Action.DEFECTING
        return Action.COOPERATING
    return Strategy("grudger", action)


def strategy_angry_grudger() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        if len(opponent_decisions) == 0:
            return Action.DEFECTING
        if has_defection(local_state, opponent_decisions):
            return Action.DEFECTING
        return Action.COOPERATING
    return Strategy("angry grudger", action)


def strategy_gandhi() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        return Action.COOPERATING
    return Strategy("gandhi", action)


def strategy_random() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        return random.choice(list(Action))
    return Strategy("random", action)


def get_coop_and_defect_count(local_state, opponent_decisions):
    cnt_def, cnt_coop = 0, 0
    if 'cnt_def' in local_state:
        assert 'cnt_coop' in local_state
        assert len(opponent_decisions) >= 1
        last_defecting = opponent_decisions[-1] == Action.DEFECTING
        cnt_def = local_state['cnt_def'] + (1 if last_defecting else 0)
        cnt_coop = local_state['cnt_coop'] + (0 if last_defecting else 1)
    else:
        for d in opponent_decisions:
            if d == Action.DEFECTING:
                cnt_def += 1
            else:
                cnt_coop += 1
    local_state['cnt_def'] = cnt_def
    local_state['cnt_coop'] = cnt_coop
    return cnt_coop, cnt_def


def strategy_sophist() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        cnt_coop, cnt_def = get_coop_and_defect_count(local_state, opponent_decisions)
        if cnt_def > cnt_coop:
            return Action.DEFECTING
        return Action.COOPERATING
    return Strategy("sophist", action)


def strategy_suspicious_sophist() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        if len(opponent_decisions) == 0:
            return Action.DEFECTING
        cnt_coop, cnt_def = get_coop_and_defect_count(local_state, opponent_decisions)
        if cnt_def >= cnt_coop:
            return Action.DEFECTING
        return Action.COOPERATING
    return Strategy("suspicious sophist", action)


def strategy_tit_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        if len(opponent_decisions) >= 1:
            return opponent_decisions[-1]
        return Action.COOPERATING
    return Strategy("tit for tat", action)


def strategy_suspicious_tit_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        if len(opponent_decisions) >= 1:
            return opponent_decisions[-1]
        return Action.DEFECTING
    return Strategy("suspicious tit for tat", action)


def strategy_forgiving_tit_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        if len(opponent_decisions) >= 2 and opponent_decisions[-1] == opponent_decisions[-2]:
            return opponent_decisions[-1]
        return Action.COOPERATING
    return Strategy("forgiving tit for tat", action)


def strategy_pavlov() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # switch strategy if opponent defected
        # otherwise keep doing whatever we did last time
        if len(opponent_decisions) >= 1 and opponent_decisions[-1] == Action.DEFECTING:
            return complement_action(own_decisions[-1])
        if len(own_decisions) > 0:
            return own_decisions[-1]
        return Action.COOPERATING
    return Strategy("pavlov", action)


def strategy_suspicious_pavlov() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # switch strategy if opponent defected
        # otherwise keep doing whatever we did last time
        if len(opponent_decisions) >= 1 and opponent_decisions[-1] == Action.DEFECTING:
            return complement_action(own_decisions[-1])
        if len(own_decisions) > 0:
            return own_decisions[-1]
        return Action.DEFECTING
    return Strategy("suspicious pavlov", action)


def strategy_pavlovish() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # defect if opponent defected and we didn't
        # otherwise keep doing whatever we did last time
        if len(opponent_decisions) >= 1 and opponent_decisions[-1] == Action.DEFECTING and own_decisions[-1] == Action.COOPERATING:
            return Action.DEFECTING
        if len(own_decisions) > 0:
            return own_decisions[-1]
        return Action.COOPERATING
    return Strategy("pavlovish", action)


def strategy_suspicious_pavlovish() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        if len(opponent_decisions) >= 1 and opponent_decisions[-1] == Action.DEFECTING and own_decisions[-1] == Action.COOPERATING:
            return Action.DEFECTING
        if len(own_decisions) > 0:
            return own_decisions[-1]
        return Action.DEFECTING
    return Strategy("suspicious pavlovish", action)


def strategy_pavlov_spooky() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # switch strategy if opponent's action didn't match ours
        # otherwise keep doing whatever we did last time
        if len(opponent_decisions) >= 1 and opponent_decisions[-1] != own_decisions[-1]:
            return complement_action(own_decisions[-1])
        if len(own_decisions) > 0:
            return own_decisions[-1]
        return Action.COOPERATING
    return Strategy("pavlov spooky", action)
