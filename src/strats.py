#!/usr/bin/env python3

import random

from enum import Enum
from typing import Callable, Dict, List


class Action(Enum):
    DEFECTING = 'Defecting'
    COOPERATING = 'Cooperating'


def complement_action(a: Action) -> Action:
    if a == Action.DEFECTING:
        return Action.COOPERATING
    else:
        return Action.DEFECTING


class Strategy:
    """
    prototype for action:
    ---------------------
      action(self, own_decisions: List[Action], opponent_decisions: List[Action], local_state: Dict) -> Action

    local_state can be used arbitrarily by the action and should remain
    persistent across all iterations in a game
    """
    def __init__(self,
                 name: str,
                 action: Callable[[List[Action],List[Action],Dict], Action]):
        assert action is not None
        self.name = name
        self.action = action


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
    return Strategy("suspicious-sophist", action)


def strategy_tit_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        if len(opponent_decisions) >= 1:
            return opponent_decisions[-1]
        return Action.COOPERATING
    return Strategy("tit-for-tat", action)


def strategy_suspicious_tit_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        if len(opponent_decisions) >= 1:
            return opponent_decisions[-1]
        return Action.DEFECTING
    return Strategy("suspicious-tit-for-tat", action)


def strategy_forgiving_tit_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        if len(opponent_decisions) >= 2 and opponent_decisions[-1] == opponent_decisions[-2]:
            return opponent_decisions[-1]
        return Action.COOPERATING
    return Strategy("forgiving-tit-for-tat", action)


def strategy_firm_but_fair() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        if len(own_decisions) == 0 or opponent_decisions[-1] == own_decisions[-1]:
            return Action.COOPERATING
        elif has_defection(local_state, opponent_decisions):
            return Action.DEFECTING
        else:
            return Action.COOPERATING
    return Strategy("firm-but-fair", action)


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
    return Strategy("suspicious-pavlov", action)


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
    return Strategy("suspicious-pavlovish", action)


def strategy_pavlov_spooky() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # switch strategy if opponent's action didn't match ours
        # otherwise keep doing whatever we did last time
        if len(opponent_decisions) >= 1 and opponent_decisions[-1] != own_decisions[-1]:
            return complement_action(own_decisions[-1])
        if len(own_decisions) > 0:
            return own_decisions[-1]
        return Action.COOPERATING
    return Strategy("pavlov-spooky", action)


def strategy_suspicious_pavlov_spooky() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        if len(opponent_decisions) >= 1 and opponent_decisions[-1] != own_decisions[-1]:
            return complement_action(own_decisions[-1])
        if len(own_decisions) > 0:
            return own_decisions[-1]
        return Action.DEFECTING
    return Strategy("suspicious-pavlov-spooky", action)


name2strategy = {
    "defector": strategy_defector(),
    "gandhi": strategy_gandhi(),
    "random": strategy_random(),
    "sophist": strategy_sophist(),
    "tit-for-tat": strategy_tit_for_tat(),
    "forgiving-tit-for-tat": strategy_forgiving_tit_for_tat(),
    "suspicious-tit-for-tat": strategy_suspicious_tit_for_tat(),
    "pavlov": strategy_pavlov(),
    "pavlovish": strategy_pavlovish(),
    "pavlov-spooky": strategy_pavlov_spooky(),
    "alternator": strategy_alternator(),
    "grudger": strategy_grudger(),
    "hate-opponent": strategy_hate_opponent(),
    "angry-grudger": strategy_angry_grudger(),
    "suspicious-sophist": strategy_suspicious_sophist(),
    "suspicious-pavlov": strategy_suspicious_pavlov(),
    "suspicious-pavlovish": strategy_suspicious_pavlovish(),
    "suspicious-pavlov-spooky": strategy_suspicious_pavlov_spooky(),
    "firm-but-fair": strategy_firm_but_fair(),
}
