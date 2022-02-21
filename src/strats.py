#!/usr/bin/env python3

import random

from enum import Enum
from typing import Callable, Dict, List


class Action(Enum):
    DEFECTING, COOPERATING = 'Defecting', 'Cooperating'


def complement_action(a: Action, suspicious: bool = False) -> Action:
    return a if suspicious else (Action.COOPERATING if a == Action.DEFECTING else Action.DEFECTING)


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
        return Action.DEFECTING if has_defection(local_state, opponent_decisions) else Action.COOPERATING
    return Strategy("grudger", action)


def strategy_angry_grudger() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        if len(opponent_decisions) == 0:
            return Action.DEFECTING
        return Action.DEFECTING if has_defection(local_state, opponent_decisions) else Action.COOPERATING
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
    local_state['cnt_def'], local_state['cnt_coop'] = cnt_def, cnt_coop
    return cnt_coop, cnt_def


def strategy_sophist(suspicious=False) -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        cnt_coop, cnt_def = get_coop_and_defect_count(local_state, opponent_decisions)
        return Action.DEFECTING if cnt_def > cnt_coop else complement_action(Action.COOPERATING, suspicious)
    return Strategy("suspicious-sophist", action) if suspicious else Strategy("sophist", action)


def strategy_tit_for_tat(suspicious=False) -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        return opponent_decisions[-1] if len(opponent_decisions) >= 1 else complement_action(Action.COOPERATING, suspicious)
    return Strategy("suspicious-tit-for-tat", action) if suspicious else Strategy("tit-for-tat", action)

def strategy_forgiving_tit_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        return opponent_decisions[-1] if len(opponent_decisions) >= 2 and opponent_decisions[-1] == opponent_decisions[-2] else Action.COOPERATING
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


def strategy_pavlov(suspicious=False) -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # switch strategy if opponent defected
        # otherwise keep doing whatever we did last time
        if len(opponent_decisions) >= 1 and opponent_decisions[-1] == Action.DEFECTING:
            return complement_action(own_decisions[-1])
        return own_decisions[-1] if len(own_decisions) > 0 else complement_action(Action.COOPERATING, suspicious)
    return Strategy("suspicious-pavlov", action) if suspicious else Strategy("pavlov", action)


def strategy_pavlovish(suspicious=False) -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # defect if opponent defected and we didn't
        # otherwise keep doing whatever we did last time
        if len(opponent_decisions) >= 1 and opponent_decisions[-1] == Action.DEFECTING and own_decisions[-1] == Action.COOPERATING:
            return Action.DEFECTING
        return own_decisions[-1] if len(own_decisions) > 0 else complement_action(Action.COOPERATING, suspicious)
    return Strategy("suspicious-pavlovish", action) if suspicious else Strategy("pavlovish", action)


def strategy_pavlov_spooky(suspicious=False) -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # switch strategy if opponent's action didn't match ours
        # otherwise keep doing whatever we did last time
        if len(opponent_decisions) >= 1 and opponent_decisions[-1] != own_decisions[-1]:
            return complement_action(own_decisions[-1])
        return own_decisions[-1] if len(own_decisions) > 0 else complement_action(Action.COOPERATING, suspicious)
    return Strategy("suspicious-pavlov-spooky", action) if suspicious else Strategy("pavlov-spooky", action)

def strategy_two_tits_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        if len(own_decisions) > 0:
            return Action.DEFECTING if own_decisions[-1] == Action.DEFECTING else opponent_decisions[-1]
        return Action.COOPERATING
    return Strategy("two-tits-for-tat", action)


def strategy_suspicious_two_tits_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        if len(own_decisions) < 1:
            return Action.DEFECTING
        if len(opponent_decisions) == 1:
            return opponent_decisions[-1]
        return Action.DEFECTING if own_decisions[-1] == Action.DEFECTING else opponent_decisions[-1]
    return Strategy("suspicious-two-tits-for-tat", action)


def strategy_hard_tit_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        return Action.DEFECTING if len(opponent_decisions) >= 3 and Action.DEFECTING in opponent_decisions[-3:] else Action.COOPERATING
    return Strategy("hard-tit-for-tat", action)


def strategy_soft_grudger() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        if len(opponent_decisions) > 0 and opponent_decisions[-1] == Action.DEFECTING:
            local_state["next_decisions"] = [Action.DEFECTING]*4 + [Action.COOPERATING]*2
        next_decisions = local_state.get("next_decisions")
        if next_decisions is not None and len(next_decisions) > 0:
            local_state["next_decisions"] = next_decisions[1:]
            return next_decisions[0]
        return Action.COOPERATING
    return Strategy("soft-grudger", action)


def strategy_hard_majority() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        cnt_coop, cnt_def = get_coop_and_defect_count(local_state, opponent_decisions)
        return Action.COOPERATING if cnt_coop > cnt_def else Action.DEFECTING
    return Strategy("hard-majority", action)


def strategy_prober() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        if len(own_decisions) == 0:
            local_state["startup_decisions"] = [Action.DEFECTING] + [Action.COOPERATING]*2
        startup_decisions = local_state.get("startup_decisions")
        if startup_decisions is not None and len(startup_decisions) > 0:
            local_state["startup_decisions"] = startup_decisions[1:]
            return startup_decisions[0]
        if len(opponent_decisions) == 3 and opponent_decisions[-1] == opponent_decisions[-2] and opponent_decisions[-1] == Action.COOPERATING:
            return Action.DEFECTING
        else:
            return opponent_decisions[-1]
    return Strategy("prober", action)


def strategy_handshake() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        if len(own_decisions) == 0:
            local_state["startup_decisions"] = [Action.DEFECTING] + [Action.COOPERATING]
        startup_decisions = local_state.get("startup_decisions")
        if startup_decisions is not None and len(startup_decisions) > 0:
            local_state["startup_decisions"] = startup_decisions[1:]
            return startup_decisions[0]
        if len(opponent_decisions) == 2 and opponent_decisions[0] == Action.DEFECTING and opponent_decisions[1] == Action.COOPERATING:
            return Action.COOPERATING
        else:
            return Action.DEFECTING
    return Strategy("handshake", action)


name2strategy = {
    "defector": strategy_defector(),
    "hate-opponent": strategy_hate_opponent(),
    "angry-grudger": strategy_angry_grudger(),
    "grudger": strategy_grudger(),
    "soft-grudger": strategy_soft_grudger(),
    "hard-majority":strategy_hard_majority(),
    "suspicious-pavlovish": strategy_pavlovish(True),
    "suspicious-pavlov-spooky": strategy_pavlov_spooky(True),
    "suspicious-pavlov": strategy_pavlov(True),
    "suspicious-sophist": strategy_sophist(True),
    "suspicious-tit-for-tat": strategy_tit_for_tat(True),
    "random": strategy_random(),
    "alternator": strategy_alternator(),
    "hard-tit-for-tat": strategy_hard_tit_for_tat(),
    "handshake": strategy_handshake(),
    "prober": strategy_prober(),
    "suspicious-two-tits-for-tat": strategy_suspicious_two_tits_for_tat(),
    "two-tits-for-tat": strategy_two_tits_for_tat(),
    "firm-but-fair": strategy_firm_but_fair(),
    "sophist": strategy_sophist(),
    "pavlov": strategy_pavlov(),
    "pavlov-spooky": strategy_pavlov_spooky(),
    "pavlovish": strategy_pavlovish(),
    "tit-for-tat": strategy_tit_for_tat(),
    "forgiving-tit-for-tat": strategy_forgiving_tit_for_tat(),
    "gandhi": strategy_gandhi(),
}