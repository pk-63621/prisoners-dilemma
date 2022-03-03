#!/usr/bin/env python3

import random

from enum import Enum
from typing import Callable, Dict, List


class Action(Enum):
    DEFECTING, COOPERATING = 'Defecting', 'Cooperating'


def complement_action(a: Action) -> Action:
    if a == Action.DEFECTING:
        return Action.COOPERATING
    else:
        return Action.DEFECTING


def action_trace(al: List[Action]) -> str:
    return "".join(map(lambda a: 'C' if a == Action.COOPERATING else 'D', al))


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
        # startup
        if len(own_decisions) == 0:
            return Action.COOPERATING

        if len(own_decisions) > 0:
            return complement_action(own_decisions[-1])
        else:
            print("Wrong config")
            return None

    return Strategy("alternator", action)


def strategy_hate_opponent() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # startup
        if len(own_decisions) == 0:
            return Action.DEFECTING

        if len(opponent_decisions) > 0:
            return complement_action(opponent_decisions[-1])
        else:
            print("Wrong config")
            return None

    return Strategy("hate-opponent", action)


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
        """
        start with cooperation
        defect if opponent defected at some point
        otherwise keep cooperating
        The trace of grudger has the shape: C*D*
        """
        if len(own_decisions) == 0:
            return Action.COOPERATING

        if has_defection(local_state, opponent_decisions):
            return Action.DEFECTING
        else:
            return Action.COOPERATING

    return Strategy("grudger", action)


# NOTE: pavlovish is same as grudger
# def strategy_pavlovish() -> Strategy:
#     def action(own_decisions, opponent_decisions, local_state):
#         """
#         start with cooperation
#         defect if opponent defected and we didn't
#         otherwise keep doing whatever we did last time
#         """
#         if len(own_decisions) == 0:
#             return Action.COOPERATING
#         assert len(opponent_decisions) >= 1
#         if opponent_decisions[-1] == Action.DEFECTING and own_decisions[-1] == Action.COOPERATING:
#             return Action.DEFECTING
#         else:
#             return own_decisions[-1]
#
#     return Strategy("pavlovish", action)


def strategy_angry_grudger() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # startup
        if len(opponent_decisions) == 0:
            return Action.DEFECTING

        if has_defection(local_state, opponent_decisions):
            return Action.DEFECTING
        else:
            return Action.COOPERATING

    return Strategy("angry-grudger", action)


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


def strategy_sophist() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # startup
        if len(own_decisions) == 0:
            return Action.COOPERATING

        cnt_coop, cnt_def = get_coop_and_defect_count(local_state, opponent_decisions)
        if cnt_def > cnt_coop:
            return Action.DEFECTING
        else:
            return Action.COOPERATING

    return Strategy("sophist", action)


def strategy_suspicious_sophist() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # startup
        if len(opponent_decisions) == 0:
            return Action.DEFECTING

        cnt_coop, cnt_def = get_coop_and_defect_count(local_state, opponent_decisions)
        if cnt_def >= cnt_coop:
            return Action.DEFECTING
        else:
            return Action.COOPERATING

    return Strategy("suspicious-sophist", action)


def strategy_tit_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # startup
        if len(own_decisions) == 0:
            return Action.COOPERATING

        if len(opponent_decisions) >= 1:
            return opponent_decisions[-1]
        else:
            print("Wrong config")
            return None

    return Strategy("tit-for-tat", action)


def strategy_suspicious_tit_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # startup
        if len(own_decisions) == 0:
            return Action.DEFECTING

        if len(opponent_decisions) >= 1:
            return opponent_decisions[-1]
        else:
            print("Wrong config")
            return None

    return Strategy("suspicious-tit-for-tat", action)


def strategy_forgiving_tit_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # startup
        if len(own_decisions) == 0:
            return Action.COOPERATING

        if len(opponent_decisions) >= 2 and (opponent_decisions[-1] == opponent_decisions[-2]):
            return opponent_decisions[-1]
        else:
            return Action.COOPERATING

    return Strategy("forgiving-tit-for-tat", action)


def strategy_firm_but_fair() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # startup
        if len(own_decisions) == 0:
            return Action.COOPERATING

        if opponent_decisions[-1] == own_decisions[-1]:
            return Action.COOPERATING
        elif has_defection(local_state, opponent_decisions):
            return Action.DEFECTING
        else:
            return Action.COOPERATING

    return Strategy("firm-but-fair", action)


def strategy_pavlov() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        """
        start with cooperation
        switch strategy if opponent defected in last move
        otherwise keep doing whatever we did last time
        """
        if len(own_decisions) == 0:
            return Action.COOPERATING
        assert len(opponent_decisions) >= 1
        if opponent_decisions[-1] == Action.DEFECTING:
            return complement_action(own_decisions[-1])
        else:
            return own_decisions[-1]

    return Strategy("pavlov", action)


def strategy_suspicious_pavlov() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        """
        same as pavlov except it starts with defection
        """
        if len(own_decisions) == 0:
            return Action.DEFECTING
        assert len(opponent_decisions) >= 1
        if opponent_decisions[-1] == Action.DEFECTING:
            return complement_action(own_decisions[-1])
        else:
            return own_decisions[-1]

    return Strategy("suspicious-pavlov", action)


def strategy_spooky_pavlov() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        """
        start with cooperation
        switch strategy if opponent's action didn't match ours
        otherwise keep doing whatever we did last time
        """
        if len(own_decisions) == 0:
            return Action.COOPERATING
        assert len(opponent_decisions) >= 1
        if opponent_decisions[-1] != own_decisions[-1]:
            return complement_action(own_decisions[-1])
        else:
            return own_decisions[-1]

    return Strategy("spooky-pavlov", action)


def strategy_suspicious_spooky_pavlov() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        """
        same as spooky-pavlov except it starts with defection
        """
        if len(own_decisions) == 0:
            return Action.DEFECTING
        assert len(opponent_decisions) >= 1
        if opponent_decisions[-1] != own_decisions[-1]:
            return complement_action(own_decisions[-1])
        else:
            return own_decisions[-1]

    return Strategy("suspicious-spooky-pavlov", action)


def strategy_two_tits_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # startup
        if len(own_decisions) == 0:
            return Action.COOPERATING

        if local_state.get("pending") is not None:
            local_state["pending"] = None
            if opponent_decisions[-1] == Action.DEFECTING:
                local_state["pending"] = 1
            return Action.DEFECTING

        if opponent_decisions[-1] == Action.DEFECTING:
            local_state["pending"] = 1
            return Action.DEFECTING
        else:
            return opponent_decisions[-1]

    return Strategy("two-tits-for-tat", action)


def strategy_suspicious_two_tits_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # startup
        if len(own_decisions) < 1:
            return Action.DEFECTING

        # second round case
        if len(opponent_decisions) == 1:
            return opponent_decisions[-1]

        if local_state.get("pending") is not None:
            local_state["pending"] = None
            if opponent_decisions[-1] == Action.DEFECTING:
                local_state["pending"] = 1
            return Action.DEFECTING

        if opponent_decisions[-1] == Action.DEFECTING:
            local_state["pending"] = 1
            return Action.DEFECTING
        else:
            return opponent_decisions[-1]

    return Strategy("suspicious-two-tits-for-tat", action)


def strategy_hard_tit_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # startup
        if len(own_decisions) == 0:
            return Action.COOPERATING

        if len(opponent_decisions) >= 3 and (Action.DEFECTING in opponent_decisions[-3:]):
            return Action.DEFECTING
        else:
            return Action.COOPERATING

    return Strategy("hard-tit-for-tat", action)


def strategy_soft_grudger() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # startup
        if len(own_decisions) == 0:
            return Action.COOPERATING

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
        # startup
        if len(own_decisions) == 0:
            return Action.COOPERATING

        cnt_coop, cnt_def = get_coop_and_defect_count(local_state, opponent_decisions)
        if cnt_coop > cnt_def:
            return Action.COOPERATING
        else:
            return Action.DEFECTING

    return Strategy("hard-majority", action)


def strategy_prober() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # startup
        if len(own_decisions) == 0:
            local_state["startup_decisions"] = [Action.DEFECTING] + [Action.COOPERATING]*2
        startup_decisions = local_state.get("startup_decisions")
        if startup_decisions is not None and len(startup_decisions) > 0:
            local_state["startup_decisions"] = startup_decisions[1:]
            return startup_decisions[0]

        if local_state.get("fix_state"):
            return Action.DEFECTING

        if len(opponent_decisions) == 3 and (opponent_decisions[-1] == opponent_decisions[-2]) and (opponent_decisions[-1] == Action.COOPERATING):
            local_state["fix_state"] = True
            return Action.DEFECTING
        else:
            return opponent_decisions[-1]

    return Strategy("prober", action)


def strategy_handshake() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        # startup
        if len(own_decisions) == 0:
            local_state["startup_decisions"] = [Action.DEFECTING] + [Action.COOPERATING]
        startup_decisions = local_state.get("startup_decisions")
        if startup_decisions is not None and len(startup_decisions) > 0:
            local_state["startup_decisions"] = startup_decisions[1:]
            return startup_decisions[0]

        if len(opponent_decisions) >= 2 and (opponent_decisions[0] == Action.DEFECTING) and (opponent_decisions[1] == Action.COOPERATING):
            return Action.COOPERATING
        else:
            return Action.DEFECTING

    return Strategy("handshake", action)


def strategy_user() -> Strategy:
    def action(own_decisions, opponent_decisions, local_state):
        """
        read from user:
        * input starting with "C" is treated as cooperation
        * last action is repeated if no input is provided
        * otherwise defection is assumed
        """
        if len(opponent_decisions) >= 1:
            print(f"Opponent's last move: {opponent_decisions[-1].value}")
        move = input("Your move: ")
        if move.lower().startswith("c"):
            return Action.COOPERATING
        elif len(move) == 0 and len(own_decisions) >= 1:
            return own_decisions[-1]
        else:
            return Action.DEFECTING

    return Strategy("user", action)


name2strategy = {
    "defector": strategy_defector(),
    "hate-opponent": strategy_hate_opponent(),
    "angry-grudger": strategy_angry_grudger(),
    "grudger": strategy_grudger(),
    "soft-grudger": strategy_soft_grudger(),
    "hard-majority":strategy_hard_majority(),
    "suspicious-spooky-pavlov": strategy_suspicious_spooky_pavlov(),
    "suspicious-pavlov": strategy_suspicious_pavlov(),
    "suspicious-sophist": strategy_suspicious_sophist(),
    "suspicious-tit-for-tat": strategy_suspicious_tit_for_tat(),
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
    "spooky-pavlov": strategy_spooky_pavlov(),
    "tit-for-tat": strategy_tit_for_tat(),
    "forgiving-tit-for-tat": strategy_forgiving_tit_for_tat(),
    "gandhi": strategy_gandhi(),
    "user": strategy_user(),
}


def all_strategies_name():
    return name2strategy.keys()


def all_strategies():
    return name2strategy.values()


def all_strategies_mod(excluding: List[str] = []) -> List[Strategy]:
    ret = []
    for k,v in name2strategy.items():
        if k in excluding:
            continue
        ret.append(v)
    return ret


def random_strategy() -> Strategy:
    choice: Strategy = random.choice(list(all_strategies()))
    return choice


def get_strategies_by_name(sc: str, random_if_not_found=False) -> List[Strategy]:
    """
     sc is assumed to be of the form:
      sc       ::= s | s*N
      s        ::= STRATEGY | "all" | "all-"ls
      ls       ::= STRATEGY | ls","STRATEGY

      N in { positive-integers }
      STRATEGY in { strategies }

      if random_if_not_found is True then an invalid strategy is replaced by a randomly chosen strategy
                                     otherwise silently ignored
    """
    def decode_into_strategy_and_count(sc):
        ss = sc.split('*')
        if len(ss) == 2 and ss[1].isdigit():
            cnt = max(int(ss[1]), 1)
            return ss[0], cnt
        return sc, 1
    s, cnt = decode_into_strategy_and_count(sc)
    ret = []
    if s in name2strategy:
        ret = [name2strategy[s]]
    elif s.startswith('all'):
        excluding_strategies:List[str] = []
        remaining_str: str = s[len('all'):]
        if len(remaining_str) >= 1 and remaining_str[0] == '-':
            excluding_strategies = remaining_str[1:].split(',')
        ret = all_strategies_mod(excluding_strategies)
    elif random_if_not_found:
        ret = [random_strategy()]
    return ret*cnt
