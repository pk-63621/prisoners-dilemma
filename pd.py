#!/usr/bin/env python3

import random
import sys
import traceback

from enum import Enum
from typing import Callable, Dict, Tuple, List


class Action(Enum):
    DEFECTING = 'Defecting'
    COOPERATING = 'Cooperating'


class Strategy:
    def __init__(self, name, action):
        self.name: str = name
        self.action: Callable = action

    def action(
            self, own_decisions: List[Action],
            opponent_decisions: List[Action]) -> Action:
        return self.action(own_decisions, opponent_decisions)


class Prisoner:
    def __init__(self, name: str, strategy: Strategy):
        assert strategy is not None
        self.name = name
        self.jail_time: List[int] = []
        self.opponent_sum = 0
        self.opponent_decisions: List[Action] = []
        self.decisions: List[Action] = []
        self.strategy = strategy

    def add_play(self, decision, play):
        try:
            if play is None or not decision:
                print("No value for required variables: {} {}".format(
                    play, decision))
                return
            self.jail_time.append(int(play))
            self.decisions.append(decision)
        except Exception as e:
            print("fix the play value", e)

    def opponent_history(self, opponent_decision, last_play):
        try:
            if last_play is None or not opponent_decision:
                print("No value for last_play")
                return
            self.opponent_sum += int(last_play)
            self.opponent_decisions.append(opponent_decision)
        except Exception as e:
            print("Fix opponent history", e)

    def get_result(self):
        try:
            return sum(self.jail_time)
        except Exception as e:
            print("jail time values: {} with exception {}".format(
                self.jail_time, e))

    def get_decision(self) -> Action:
        return self.strategy.action(self.decisions, self.opponent_decisions)


class PrisonersDilemma:
    def __init__(self, matrix: Dict, prisoners: List[Prisoner]):
        assert len(matrix) != 0
        assert len(list(matrix.items())[0][0]) == len(prisoners)
        self.play_matrix = matrix
        self.prisoners = prisoners

    def get_result(self, decisions):
        try:
            return self.play_matrix[decisions]
        except Exception as e:
            print(
                f"play_matrix is ill-formed!  Error {traceback.format_exc()}",
                file=sys.stderr)
            print("Exception {} raised".format(e))

    def play_next_round(self) -> Tuple[Tuple[Action, ...], List[int]]:
        decisions = tuple(
                        prisoner.get_decision() for prisoner in self.prisoners)
        results = self.get_result(decisions)

        # HACK assuming 2 players!
        assert len(self.prisoners) == 2
        for i in range(2):
            self.prisoners[i].add_play(decisions[i], results[i])
            self.prisoners[i].opponent_history(decisions[1-i], results[1-i])

        return decisions, results


# sample strategies:
def strategy_defector() -> Strategy:
    def action(own_decisions, opponent_decisions):
        return Action.DEFECTING
    return Strategy("defector", action)


def strategy_idiot() -> Strategy:
    def action(own_decisions, opponent_decisions):
        return Action.COOPERATING
    return Strategy("idiot", action)


def strategy_random() -> Strategy:
    def action(own_decisions, opponent_decisions):
        return random.choice(list(Action))
    return Strategy("random", action)


def strategy_sophist() -> Strategy:
    def action(own_decisions, opponent_decisions):
        cnt_def = sum(
                    1 for d in opponent_decisions if d == Action.DEFECTING)
        cnt_coop = sum(
                    1 for d in opponent_decisions if
                    d == Action.COOPERATING)
        if cnt_def > cnt_coop:
            return Action.DEFECTING
        else:
            return Action.COOPERATING
    return Strategy("sophist", action)


def strategy_tit_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions):
        if len(opponent_decisions) > 0:
            return opponent_decisions[-1]
        return Action.COOPERATING
    return Strategy("tit for tat", action)


def strategy_forgiving_tit_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions):
        if (len(opponent_decisions) > 1
                and opponent_decisions[-1] == opponent_decisions[-2]):
            return opponent_decisions[-1]
        return Action.COOPERATING
    return Strategy("forgiving tit for tat", action)


def strategy_pavlov() -> Strategy:
    def action(own_decisions, opponent_decisions):
        pavlov_decision = own_decisions[-1] if len(own_decisions) > 0 else\
                            Action.COOPERATING
        if(
            len(opponent_decisions) > 0 and
                opponent_decisions[-1] == Action.DEFECTING):
            pavlov_decision = Action.COOPERATING if own_decisions[-1] ==\
                                Action.DEFECTING else Action.DEFECTING
        return pavlov_decision
    return Strategy("pavlov", action)


def main():
    play_matrix = {
                    (Action.COOPERATING, Action.COOPERATING): (3, 3),
                    (Action.COOPERATING, Action.DEFECTING): (0, 5),
                    (Action.DEFECTING,  Action.COOPERATING): (5, 0),
                    (Action.DEFECTING,  Action.DEFECTING): (1, 1),
                  }
    # prisoner1 = Prisoner("prisoner1.aka.idiot", strategy_idiot())
    # prisoner2 = Prisoner("prisoner2.aka.defector", strategy_defector())
    prisoner2 = Prisoner("prisoner2.aka.pavlov", strategy_pavlov())
    # prisoner1 = Prisoner(
    #                "prisoner2.aka.ft4t",
    #                strategy_forgiving_tit_for_tat())
    prisoner1 = Prisoner("prisoner1.aka.pavlov", strategy_pavlov())
    # prisoner1 = Prisoner("prisoner1.aka.tit4tat", strategy_tit_for_tat())
    # prisoner2 = Prisoner("prisoner2.aka.sophist", strategy_sophist())
    game = PrisonersDilemma(play_matrix, [prisoner1, prisoner2])
    for i in range(10):
        decisions, results = game.play_next_round()
        print(f"Game: {', '.join(d.value for d in decisions)}")
        print(f"Result: {', '.join(str(r) for r in results)}")
        print()

    print(f"Result for {prisoner1.name: <25} {prisoner1.get_result()}")
    print(f"Result for {prisoner2.name: <25} {prisoner2.get_result()}")


if __name__ == '__main__':
    main()
