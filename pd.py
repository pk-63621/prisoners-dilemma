#!/usr/bin/env python3

import argparse
import random
import sys
import traceback
import itertools

from enum import Enum
from typing import Callable, Dict, Tuple, List, Optional, TypeVar, Union

T = TypeVar('T')


def unpack_tuple_if_singleton(a: Tuple[T,...]) -> Union[Tuple[T,...],T]:
    if len(a) == 1:
        return a[0]
    return a


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

    def action(self, own_decisions: List[Action], opponent_decisions: List[Action]) -> Action:
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
            raise e

    def opponent_history(self, opponent_decision, last_play):
        try:
            self.opponent_sum += int(last_play)
            self.opponent_decisions.append(opponent_decision)
        except Exception as e:
            print("Fix opponent history", e)

    def get_result(self) -> int:
        try:
            return sum(self.jail_time)
        except Exception as e:
            print("jail time values: {} with exception {}".format(
                self.jail_time, e))
            raise e

    def get_decision(self) -> Action:
        return self.strategy.action(self.decisions, self.opponent_decisions)


class PrisonersDilemma:
    def __init__(self, matrix: Dict, prisoners: List[Prisoner], noise=0.0):
        assert len(matrix) != 0
        assert len(list(matrix.items())[0][0]) == len(prisoners)
        self.play_matrix = matrix
        self.prisoners = prisoners
        self.noise_probability = noise

    def get_result(self, decisions):
        try:
            return self.play_matrix[decisions]
        except Exception as e:
            print(f"play_matrix is ill-formed!  Error {traceback.format_exc()}", file=sys.stderr)
            raise e

    def get_noise(self):
        random_floating_value = random.uniform(0, 1)
        return random_floating_value < self.noise_probability

    def play_next_iteration(self) -> Tuple[Tuple[Action, ...], List[int]]:
        decisions             = tuple(prisoner.get_decision() for prisoner in self.prisoners)
        decisions_after_noise = tuple(complement_action(d) if self.get_noise() else d for d in decisions)
        results = self.get_result(decisions_after_noise)

        for i in range(len(self.prisoners)):
            self.prisoners[i].add_play(decisions[i], results[i])
            opponents_decisions = decisions_after_noise[:i] + decisions_after_noise[i+1:]
            opponents_results   = results[:i] + results[i+1:]
            # HACK for 2 players only game, unpack the tuples
            unpacked_opponents_decisions = unpack_tuple_if_singleton(opponents_decisions)
            unpacked_opponents_results   = unpack_tuple_if_singleton(opponents_results)
            self.prisoners[i].opponent_history(unpacked_opponents_decisions, unpacked_opponents_results)

        # TODO indicate noise as well?
        return decisions_after_noise, results


# all participants play against each other
class PrisonersDilemmaTournament:
    def __init__(self, play_matrix, strategies: List[Strategy], participants_per_game=2, iterations=10, noise_error_prob=0.0):
        self.play_matrix = play_matrix
        self.strategies = strategies
        self.iterations = iterations
        self.noise_error_prob = noise_error_prob
        self.participants_per_game = participants_per_game
        self.final_outcome = dict()

    def play_tournament(self, verbose=False):
        r = 0
        for strats in itertools.combinations_with_replacement(self.strategies, self.participants_per_game):
            r += 1
            prisoners = [Prisoner(f"prisoner{i+1}.aka.{strat.name}", strat) for i,strat in enumerate(strats)]
            game = PrisonersDilemma(self.play_matrix, prisoners, self.noise_error_prob)

            if verbose:
                print()
                print(f"=== Tournament Round #{r} ===")
                print()
                print(f"Participants: {', '.join(p.name for p in prisoners)}")
                print()

            for i in range(self.iterations):
                decisions, results = game.play_next_iteration()
                s = f"Iteration #{i+1: <3}:"
                if verbose:
                    print(f"{s} Actions: {', '.join(d.value for d in decisions)}")
                    print(f"{' '*len(s)} Result: {', '.join(str(r) for r in results)}")
                    print()

            if verbose:
                print(f"Result for Round #{r}:")
            padding_len = max(map(lambda p: len(p.name), prisoners))
            for p in prisoners:
                if self.final_outcome.get(p.strategy.name) is None:
                    self.final_outcome[p.strategy.name] = [p.get_result()]
                else:
                    self.final_outcome[p.strategy.name].append(p.get_result())
                if verbose:
                    print(f"\t{p.name: <{padding_len}} = {p.get_result()}")

    def get_final_outcome(self):
        return self.final_outcome


# strategies -- add to name2strategy if adding new strategy

def strategy_defector() -> Strategy:
    def action(own_decisions, opponent_decisions):
        return Action.DEFECTING
    return Strategy("defector", action)


def strategy_gandhi() -> Strategy:
    def action(own_decisions, opponent_decisions):
        return Action.COOPERATING
    return Strategy("gandhi", action)


def strategy_random() -> Strategy:
    def action(own_decisions, opponent_decisions):
        return random.choice(list(Action))
    return Strategy("random", action)


def strategy_sophist() -> Strategy:
    def action(own_decisions, opponent_decisions):
        cnt_def  = sum(1 for d in opponent_decisions if d == Action.DEFECTING)
        cnt_coop = sum(1 for d in opponent_decisions if d == Action.COOPERATING)
        if cnt_def > cnt_coop:
            return Action.DEFECTING
        return Action.COOPERATING
    return Strategy("sophist", action)


def strategy_tit_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions):
        if len(opponent_decisions) >= 1:
            return opponent_decisions[-1]
        return Action.COOPERATING
    return Strategy("tit for tat", action)


def strategy_forgiving_tit_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions):
        if (len(opponent_decisions) >= 2
                and opponent_decisions[-1] == opponent_decisions[-2]):
            return opponent_decisions[-1]
        return Action.COOPERATING
    return Strategy("forgiving tit for tat", action)


def strategy_pavlov() -> Strategy:
    def action(own_decisions, opponent_decisions):
        # switch strategy if opponent defected
        # otherwise keep doing whatever we did last time
        if len(opponent_decisions) >= 1 and opponent_decisions[-1] == Action.DEFECTING:
            return complement_action(own_decisions[-1])
        if len(own_decisions) > 0:
            return own_decisions[-1]
        return Action.COOPERATING
    return Strategy("pavlov", action)


def strategy_pavlovish() -> Strategy:
    def action(own_decisions, opponent_decisions):
        # defect if opponent defected and we didn't
        # otherwise keep doing whatever we did last time
        if len(opponent_decisions) >= 1 and opponent_decisions[-1] == Action.DEFECTING and own_decisions[-1] == Action.COOPERATING:
            return Action.DEFECTING
        if len(own_decisions) > 0:
            return own_decisions[-1]
        return Action.COOPERATING
    return Strategy("pavlovish", action)


def strategy_pavlov_spooky() -> Strategy:
    def action(own_decisions, opponent_decisions):
        # switch strategy if opponent's action didn't match ours
        # otherwise keep doing whatever we did last time
        if len(opponent_decisions) >= 1 and opponent_decisions[-1] != own_decisions[-1]:
            return complement_action(own_decisions[-1])
        if len(own_decisions) > 0:
            return own_decisions[-1]
        return Action.COOPERATING
    return Strategy("pavlov_spooky", action)


name2strategy = {
    "defector": strategy_defector(),
    "gandhi": strategy_gandhi(),
    "random": strategy_random(),
    "sophist": strategy_sophist(),
    "tit-for-tat": strategy_tit_for_tat(),
    "forgiving-tit-for-tat": strategy_forgiving_tit_for_tat(),
    "pavlov": strategy_pavlov(),
    "pavlovish": strategy_pavlovish(),
    "pavlov-spooky": strategy_pavlov_spooky(),
}


def random_strategy() -> Strategy:
    choice: str = random.choice(list(name2strategy.keys()))
    return name2strategy[choice]


def get_strategy_by_name(s: str, random_if_not_found=False) -> Optional[Strategy]:
    if s in name2strategy:
        return name2strategy[s]
    if random_if_not_found:
        return random_strategy()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='store_true', help='Show output of each game')
    parser.add_argument('--iterations', '-i', default=30, type=int, help='Number of iterations of game')
    parser.add_argument('--error-prob', '-ep', default=0.0, type=float, help='Probability of error due to noise (Due to noise decision gets flipped)')
    parser.add_argument('strategies', metavar='STRATEGY', type=str, nargs='+', default=['defector','defector'], help='Strategies for prisoners')
    args = parser.parse_args()

    verbose = args.verbose
    iterations = args.iterations
    noise = args.error_prob
    strategies_name = args.strategies

    if len(strategies_name) <= 1:
        # return silently
        return

    strategies = map(lambda s: get_strategy_by_name(s, random_if_not_found=True), strategies_name)

    play_matrix = {
                    (Action.COOPERATING, Action.COOPERATING): (2, 2),
                    (Action.COOPERATING, Action.DEFECTING):   (0, 3),
                    (Action.DEFECTING,   Action.COOPERATING): (3, 0),
                    (Action.DEFECTING,   Action.DEFECTING):   (1, 1),
                  }
    tournament = PrisonersDilemmaTournament(play_matrix, strategies, iterations=iterations, noise_error_prob=noise)
    tournament.play_tournament(verbose)
    final_result = tournament.get_final_outcome()

    print("Strategy wise result")
    print("--------------------")
    best_strat, best_score = None, 0
    for strat in final_result.keys():
        total_score = sum(final_result[strat])
        print("Strategy: {0:30} Result: {1:10}".format(strat, total_score))
        if best_score < total_score:
            best_score = total_score
            best_strat = strat
    print("--------------------")
    print("\nBest strategy is {0:23} with score {1:7}".format(best_strat, best_score))



if __name__ == '__main__':
    main()
