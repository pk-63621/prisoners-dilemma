#!/usr/bin/env python3

import shlex
import argparse
import random
import sys
import traceback
import itertools
import math

from enum import Enum
from collections import defaultdict
from typing import Callable, DefaultDict, Dict, List, Optional, Tuple, TypeVar, Union

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


class TournamentParticipant:
    def __init__(self, name: str, strategy: Strategy):
        self.name = name
        self.strategy = strategy

    def replicate(self, generation: int):
        new_name = f"{self.name}.{generation}"
        new_strategy = self.strategy
        return TournamentParticipant(new_name, new_strategy)


# all participants play against each other
class PrisonersDilemmaTournament:
    def __init__(self, play_matrix, participants: List[TournamentParticipant], participants_per_game=2, iterations=10, noise_error_prob=0.0):
        self.play_matrix = play_matrix
        self.participants = participants
        self.iterations = iterations
        self.noise_error_prob = noise_error_prob
        self.participants_per_game = participants_per_game

    def play_tournament(self, verbose=0, quiet=False) -> Dict[TournamentParticipant,int]:
        r = 0
        outcome: DefaultDict[TournamentParticipant,int] = defaultdict(int)
        if not quiet:
            print(f"Tournament participants[{len(self.participants)}]: {', '.join(s.name for s in self.participants)}")
        for round_participants in itertools.combinations(self.participants, self.participants_per_game):
            r += 1
            prisoners = [Prisoner(f"prisoner{i+1}.aka.{part.name}", part.strategy) for i,part in enumerate(round_participants)]
            game = PrisonersDilemma(self.play_matrix, prisoners, self.noise_error_prob)

            if verbose >= 3:
                print()
                print(f"=== Tournament Round #{r} ===")
                print()
                print(f"Participants: {', '.join(p.name for p in prisoners)}")
                print()

            for i in range(self.iterations):
                decisions, results = game.play_next_iteration()
                s = f"Iteration #{i+1: <3}:"
                if verbose >= 3:
                    print(f"{s} Actions: {', '.join(d.value for d in decisions)}")
                    print(f"{' '*len(s)} Result: {', '.join(str(r) for r in results)}")
                    print()
                    pass

            if verbose >= 2:
                print(f"Result for Round #{r}:")
            for prisoner,participant in zip(prisoners,round_participants):
                result = prisoner.get_result()
                outcome[participant] += result
                if verbose >= 2:
                    print(f"\t{participant.name: <{40}} {result}")

        if verbose >= 1:
            print()
            print("Tournament outcome")
            sorted_results = sorted(outcome.items(), key=lambda p: p[1])
            print("------------------")
            for part,score in sorted_results:
                print("\t{0:40} {1:10}".format(part.name, score))
            print("------------------")
        return outcome


class PrisonersDilemmaTournamentWithEvolution:
    def __init__(self, play_matrix, participants: List[TournamentParticipant],
                 participants_per_game=2,
                 iterations=10,
                 noise_error_prob=0.0,
                 fraction_eliminated_after_each_tournament=0.1,
                 rounds_of_evolution=2):
        self.play_matrix = play_matrix
        self.orig_participants = participants
        self.iterations = iterations
        self.noise_error_prob = noise_error_prob
        self.participants_per_game = participants_per_game
        self.fraction_eliminated_after_each_tournament = fraction_eliminated_after_each_tournament
        self.rounds_of_evolution = rounds_of_evolution

    def play_tournament(self, participants, verbose=0, quiet=False) -> Dict[TournamentParticipant,int]:
        tournament = PrisonersDilemmaTournament(self.play_matrix, participants,
                                                participants_per_game=self.participants_per_game,
                                                iterations=self.iterations,
                                                noise_error_prob=self.noise_error_prob)
        return tournament.play_tournament(verbose=verbose, quiet=False)

    def eliminate_and_replicate(self,
                                last_participants: List[TournamentParticipant],
                                last_outcome: Dict[TournamentParticipant,int],
                                generation: int, verbose=0) -> List[TournamentParticipant]:
        assert last_outcome is not None
        assert len(last_outcome) == len(last_participants)
        last_outcome_sorted = sorted(last_outcome.items(), key=lambda p: p[1])
        nr = math.floor(len(last_participants)*self.fraction_eliminated_after_each_tournament)
        to_be_eliminated = set([p.name for p,_ in last_outcome_sorted[:nr]])
        to_be_replicated = set([p.name for p,_ in last_outcome_sorted[-nr:]])
        new_participants = [p for p in last_participants if p.name not in to_be_eliminated]
        new_participants.extend([p.replicate(generation) for p in last_participants if p.name in to_be_replicated])
        if verbose >= 1:
            print()
            print(f"*** Eliminating bottom {nr} and replicating top {nr}")
            print()
        if verbose >= 2:
            print()
            print(f"*** Eliminated: {', '.join(to_be_eliminated)}")
            print(f"*** Replicated: {', '.join(to_be_replicated)}")
            print()
        return new_participants

    def play_tournament_with_evolution(self, verbose=0, quiet=False) -> Optional[Dict[TournamentParticipant,int]]:
        last_outcome = None
        last_participants = self.orig_participants
        for i in range(self.rounds_of_evolution):
            last_outcome = self.play_tournament(last_participants, verbose, quiet)
            last_participants = self.eliminate_and_replicate(last_participants, last_outcome, i+1, verbose)
        return last_outcome


# strategies -- add to name2strategy if adding new strategy

def strategy_defector() -> Strategy:
    def action(own_decisions, opponent_decisions):
        return Action.DEFECTING
    return Strategy("defector", action)


def strategy_alternator() -> Strategy:
    def action(own_decisions, opponent_decisions):
        return complement_action(own_decisions[-1]) if len(own_decisions) > 0 else Action.COOPERATING
    return Strategy("alternator", action)


def strategy_hate_opponent() -> Strategy:
    def action(own_decisions, opponent_decisions):
        return complement_action(opponent_decisions[-1]) if len(opponent_decisions) > 0 else Action.DEFECTING
    return Strategy("hate opponent", action)


def strategy_grudger() -> Strategy:
    def action(own_decisions, opponent_decisions):
        if Action.DEFECTING in opponent_decisions:
            return Action.DEFECTING
        else:
            return Action.COOPERATING
    return Strategy("grudger", action)


def strategy_angry_grudger() -> Strategy:
    def action(own_decisions, opponent_decisions):
        if Action.DEFECTING in opponent_decisions or len(opponent_decisions) == 0:
            return Action.DEFECTING
        else:
            return Action.COOPERATING
    return Strategy("angry grudger", action)


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


def strategy_suspicious_sophist() -> Strategy:
    def action(own_decisions, opponent_decisions):
        if len(opponent_decisions) == 0:
            return Action.DEFECTING
        cnt_def  = sum(1 for d in opponent_decisions if d == Action.DEFECTING)
        cnt_coop = sum(1 for d in opponent_decisions if d == Action.COOPERATING)
        if cnt_def >= cnt_coop:
            return Action.DEFECTING
        return Action.COOPERATING
    return Strategy("suspicious sophist", action)


def strategy_tit_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions):
        if len(opponent_decisions) >= 1:
            return opponent_decisions[-1]
        return Action.COOPERATING
    return Strategy("tit for tat", action)


def strategy_suspicious_tit_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions):
        if len(opponent_decisions) >= 1:
            return opponent_decisions[-1]
        return Action.DEFECTING
    return Strategy("suspicious tit for tat", action)


def strategy_forgiving_tit_for_tat() -> Strategy:
    def action(own_decisions, opponent_decisions):
        if len(opponent_decisions) >= 2 and opponent_decisions[-1] == opponent_decisions[-2]:
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


def strategy_suspicious_pavlov() -> Strategy:
    def action(own_decisions, opponent_decisions):
        # switch strategy if opponent defected
        # otherwise keep doing whatever we did last time
        if len(opponent_decisions) >= 1 and opponent_decisions[-1] == Action.DEFECTING:
            return complement_action(own_decisions[-1])
        if len(own_decisions) > 0:
            return own_decisions[-1]
        return Action.DEFECTING
    return Strategy("suspicious pavlov", action)


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
    return Strategy("pavlov spooky", action)


name2strategy = {
    "defector": strategy_defector(),
    "gandhi": strategy_gandhi(),
    "random": strategy_random(),
    "sophist": strategy_sophist(),
    "tit-for-tat": strategy_tit_for_tat(),
    "forgiving-tit-for-tat": strategy_forgiving_tit_for_tat(),
    "syspicious-tit-for-tat": strategy_suspicious_tit_for_tat(),
    "pavlov": strategy_pavlov(),
    "pavlovish": strategy_pavlovish(),
    "pavlov-spooky": strategy_pavlov_spooky(),
    "alternator": strategy_alternator(),
    "grudger": strategy_grudger(),
    "hate-opponent": strategy_hate_opponent(),
    "angry-grudger": strategy_angry_grudger(),
    "suspicious-sophist": strategy_suspicious_sophist(),
    "suspicious-pavlov": strategy_suspicious_pavlov(),
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


def get_strategies_by_name(s: str, random_if_not_found=False) -> List[Strategy]:
    if s in name2strategy:
        return [name2strategy[s]]
    if s.startswith('all'):
        excluding_strategies:List[str] = []
        remaining_str: str = s[len('all'):]
        if len(remaining_str) >= 1 and remaining_str[0] == '-':
            excluding_strategies = remaining_str[1:].split(',')
        return all_strategies_mod(excluding_strategies)
    if random_if_not_found:
        return [random_strategy()]
    return []


def participant_to_strategy_wise_results(participant_results: Dict[TournamentParticipant,int]):
    ret: DefaultDict[str,int] = defaultdict(int)
    for part, score in participant_results.items():
        ret[part.strategy.name] += score
    return ret


def str_to_argv(s: str) -> List[str]:
    return shlex.split(s)


def main():
    parser = argparse.ArgumentParser()
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument('--quiet', '-q', action='store_true', help='Just print final result')
    verbosity.add_argument('--verbose', '-v', action='count', default=0, help='Show verbose output of each game')
    parser.add_argument('--iterations', '-i', default=30, type=int, help='Number of iterations of game')
    parser.add_argument('--rounds', '-r', default=1, type=int, help='Rounds of evolution')
    parser.add_argument('--error-prob', '-ep', default=0.0, type=float, help='Probability of error due to noise (Due to noise decision gets flipped)')
    parser.add_argument('--config', '-c', default=None, type=argparse.FileType('r'), help='Configuration file.  Other options are disregarded.')
    parser.add_argument('strategies', metavar='STRATEGY', type=str, nargs='*', default=['all'],
                        help=f"Strategies for prisoners.  Possible values are: all, all-[S1,S2,...], {', '.join(all_strategies_name())}")
    args = parser.parse_args()

    config = args.config
    if config is not None:
        config_str = config.read()
        # override args
        args = parser.parse_args(str_to_argv(config_str))
    quiet = args.quiet
    verbose = args.verbose
    iterations = args.iterations
    rounds = args.rounds
    noise = args.error_prob
    strategies_name = args.strategies

    strategies = []
    for s in strategies_name:
        strategies.extend(get_strategies_by_name(s, random_if_not_found=True))

    if len(strategies) <= 1:
        # exit silently if enough strategies not provided
        return

    play_matrix = {
                    (Action.COOPERATING, Action.COOPERATING): (2, 2),
                    (Action.COOPERATING, Action.DEFECTING):   (0, 3),
                    (Action.DEFECTING,   Action.COOPERATING): (3, 0),
                    (Action.DEFECTING,   Action.DEFECTING):   (1, 1),
                  }

    participants = [TournamentParticipant(f"p{i}.{s.name}", s) for i,s in enumerate(strategies)]
    tournament = PrisonersDilemmaTournamentWithEvolution(play_matrix, participants,
                                                         iterations=iterations,
                                                         noise_error_prob=noise,
                                                         rounds_of_evolution=rounds)
    final_result = tournament.play_tournament_with_evolution(verbose, quiet=quiet)

    strategy_results = participant_to_strategy_wise_results(final_result)
    sorted_strategy_results = sorted(strategy_results.items(), key=lambda p: p[1])
    if not quiet:
        print()
        print("Strategy wise result")
        print("--------------------")
    best_strats, best_score = [], 0
    for strat,total_score in sorted_strategy_results:
        if not quiet:
            print("Strategy: {0:30} Result: {1:10}".format(strat, total_score))
        if best_score < total_score:
            best_score = total_score
            best_strats = [strat]
        elif best_score == total_score:
            best_strats.append(strat)
    if not quiet:
        print("--------------------")
        print()
    print(f"Best strategies are {', '.join(best_strats)}")
    print(f"Best score is {best_score}")


if __name__ == '__main__':
    main()
