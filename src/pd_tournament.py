#!/usr/bin/env python3

import shlex
import argparse
import random
import itertools
import math

from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple, TypeVar, Union

from strats import *


T = TypeVar('T')


def unpack_tuple_if_singleton(a: Tuple[T,...]) -> Union[Tuple[T,...],T]:
    if len(a) == 1:
        return a[0]
    return a


class Logging:
    def __init__(self, verbose=0, quiet=False, dump_trace=False):
        self.verbose = verbose
        self.quiet = quiet
        self.dump_trace = dump_trace


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
        best_strats, best_score = [], 0
        for strat,sl in self.get_sorted_items():
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


class Prisoner:
    def __init__(self, name: str, strategy: Strategy):
        assert strategy is not None
        self.name = name
        self.jail_time: int = 0
        self.opponent_sum: int = 0
        self.opponent_decisions: List[Action] = []
        self.decisions: List[Action] = []
        self.strategy = strategy
        self.strategy_local_state: Dict = {}

    def add_play(self, decision, play):
        self.jail_time += int(play)
        self.decisions.append(decision)

    def opponent_history(self, opponent_decision, last_play):
        self.opponent_sum += int(last_play)
        self.opponent_decisions.append(opponent_decision)

    def get_result(self) -> int:
        return self.jail_time

    def get_decision(self) -> Action:
        return self.strategy.action(self.decisions, self.opponent_decisions, self.strategy_local_state)

def play_matrix_is_well_formed(play_matrix: Dict) -> bool:
    """
    Terminology:
    * Reward (R)     : both cooperate
    * Temptation (T) : player defect, opponent cooperate
    * Punishment (P) : both defect
    * Sucker (S)     : player cooperate, opponent defect (opposite of T)

    player's matrix:
     R S
     T P

    opponent's matrix:
     R T
     S P

    Requirements for prisoner dilemma matrix:
    * T > R > P > S
    * 2R > T+S

    """

    Rp, Ro = play_matrix[(Action.COOPERATING, Action.COOPERATING)]
    Sp, To = play_matrix[(Action.COOPERATING, Action.DEFECTING)]
    Tp, So = play_matrix[(Action.DEFECTING, Action.COOPERATING)]
    Pp, Po = play_matrix[(Action.DEFECTING, Action.DEFECTING)]

    if Rp != Ro or Sp != So or Tp != To or Pp != Po:
        # print("player and opponent matrices are not synced with each other")
        return False

    if not (Tp > Rp and Rp > Pp and Pp > Sp):
        # print("ordering requirement not met")
        return False

    if not (2*Rp > Tp + Sp):
        # print("reward, temptation inequality not satisfied")
        return False

    return True

class PrisonersDilemma:
    def __init__(self, matrix: Dict, prisoners: List[Prisoner], noise=0.0, rng_seed=None):
        assert len(matrix) != 0
        assert len(prisoners) >= 2
        assert len(list(matrix.items())[0][0]) == len(prisoners)
        self.play_matrix = matrix
        self.prisoners = prisoners
        self.noise_probability = noise
        self.rng = random.Random(rng_seed)

    def get_result(self, decisions):
        return self.play_matrix[decisions]

    def noise_occurred(self):
        return self.rng.uniform(0, 1) < self.noise_probability

    def play_next_iteration(self) -> Tuple[Tuple[Action, ...], List[int]]:
        decisions             = tuple(p.get_decision() for p in self.prisoners)
        decisions_after_noise = tuple(complement_action(d) if self.noise_occurred() else d for i,d in enumerate(decisions))
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
        assert strategy is not None
        self.name = name
        self.strategy = strategy

    def replicate(self, generation: int):
        new_name = f"{self.name}.{generation}"
        new_strategy = self.strategy
        return TournamentParticipant(new_name, new_strategy)


class TournamentParticipantResults:
    def __init__(self):
        self.dict: DefaultDict[TournamentParticipant,int] = defaultdict(int)
        self.cache_invalidated = False
        self.sorted_items: List[Tuple[TournamentParticipant,int]] = []

    def add_score(self, name: TournamentParticipant, score: int):
        self.dict[name] += score
        self.cache_invalidated = True

    def items(self):
        return self.dict.items()

    def len(self):
        return len(self.dict)

    def get_sorted_items(self) -> List[Tuple[TournamentParticipant,int]]:
        if self.cache_invalidated:
            self.sorted_items = sorted(self.dict.items(), key=lambda p: p[1])
            self.cache_invalidated = False
        return self.sorted_items

    def print(self):
        print()
        print("Participant results")
        print("-------------------")
        for part,score in self.get_sorted_items():
            print("\t{0:40} {1:10}".format(part.name, score))
        print("------------------")
        print()


def participant_to_strategy_wise_results(participant_results: TournamentParticipantResults) -> StrategyResults:
    ret = StrategyResults()
    for part, score in participant_results.items():
        ret.add_score(part.strategy.name, score)
    return ret


# all participants play against each other
class PrisonersDilemmaTournament:
    def __init__(self, play_matrix, participants: List[TournamentParticipant],
                 participants_per_game=2,
                 iterations=10,
                 noise_error_prob=0.0,
                 rng_seed=None):
        assert len(participants) >= 2
        assert participants_per_game >= 2
        assert iterations >= 1
        self.play_matrix = play_matrix
        self.participants = participants
        self.participants_per_game = participants_per_game
        self.iterations = iterations
        self.noise_error_prob = noise_error_prob
        self.rng_seed = rng_seed

    def play_tournament(self, logging: Logging) -> TournamentParticipantResults:
        # outcome accumulated across all rounds
        outcome = TournamentParticipantResults()
        rng_seed = self.rng_seed
        if not logging.quiet:
            print(f"Tournament participants[{len(self.participants)}]: {', '.join(s.name for s in self.participants)}")
        for r, round_participants in enumerate(itertools.combinations(self.participants, self.participants_per_game)):
            r += 1
            prisoners = [Prisoner(f"prisoner{i+1}.aka.{part.name}", part.strategy) for i,part in enumerate(round_participants)]
            game = PrisonersDilemma(self.play_matrix, prisoners, self.noise_error_prob, rng_seed)
            if rng_seed is not None:
                rng_seed = rng_seed+1

            if logging.verbose >= 3:
                print()
                print(f"=== Tournament Round #{r} ===")
                print()
                print(f"Participants: {', '.join(p.name for p in prisoners)}")
                print()

            for i in range(self.iterations):
                decisions, results = game.play_next_iteration()
                s = f"Iteration #{i+1: <3}:"
                if logging.verbose >= 3:
                    print(f"{s} Actions: {', '.join(d.value for d in decisions)}")
                    print(f"{' '*len(s)} Result: {', '.join(str(r) for r in results)}")
                    print()
                    pass

            if logging.verbose >= 2:
                print(f"Result for Round #{r}:")
            for prisoner,participant in zip(prisoners,round_participants):
                result = prisoner.get_result()
                outcome.add_score(participant, result)
                if logging.dump_trace:
                    print(f"{participant.name: <{40}} {action_trace(prisoner.decisions)}")
                if logging.verbose >= 2:
                    print(f"\t{participant.name: <{40}} {result}")

        if logging.verbose >= 1:
            outcome.print()
            strategy_results = participant_to_strategy_wise_results(outcome)
            strategy_results.print()
        return outcome


class PrisonersDilemmaTournamentWithEvolution:
    def __init__(self, play_matrix, participants: List[TournamentParticipant],
                 participants_per_game=2,
                 iterations=10,
                 noise_error_prob=0.0,
                 rng_seed: Optional[int]=None,
                 fraction_eliminated_after_each_tournament=0.1,
                 rounds_of_evolution=2):
        assert fraction_eliminated_after_each_tournament > 0.0
        assert rounds_of_evolution >= 1
        self.play_matrix = play_matrix
        self.orig_participants = participants
        self.iterations = iterations
        self.noise_error_prob = noise_error_prob
        self.participants_per_game = participants_per_game
        self.rng_seed = rng_seed
        self.fraction_eliminated_after_each_tournament = fraction_eliminated_after_each_tournament
        self.rounds_of_evolution = rounds_of_evolution

    def play_tournament(self, participants, logging: Logging) -> TournamentParticipantResults:
        tournament = PrisonersDilemmaTournament(self.play_matrix, participants,
                                                participants_per_game=self.participants_per_game,
                                                iterations=self.iterations,
                                                noise_error_prob=self.noise_error_prob,
                                                rng_seed=self.rng_seed)
        if self.rng_seed is not None:
            self.rng_seed += 1
        return tournament.play_tournament(logging)

    def eliminate_and_replicate(self,
                                last_participants: List[TournamentParticipant],
                                last_outcome: TournamentParticipantResults,
                                generation: int, logging: Logging) -> List[TournamentParticipant]:
        len_last_participants = len(last_participants)
        assert last_outcome is not None
        assert last_outcome.len() == len_last_participants
        last_outcome_sorted = last_outcome.get_sorted_items()
        nr = math.ceil(len_last_participants*self.fraction_eliminated_after_each_tournament)
        assert nr != 0
        to_be_eliminated = set([p.name for p,_ in last_outcome_sorted[:nr]])
        to_be_replicated = set([p.name for p,_ in last_outcome_sorted[-nr:]])
        assert len(to_be_eliminated) == len(to_be_replicated)
        new_participants = [p for p in last_participants if p.name not in to_be_eliminated]
        new_participants.extend([p.replicate(generation) for p in last_participants if p.name in to_be_replicated])
        if logging.verbose >= 1:
            print()
            print(f"*** Eliminating bottom {nr} and replicating top {nr}")
            print()
        if logging.verbose >= 2:
            print()
            print(f"*** Eliminated: {', '.join(to_be_eliminated)}")
            print(f"*** Replicated: {', '.join(to_be_replicated)}")
            print()
        assert len(new_participants) == len_last_participants
        return new_participants

    def play_tournament_with_evolution(self, logging: Logging) -> Optional[TournamentParticipantResults]:
        last_outcome = None
        last_participants = self.orig_participants
        for i in range(self.rounds_of_evolution):
            last_outcome = self.play_tournament(last_participants, logging)
            last_participants = self.eliminate_and_replicate(last_participants, last_outcome, i+1, logging)
        assert last_outcome is not None
        return last_outcome


def str_to_argv(s: str) -> List[str]:
    return shlex.split(s)


def read_config(config):
    ret = ''
    for line in config:
        line = line.strip()
        if line.startswith('#'):
            continue
        ret += line
    return ret


def main():
    parser = argparse.ArgumentParser()
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument('--quiet', '-q', action='store_true', default=False, help='Just print final result')
    verbosity.add_argument('--verbose', '-v', action='count', default=0, help='Show verbose output of each game')
    parser.add_argument('--dump-trace', '-d', action='store_true', default=False,
                        help='Dump decisions made by a participant (without factoring in noise) for each round and iteration as a string.')
    parser.add_argument('--iterations', '-i', default=30, type=int, help='Number of iterations of game')
    parser.add_argument('--rounds', '-r', default=1, type=int, help='Rounds of evolution')
    parser.add_argument('--error-prob', '-ep', default=0.0, type=float, help='Probability of error due to noise (Due to noise decision gets flipped)')
    parser.add_argument('--rng-seed', '-s', default=None, type=int, help='Seed to be passed to RNG')
    parser.add_argument('--config', '-c', default=None, type=argparse.FileType('r'), help='Configuration file.  Other options are disregarded.')
    parser.add_argument('strategies', metavar='STRATEGY', type=str, nargs='*', default=['all-user'],
                        help=f"Strategies for prisoners.  Use multiplier '*N' for specifying multiple copies (e.g. all*4)."
                             f"Possible values are: all, all-[S1,S2,...], {', '.join(all_strategies_name())}")
    args = parser.parse_args()

    config = args.config
    if config is not None:
        config_str = read_config(config)
        if not args.quiet:
            print(f"Using args from config: {config_str}")
        # override args
        args = parser.parse_args(str_to_argv(config_str))
    logging = Logging(verbose=args.verbose, quiet=args.quiet, dump_trace=args.dump_trace)
    iterations = args.iterations
    rounds = args.rounds
    noise = args.error_prob
    rng_seed = args.rng_seed
    strategies_name = args.strategies

    strategies = []
    for s in strategies_name:
        strategies.extend(get_strategies_by_name(s, random_if_not_found=True))

    if len(strategies) <= 1:
        # exit silently if enough strategies not provided
        return

    play_matrix = {
                    (Action.COOPERATING, Action.COOPERATING): (3, 3),
                    (Action.COOPERATING, Action.DEFECTING):   (0, 5),
                    (Action.DEFECTING,   Action.COOPERATING): (5, 0),
                    (Action.DEFECTING,   Action.DEFECTING):   (1, 1),
                  }
    assert play_matrix_is_well_formed(play_matrix)

    participants = [TournamentParticipant(f"p{i}.{s.name}", s) for i,s in enumerate(strategies)]
    tournament = PrisonersDilemmaTournamentWithEvolution(play_matrix, participants,
                                                         iterations=iterations,
                                                         noise_error_prob=noise,
                                                         rng_seed=rng_seed,
                                                         rounds_of_evolution=rounds)
    final_result = tournament.play_tournament_with_evolution(logging)

    strategy_results = participant_to_strategy_wise_results(final_result)
    best_strats, best_score = strategy_results.get_best_strategies_and_score()
    if not logging.quiet:
        strategy_results.print()
    if len(best_strats) > 1:
        print(f"Best strategies are {', '.join(best_strats)}")
    else:
        print(f"Best strategy is {', '.join(best_strats)}, congrats")
    print(f"Best score is {best_score}")


if __name__ == '__main__':
    main()
