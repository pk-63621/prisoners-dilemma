#!/usr/bin/env python3

import abc
import shlex
import argparse
import random
import sys
import itertools
import math

from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Set, Tuple, TypeVar, Union
from pathos.multiprocessing import ProcessPool

from strats import *


T = TypeVar('T')


class Logging:
    def __init__(self, file=sys.stdout, verbose=0, quiet=False, dump_trace=False):
        self.file = file
        self.verbose = verbose
        self.quiet = quiet
        self.dump_trace = dump_trace

    def logQ(self, s: str):
        if not self.quiet:
            self.log(s)

    def logV(self, lvl: int, s: str):
        if lvl <= self.verbose:
            self.log(s)

    def logT(self, s: str):
        if self.dump_trace:
            self.log(s)

    def log(self, s: str):
        print(s, file=self.file)


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

    def __str__(self):
        s =  "Strategy wise result\n"
        s += "--------------------\n"
        for strat,sl in self.get_sorted_items():
            s += "Strategy: {0:30} Count: {1:<10} Score: {2:10}\n".format(strat, len(sl), sum(sl))
        s += "--------------------\n"
        return s


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
        print("player and opponent matrices are not synced with each other")
        return False

    if not (Tp > Rp and Rp > Pp and Pp > Sp):
        print("ordering requirement not met")
        return False

    if not (2*Rp > Tp + Sp):
        print("reward, temptation inequality not satisfied")
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

    def noise_occurred(self):
        if self.noise_probability == 0.0:
            return False
        return self.rng.uniform(0, 1) < self.noise_probability

    def play_next_iteration(self) -> Tuple[Tuple[Action, ...], List[int]]:
        decisions             = tuple(p.get_decision() for p in self.prisoners)
        decisions_after_noise = tuple(complement_action(d) if self.noise_occurred() else d for i,d in enumerate(decisions))
        results = self.play_matrix[decisions_after_noise]

        for i in range(len(self.prisoners)):
            self.prisoners[i].add_play(decisions[i], results[i])
            opponents_decisions = decisions_after_noise[:i] + decisions_after_noise[i+1:]
            opponents_results   = results[:i] + results[i+1:]
            # HACK for 2 players only game, unpack the tuples
            unpacked_opponents_decisions = opponents_decisions[0]
            unpacked_opponents_results   = opponents_results[0]
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

    def get_score(self, name: TournamentParticipant) -> int:
        return self.dict[name]

    def items(self):
        return self.dict.items()

    def len(self):
        return len(self.dict)

    def get_sorted_items(self) -> List[Tuple[TournamentParticipant,int]]:
        if self.cache_invalidated:
            self.sorted_items = sorted(self.dict.items(), key=lambda p: p[1])
            self.cache_invalidated = False
        return self.sorted_items

    def __str__(self):
        s =  "Participant results\n"
        s += "-------------------\n"
        for part,score in self.get_sorted_items():
            s += "\t{0:40} {1:10}\n".format(part.name, score)
        s += "------------------\n"
        return s


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

    def use_mp(self):
        global mp_iters_threshold
        return self.iterations > mp_iters_threshold

    #def play_game(self, round_participants_idx: List[int]) -> List[Tuple[int,int]]:
    def play_game(self, round_participants_idx: List[int]):
        round_participants = [self.participants[i] for i in round_participants_idx]
        prisoners = [Prisoner(f"prisoner{i+1}.aka.{part.name}", part.strategy) for i,part in enumerate(round_participants)]
        game = PrisonersDilemma(self.play_matrix, prisoners, self.noise_error_prob, self.rng_seed)
        #if rng_seed is not None:
        #    rng_seed = rng_seed+1
        #logging.logV(3, f"\n=== Tournament Round ===\nParticipants: {', '.join(p.name for p in prisoners)}\n")
        for i in range(self.iterations):
            decisions, results = game.play_next_iteration()
            #s = f"Iteration #{i+1: <3}:"
            #logging.logV(3, f"{s} Actions: {', '.join(d.value for d in decisions)}\n{' '*len(s)} Result: {', '.join(str(r) for r in results)}\n")
        #logging.logV(2, "Result for Round:")
        ret = []
        for prisoner,pidx in zip(prisoners,round_participants_idx):
            result = prisoner.get_result()
            ret.append((pidx, result))
            #participant = self.participants[pidx]
            #logging.logT(f"{participant.name: <{40}} {action_trace(prisoner.decisions)}")
            #logging.logV(2, "\t{participant.name: <{40}} {result}")
        return ret

    def play_tournament(self, logging: Logging) -> TournamentParticipantResults:
        # outcome accumulated across all rounds
        outcome = TournamentParticipantResults()
        logging.logQ(f"Tournament participants[{len(self.participants)}]: {', '.join(s.name for s in self.participants)}")
        games = itertools.combinations(range(len(self.participants)), self.participants_per_game)
        if self.use_mp():
            round_results = ProcessPool().uimap(self.play_game, games)
        else:
            round_results = map(self.play_game, games)
        for res in round_results:
            for pidx,score in res:
                part = self.participants[pidx]
                outcome.add_score(part, score)
        #logging.logV(1, f"{outcome}\n{participant_to_strategy_wise_results(outcome)}")
        return outcome


class PrisonersDilemmaTournamentWithEvolutionBase(metaclass=abc.ABCMeta):
    def __init__(self, play_matrix, participants: List[TournamentParticipant],
                 participants_per_game=2,
                 iterations=10,
                 noise_error_prob=0.0,
                 rng_seed: Optional[int]=None,
                 rounds_of_evolution=2):
        assert rounds_of_evolution >= 1
        self.play_matrix = play_matrix
        self.orig_participants = participants
        self.iterations = iterations
        self.noise_error_prob = noise_error_prob
        self.participants_per_game = participants_per_game
        self.rng_seed = rng_seed
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

    @abc.abstractmethod
    def get_replication_and_elimination_candidates(self, participants: List[TournamentParticipant],
                                                   outcome: TournamentParticipantResults) -> Tuple[Set[str],Set[str]]:
        pass

    def eliminate_and_replicate(self,
                                last_participants: List[TournamentParticipant],
                                last_outcome: TournamentParticipantResults,
                                generation: int, logging: Logging) -> List[TournamentParticipant]:
        len_last_participants = len(last_participants)
        assert last_outcome is not None
        assert last_outcome.len() == len_last_participants

        to_be_replicated, to_be_eliminated = self.get_replication_and_elimination_candidates(last_participants, last_outcome)
        assert len(to_be_eliminated) == len(to_be_replicated)

        new_participants = [p for p in last_participants if p.name not in to_be_eliminated]
        new_participants.extend([p.replicate(generation) for p in last_participants if p.name in to_be_replicated])
        assert len(new_participants) == len_last_participants

        logging.logV(1, f"\n*** Eliminating {len(to_be_eliminated)} and replicating {len(to_be_replicated)}\n")
        logging.logV(2, f"*** Eliminated: {', '.join(to_be_eliminated)}\n*** Replicated: {', '.join(to_be_replicated)}\n")
        return new_participants

    def play_tournament_with_evolution(self, logging: Logging) -> Optional[TournamentParticipantResults]:
        last_outcome = None
        last_participants = self.orig_participants
        for i in range(self.rounds_of_evolution):
            last_outcome = self.play_tournament(last_participants, logging)
            if i == self.rounds_of_evolution-1:
                continue
            last_participants = self.eliminate_and_replicate(last_participants, last_outcome, i+1, logging)
        assert last_outcome is not None
        return last_outcome


class PrisonersDilemmaTournamentWithEvolutionTopReplicated(PrisonersDilemmaTournamentWithEvolutionBase):
    def __init__(self, play_matrix, participants: List[TournamentParticipant],
                 participants_per_game=2,
                 iterations=10,
                 noise_error_prob=0.0,
                 rng_seed: Optional[int]=None,
                 rounds_of_evolution=2,
                 fraction_eliminated_after_each_tournament=0.1):
        assert fraction_eliminated_after_each_tournament > 0.0
        super().__init__(play_matrix, participants, participants_per_game, iterations, noise_error_prob, rng_seed, rounds_of_evolution)
        self.fraction_eliminated_after_each_tournament = fraction_eliminated_after_each_tournament

    def get_replication_and_elimination_candidates(self, participants: List[TournamentParticipant],
                                                   outcome: TournamentParticipantResults) -> Tuple[Set[str],Set[str]]:
        outcome_sorted = outcome.get_sorted_items()
        nr = math.ceil(len(participants)*self.fraction_eliminated_after_each_tournament)
        assert nr != 0
        to_be_replicated = set([p.name for p,_ in outcome_sorted[-nr:]])
        to_be_eliminated = set([p.name for p,_ in outcome_sorted[:nr]])
        return to_be_replicated, to_be_eliminated


class MoranProcess(PrisonersDilemmaTournamentWithEvolutionBase):
    def __init__(self, play_matrix, participants: List[TournamentParticipant],
                 participants_per_game=2,
                 iterations=10,
                 noise_error_prob=0.0,
                 rng_seed: Optional[int]=None,
                 rounds_of_evolution=2):
        super().__init__(play_matrix, participants, participants_per_game, iterations, noise_error_prob, rng_seed, rounds_of_evolution)

    def get_replication_and_elimination_candidates(self, participants: List[TournamentParticipant],
                                                   outcome: TournamentParticipantResults) -> Tuple[Set[str],Set[str]]:
        participants_weights = [outcome.get_score(part) for part in participants]
        to_be_replicated = random.choices(participants, participants_weights)
        to_be_eliminated = random.choice(participants)
        return set([to_be_replicated[0].name]), set([to_be_eliminated.name])


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


def create_play_matrix(r: int, t: int, s: int, p: int) -> Dict:
    return {
               (Action.COOPERATING, Action.COOPERATING): (r, r),
               (Action.COOPERATING, Action.DEFECTING):   (s, t),
               (Action.DEFECTING,   Action.COOPERATING): (t, s),
               (Action.DEFECTING,   Action.DEFECTING):   (p, p),
           }


def parse_play_matrix(s: str) -> Dict:
    d = { p.split('=')[0]: int(p.split('=')[1]) for p in s.split(',') }
    play_matrix = create_play_matrix(r=d['r'], t=d['t'], s=d['s'], p=d['p'])
    assert play_matrix_is_well_formed(play_matrix)
    return play_matrix


mp_iters_threshold = 700 # use multiprocessing if # of iterations exceed threshold

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
    parser.add_argument('--play-matrix', '-p', default='r=3,t=5,s=0,p=1', type=str, help='Utility matrix for the game.  Expected format: '
                                                                                         'r=<REWARD>,t=<TEMPTATION>,s=<SUCKER>,p=<PUNISHMENT>.  '
                                                                                         'Example: r=3,t=5,s=0,p=1')
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
    play_matrix = parse_play_matrix(args.play_matrix)
    strategies_name = args.strategies

    strategies = []
    for s in strategies_name:
        strategies.extend(get_strategies_by_name(s, random_if_not_found=True))

    if len(strategies) <= 1:
        # exit silently if enough strategies not provided
        return

    participants = [TournamentParticipant(f"p{i}.{s.name}", s) for i,s in enumerate(strategies)]
    #tournament = PrisonersDilemmaTournamentWithEvolutionTopReplicated(play_matrix, participants,
    #                                                                  iterations=iterations,
    #                                                                  noise_error_prob=noise,
    #                                                                  rng_seed=rng_seed,
    #                                                                  rounds_of_evolution=rounds)
    tournament = MoranProcess(play_matrix, participants,
                              iterations=iterations,
                              noise_error_prob=noise,
                              rng_seed=rng_seed,
                              rounds_of_evolution=rounds)
    final_result = tournament.play_tournament_with_evolution(logging)

    strategy_results = participant_to_strategy_wise_results(final_result)
    best_strats, best_score = strategy_results.get_best_strategies_and_score()
    logging.logQ(f"{strategy_results}")
    if len(best_strats) > 1:
        print(f"Best strategies are {', '.join(best_strats)}")
    else:
        print(f"Best strategy is {', '.join(best_strats)}, congrats")
    print(f"Best score is {best_score}")


if __name__ == '__main__':
    main()
