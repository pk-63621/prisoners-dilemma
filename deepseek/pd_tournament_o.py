# pd_tournament.py (optimized)
#!/usr/bin/env python3

import abc
import shlex
import argparse
import random
import sys
import itertools
import math
import time
import numpy as np
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Set, Tuple, TypeVar, Union
from pathos.multiprocessing import ProcessPool
from strats_o import *

# Constants
COOPERATING = 0
DEFECTING = 1
PAYOFF_MATRIX = np.array([
    [[3, 3], [0, 5]],  # Cooperate row
    [[5, 0], [1, 1]]   # Defect row
], dtype=np.int16)

T = TypeVar('T')

class Logging:
    __slots__ = ('file', 'verbose', 'quiet', 'dump_trace')
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
    __slots__ = ('dict', 'cache_invalidated', 'sorted_items')
    def __init__(self):
        self.dict: DefaultDict[str, List[int]] = defaultdict(list)
        self.cache_invalidated = False
        self.sorted_items: List[Tuple[str, List[int]]] = []

    def add_score(self, name: str, score: int):
        self.dict[name].append(score)
        self.cache_invalidated = True

    def get_sorted_items(self) -> List[Tuple[str, List[int]]]:
        if self.cache_invalidated:
            self.sorted_items = sorted(self.dict.items(), key=lambda p: sum(p[1]))
            self.cache_invalidated = False
        return self.sorted_items

    def get_best_strategies_and_score(self) -> Tuple[List[str], int]:
        best_strats, best_score = [], 0
        for strat, sl in self.get_sorted_items():
            total_score = sum(sl)
            if best_score < total_score:
                best_score = total_score
                best_strats = [strat]
            elif best_score == total_score:
                best_strats.append(strrat)
        return best_strats, best_score

    def __str__(self):
        s = "Strategy wise result\n--------------------\n"
        for strat, sl in self.get_sorted_items():
            s += f"Strategy: {strat:30} Count: {len(sl):<10} Score: {sum(sl):10}\n"
        return s + "--------------------\n"

class Prisoner:
    __slots__ = ('name', 'strategy', 'jail_time', 'opponent_sum', 
                'decisions', 'opponent_decisions', 'strategy_local_state', 'idx')
    
    def __init__(self, name: str, strategy: Strategy, iterations: int):
        self.name = name
        self.strategy = strategy
        self.jail_time: int = 0
        self.opponent_sum: int = 0
        self.decisions = np.full(iterations, -1, dtype=np.int8)
        self.opponent_decisions = np.full(iterations, -1, dtype=np.int8)
        self.strategy_local_state: Dict = {}
        self.idx = 0

    def add_play(self, decision: int, play: int):
        self.jail_time += play
        self.decisions[self.idx] = decision

    def opponent_history(self, opponent_decision: int, last_play: int):
        self.opponent_sum += last_play
        self.opponent_decisions[self.idx] = opponent_decision
        self.idx += 1

    def get_decision(self) -> int:
        return self.strategy.action(
            self.decisions[:self.idx], 
            self.opponent_decisions[:self.idx], 
            self.strategy_local_state
        )

class PrisonersDilemma:
    __slots__ = ('play_matrix', 'prisoners', 'noise_probability', 'rng')
    
    def __init__(self, prisoners: List[Prisoner], noise: float = 0.0, rng_seed: int = None):
        self.play_matrix = PAYOFF_MATRIX
        self.prisoners = prisoners
        self.noise_probability = noise
        self.rng = random.Random(rng_seed) if rng_seed else random

    def noise_occurred(self) -> bool:
        return self.rng.uniform(0, 1) < self.noise_probability

    def play_next_iteration(self) -> Tuple[Tuple[int, ...], List[int]]:
        decisions = [p.get_decision() for p in self.prisoners]
        
        # Apply noise
        for i in range(len(decisions)):
            if self.noise_occurred():
                decisions[i] = 1 - decisions[i]
        
        # Get results from numpy matrix
        results = self.play_matrix[decisions[0], decisions[1]]
        
        # Update prisoners
        for i, p in enumerate(self.prisoners):
            p.add_play(decisions[i], results[i])
            opp_decision = decisions[1-i]
            p.opponent_history(opp_decision, results[1-i])
        
        return tuple(decisions), results.tolist()

class TournamentParticipant:
    __slots__ = ('name', 'strategy')
    def __init__(self, name: str, strategy: Strategy):
        self.name = name
        self.strategy = strategy

    def replicate(self, generation: int) -> 'TournamentParticipant':
        return TournamentParticipant(f"{self.name}.{generation}", self.strategy)

class TournamentParticipantResults:
    __slots__ = ('dict', 'cache_invalidated', 'sorted_items', 'stats', 
                'head_to_head', 'round_history')
    
    def __init__(self):
        self.dict: DefaultDict[TournamentParticipant, int] = defaultdict(int)
        self.cache_invalidated = False
        self.sorted_items: List[Tuple[TournamentParticipant, int]] = []
        self.stats: Dict[str, Dict] = defaultdict(dict)
        self.head_to_head: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.round_history: Dict[int, Dict[str, int]] = defaultdict(dict)

    def add_score(self, participant: TournamentParticipant, score: int, 
                 opponents: List[TournamentParticipant], round_num: int):
        self.dict[participant] += score
        self.cache_invalidated = True
        
        # Update stats
        strat_name = participant.strategy.name
        if strat_name not in self.stats:
            self.stats[strat_name] = {
                'total_score': 0,
                'games_played': 0,
                'min_score': float('inf'),
                'max_score': float('-inf'),
                'round_scores': defaultdict(list)
            }
        
        self.stats[strat_name]['total_score'] += score
        self.stats[strat_name]['games_played'] += 1
        self.stats[strat_name]['min_score'] = min(self.stats[strat_name]['min_score'], score)
        self.stats[strat_name]['max_score'] = max(self.stats[strat_name]['max_score'], score)
        self.stats[strat_name]['round_scores'][round_num].append(score)
        
        # Update head-to-head
        for opp in opponents:
            opp_name = opp.strategy.name
            self.head_to_head[strat_name][opp_name].append(score)
        
        # Update round history
        self.round_history[round_num][strat_name] = self.dict[participant]

    def get_strategy_stats(self) -> Dict[str, Dict]:
        stats = {}
        for strat, data in self.stats.items():
            stats[strat] = {
                'average_score': data['total_score'] / data['games_played'],
                'min_score': data['min_score'],
                'max_score': data['max_score'],
                'games_played': data['games_played'],
                'round_scores': dict(data['round_scores'])
            }
        return stats

    def get_sorted_items(self) -> List[Tuple[TournamentParticipant, int]]:
        if self.cache_invalidated:
            self.sorted_items = sorted(self.dict.items(), key=lambda p: p[1])
            self.cache_invalidated = False
        return self.sorted_items

    def __str__(self):
        s = "Participant results\n-------------------\n"
        for part, score in self.get_sorted_items():
            s += f"{part.name:40} {score:10}\n"
        return s + "------------------\n"

class PrisonersDilemmaTournament:
    __slots__ = ('participants', 'participants_per_game', 'iterations',
                 'noise_error_prob', 'rng_seed', 'combinations')
    
    def __init__(self, participants: List[TournamentParticipant],
                 participants_per_game: int = 2, iterations: int = 10,
                 noise_error_prob: float = 0.0, rng_seed: int = None):
        self.participants = participants
        self.participants_per_game = participants_per_game
        self.iterations = iterations
        self.noise_error_prob = noise_error_prob
        self.rng_seed = rng_seed
        self.combinations = list(itertools.combinations(range(len(participants)), participants_per_game))

    def play_game(self, combo: Tuple[int]) -> List[Tuple[int, int, List[int]]]:
        prisoners = [
            Prisoner(
                f"prisoner{i+1}.aka.{self.participants[idx].name}",
                self.participants[idx].strategy,
                self.iterations
            ) for i, idx in enumerate(combo)
        ]
        game = PrisonersDilemma(
            prisoners, 
            self.noise_error_prob,
            self.rng_seed + hash(combo) if self.rng_seed else None
        )
        for _ in range(self.iterations):
            game.play_next_iteration()
        
        # Return participant indices, scores, and opponents
        return [
            (
                idx, 
                p.jail_time,
                [x for x in combo if x != idx]  # Opponents list
            ) for idx, p in zip(combo, prisoners)
        ]
    def play_game_batch(self, batch: List[Tuple[int]]) -> List[List[Tuple[int, int]]]:
        results = []
        for combo in batch:
            # Process each combination in the batch
            prisoners = [
                Prisoner(
                    f"prisoner{i+1}.aka.{self.participants[idx].name}",
                    self.participants[idx].strategy,
                    self.iterations
                )
                for i, idx in enumerate(combo)  # Iterate through combination indices
            ]
            game = PrisonersDilemma(
                prisoners, 
                self.noise_error_prob,
                self.rng_seed + hash(combo) if self.rng_seed else None
            )
            for _ in range(self.iterations):
                game.play_next_iteration()
            results.append([(idx, p.jail_time) for idx, p in zip(combo, prisoners)])
        return results
    def play_tournament(self, logging: Logging) -> TournamentParticipantResults:
        outcome = TournamentParticipantResults()
        batch_size = 100
        
        with ProcessPool() as pool:
            batches = [self.combinations[i:i+batch_size] 
                      for i in range(0, len(self.combinations), batch_size)]
            
            for batch_index, batch in enumerate(batches):
                batch_results = pool.uimap(self.play_game, batch)
                for game_index, game_result in enumerate(batch_results):
                    round_num = batch_index * batch_size + game_index + 1
                    for idx, score, opponents in game_result:
                        outcome.add_score(
                            self.participants[idx],
                            score,
                            opponents=[self.participants[o] for o in opponents],
                            round_num=round_num
                        )
        return outcome
def main():
    parser = argparse.ArgumentParser()
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    verbosity.add_argument('--verbose', '-v', action='count', default=0, help='Verbosity level')
    parser.add_argument('--iterations', '-i', type=int, default=1000, help='Iterations per game')
    parser.add_argument('--rounds', '-r', type=int, default=5, help='Evolution rounds')
    parser.add_argument('--noise', '-n', type=float, default=0.0, help='Noise probability')
    parser.add_argument('--parallel', '-j', action='store_true', help='Use multiprocessing')
    parser.add_argument('--visualize', '-V', action='store_true', help='Show visualizations after tournament')
    parser.add_argument('--noise-analysis', '-N', action='store_true', help='Run noise impact analysis')
    args = parser.parse_args()

    logging = Logging(verbose=args.verbose, quiet=args.quiet)
    strategies = [s for s in all_strategies() if s.name != 'user']
    participants = [TournamentParticipant(f"p{i}", s) for i, s in enumerate(strategies)]

    if args.noise_analysis:
        noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
        noise_results = {}
        
        for noise in noise_levels:
            round_scores = defaultdict(list)
            for round in range(1, args.rounds + 1):
                tournament = PrisonersDilemmaTournament(
                    participants,
                    iterations=args.iterations,
                    noise_error_prob=noise,
                    rng_seed=int(time.time())
                )
                results = tournament.play_tournament(logging)
                
                # Collect results
                for part, score in results.dict.items():
                    round_scores[part.strategy.name].append(score)
            
            # Calculate average scores
            noise_results[noise] = {
                strat: np.mean(scores) for strat, scores in round_scores.items()
            }
        
        if args.visualize:
            from visualization import plot_noise_impact
            plot_noise_impact(noise_levels, noise_results)
        
        return
    
    # Regular tournament
    cumulative_scores = defaultdict(list)
    head_to_head = defaultdict(lambda: defaultdict(list))
    round_history = defaultdict(dict)
    
    for round in range(1, args.rounds + 1):
        tournament = PrisonersDilemmaTournament(
            participants,
            iterations=args.iterations,
            noise_error_prob=args.noise,
            rng_seed=int(time.time())
        )
        results = tournament.play_tournament(logging)
        
        # Collect results
        for part, score in results.dict.items():
            cumulative_scores[part.strategy.name].append(score)
            round_history[round][part.strategy.name] = results.dict[part]
            
            # Update head-to-head
            for opp in results.head_to_head[part.strategy.name]:
                head_to_head[part.strategy.name][opp].append(score)

    # Analysis and visualization
    if args.visualize:
        from visualization import (plot_strategy_performance, plot_score_distribution,
                                 plot_head_to_head, plot_evolution)
        
        # Get final stats
        final_stats = results.get_strategy_stats()
        
        # Plot performance
        plot_strategy_performance(final_stats)
        
        # Plot score distribution
        plot_score_distribution(cumulative_scores)
        
        # Plot head-to-head
        plot_head_to_head(head_to_head)
        
        # Plot evolution
        plot_evolution(round_history)

    # Print final results
    sorted_scores = sorted(cumulative_scores.items(), 
                          key=lambda x: sum(x[1]), 
                          reverse=True)
    print("\n=== Final Results ===")
    for strat, scores in sorted_scores:
        print(f"{strat:40} {sum(scores):10} (avg: {sum(scores)/len(scores):.1f})")

if __name__ == '__main__':
    main()
