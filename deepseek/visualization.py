import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List
from matplotlib.colors import LinearSegmentedColormap

def plot_strategy_performance(stats: Dict[str, Dict]):
    strategies = list(stats.keys())
    avg_scores = [s['average_score'] for s in stats.values()]
    min_scores = [s['min_score'] for s in stats.values()]
    max_scores = [s['max_score'] for s in stats.values()]
    
    x = np.arange(len(strategies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, min_scores, width, label='Min Score')
    rects2 = ax.bar(x + width/2, max_scores, width, label='Max Score')
    ax.plot(x, avg_scores, 'ro-', label='Average Score')

    ax.set_xlabel('Strategies')
    ax.set_ylabel('Scores')
    ax.set_title('Strategy Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.legend()

    fig.tight_layout()
    plt.show()

def plot_score_distribution(scores: Dict[str, List[int]]):
    plt.figure(figsize=(14, 8))
    for strat, scores in scores.items():
        sns.kdeplot(scores, label=strat, alpha=0.6, linewidth=2)
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.title('Score Distribution by Strategy')
    plt.legend()
    plt.show()

def plot_head_to_head(head_to_head: Dict[str, Dict[str, List[int]]]):
    strategies = sorted(head_to_head.keys())
    matrix = np.zeros((len(strategies), len(strategies)))
    
    for i, strat1 in enumerate(strategies):
        for j, strat2 in enumerate(strategies):
            if strat2 in head_to_head[strat1]:
                matrix[i, j] = np.mean(head_to_head[strat1][strat2])
    
    plt.figure(figsize=(12, 10))
    cmap = LinearSegmentedColormap.from_list('rg', ["red", "yellow", "green"])
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap=cmap,
                xticklabels=strategies, yticklabels=strategies)
    plt.title('Head-to-Head Performance')
    plt.xlabel('Opponent Strategy')
    plt.ylabel('Strategy')
    plt.show()

def plot_evolution(round_history: Dict[int, Dict[str, int]]):
    plt.figure(figsize=(14, 8))
    rounds = sorted(round_history.keys())
    
    # Get all strategies
    strategies = set()
    for r in rounds:
        strategies.update(round_history[r].keys())
    
    # Plot each strategy's evolution
    for strat in strategies:
        scores = [round_history[r].get(strat, 0) for r in rounds]
        plt.plot(rounds, scores, label=strat, linewidth=2)
    
    plt.xlabel('Round')
    plt.ylabel('Cumulative Score')
    plt.title('Strategy Evolution Over Rounds')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_noise_impact(noise_levels: List[float], results: Dict[float, Dict[str, float]]):
    plt.figure(figsize=(14, 8))
    
    # Get all strategies
    strategies = set()
    for noise_result in results.values():
        strategies.update(noise_result.keys())
    
    # Plot each strategy's performance
    for strat in strategies:
        scores = [results[noise].get(strat, 0) for noise in noise_levels]
        plt.plot(noise_levels, scores, label=strat, linewidth=2)
    
    plt.xlabel('Noise Level')
    plt.ylabel('Average Score')
    plt.title('Strategy Performance vs Noise Level')
    plt.legend()
    plt.grid(True)
    plt.show()
