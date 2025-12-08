"""
N-Queens Problem Solver
=======================

Implementation of the N-Queens problem using multiple algorithms:
1. Backtracking (DFS with pruning)
2. Genetic Algorithm
3. Min-Conflicts Heuristic

Author: Parth Oza
"""

import random
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import itertools


@dataclass
class AlgorithmConfig:
    """Configuration for algorithms"""
    max_generations: int = 1000
    population_size: int = 100
    mutation_rate: float = 0.1
    elite_size: int = 10
    tournament_size: int = 5
    verbose: bool = True
    crossover_rate: float = 0.8


class NQueensSolver(ABC):
    """Abstract base class for N-Queens solvers"""
    
    def __init__(self, n: int):
        self.n = n
        self.solutions = []
        self.execution_time = 0
        self.solution_count = 0
        
    @abstractmethod
    def solve(self, **kwargs) -> List[List[int]]:
        """Solve the N-Queens problem"""
        pass
    
    def visualize_solution(self, solution: List[int], title: str = "", 
                          save_path: Optional[str] = None):
        """
        Visualize a single solution
        
        Args:
            solution: List of queen positions
            title: Plot title
            save_path: Path to save the image (optional)
        """
        board = [[0] * self.n for _ in range(self.n)]
        for col, row in enumerate(solution):
            board[row][col] = 1
            
        fig, ax = plt.subplots(figsize=(max(6, self.n), max(6, self.n)))
        ax.set_aspect('equal')
        
        # Create chessboard pattern
        for i in range(self.n):
            for j in range(self.n):
                color = '#f0d9b5' if (i + j) % 2 == 0 else '#b58863'  # Chess colors
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
        
        # Add queens
        queen_color = 'black'
        for col, row in enumerate(solution):
            # Draw queen as a circle with a crown symbol
            ax.add_patch(plt.Circle((col + 0.5, row + 0.5), 0.35,
                                   color=queen_color, ec='gold', lw=2))
            # Add Q symbol
            ax.text(col + 0.5, row + 0.5, '♛', fontsize=24,
                   ha='center', va='center', color='gold')
        
        # Setup axes
        ax.set_xlim(0, self.n)
        ax.set_ylim(0, self.n)
        ax.set_xticks(range(self.n))
        ax.set_yticks(range(self.n))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Add labels
        for i in range(self.n):
            ax.text(i + 0.5, -0.1, chr(ord('A') + i), 
                   ha='center', va='center', fontweight='bold')
            ax.text(-0.1, i + 0.5, str(i + 1),
                   ha='center', va='center', fontweight='bold')
        
        # Title
        if not title:
            title = f"N-Queens Solution (N={self.n})"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Grid
        ax.grid(True, alpha=0.3, color='black', linewidth=0.5)
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Solution saved to {save_path}")
        
        plt.show()
    
    def validate_solution(self, solution: List[int]) -> bool:
        """Validate if a solution is correct"""
        if len(solution) != self.n:
            return False
        
        # Check rows and diagonals
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if solution[i] == solution[j]:  # Same row
                    return False
                if abs(solution[i] - solution[j]) == abs(i - j):  # Same diagonal
                    return False
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get solver statistics"""
        return {
            'board_size': self.n,
            'solutions_found': len(self.solutions),
            'total_solutions': self.solution_count,
            'execution_time': self.execution_time,
            'algorithm': self.__class__.__name__
        }


class BacktrackingSolver(NQueensSolver):
    """Backtracking solver with pruning and symmetry optimization"""
    
    def __init__(self, n: int):
        super().__init__(n)
        
    def solve(self, count_only: bool = False, find_all: bool = True, 
              max_solutions: int = 1000) -> List[List[int]]:
        """
        Solve using backtracking with pruning
        
        Args:
            count_only: Only count solutions, don't store them
            find_all: Find all solutions (if False, stops at first)
            max_solutions: Maximum number of solutions to find
            
        Returns:
            List of solutions (each solution is list of row positions)
        """
        start_time = time.time()
        self.solutions = []
        self.solution_count = 0
        
        # Use sets for O(1) conflict checking
        occupied_rows = set()
        occupied_diag1 = set()  # row + col
        occupied_diag2 = set()  # row - col
        
        def backtrack(col: int, board: List[int]):
            if col == self.n:
                if not count_only:
                    self.solutions.append(board.copy())
                self.solution_count += 1
                return True if not find_all else False
            
            # Try all rows
            for row in range(self.n):
                diag1 = row + col
                diag2 = row - col
                
                # Check if position is safe
                if (row not in occupied_rows and 
                    diag1 not in occupied_diag1 and 
                    diag2 not in occupied_diag2):
                    
                    # Place queen
                    board.append(row)
                    occupied_rows.add(row)
                    occupied_diag1.add(diag1)
                    occupied_diag2.add(diag2)
                    
                    # Recurse
                    found = backtrack(col + 1, board)
                    
                    # Remove queen (backtrack)
                    board.pop()
                    occupied_rows.remove(row)
                    occupied_diag1.remove(diag1)
                    occupied_diag2.remove(diag2)
                    
                    if found and not find_all:
                        return True
            
            return False
        
        # Start backtracking
        backtrack(0, [])
        
        # If only finding first solution and we have solutions, keep only first
        if not find_all and self.solutions:
            self.solutions = [self.solutions[0]]
        
        # Limit solutions if requested
        if max_solutions and len(self.solutions) > max_solutions:
            self.solutions = self.solutions[:max_solutions]
        
        self.execution_time = time.time() - start_time
        return self.solutions
    
    def count_solutions(self) -> int:
        """Count all solutions without storing them"""
        self.solve(count_only=True)
        return self.solution_count
    
    def get_first_solution(self) -> Optional[List[int]]:
        """Get just the first solution quickly"""
        solutions = self.solve(find_all=False, max_solutions=1)
        return solutions[0] if solutions else None


class GeneticSolver(NQueensSolver):
    """Genetic Algorithm solver for N-Queens"""
    
    def __init__(self, n: int, config: Optional[AlgorithmConfig] = None):
        super().__init__(n)
        self.config = config or AlgorithmConfig()
        self.fitness_history = []
        self.best_fitness_history = []
        
    def _fitness(self, board: List[int]) -> int:
        """
        Calculate fitness (higher is better)
        Fitness = -number of conflicts
        """
        conflicts = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                    conflicts += 1
        return -conflicts  # Negative because we want to minimize conflicts
    
    def _create_individual(self) -> List[int]:
        """Create random individual (each queen in different row by default)"""
        # Create a random permutation to ensure all queens in different rows
        individual = list(range(self.n))
        random.shuffle(individual)
        return individual
    
    def _mutate(self, individual: List[int]) -> List[int]:
        """Mutate an individual with swap mutation"""
        if random.random() < self.config.mutation_rate:
            # Swap two random positions
            i, j = random.sample(range(self.n), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order Crossover (OX) for permutation problems"""
        if random.random() > self.config.crossover_rate:
            return parent1.copy() if random.random() > 0.5 else parent2.copy()
        
        # Choose two random cut points
        size = self.n
        a, b = sorted(random.sample(range(size), 2))
        
        # Initialize child with None values
        child = [None] * size
        
        # Copy segment from parent1
        child[a:b] = parent1[a:b]
        
        # Fill remaining positions with values from parent2
        ptr = b
        for value in parent2[b:] + parent2[:b]:
            if value not in child[a:b]:
                if ptr >= size:
                    ptr = 0
                while child[ptr] is not None:
                    ptr += 1
                    if ptr >= size:
                        ptr = 0
                child[ptr] = value
                ptr += 1
        
        return child
    
    def _tournament_selection(self, population: List[List[int]], 
                             fitnesses: List[int]) -> List[int]:
        """Select parents using tournament selection"""
        tournament_indices = random.sample(range(len(population)), 
                                          self.config.tournament_size)
        tournament_fitness = [(i, fitnesses[i]) for i in tournament_indices]
        tournament_fitness.sort(key=lambda x: x[1], reverse=True)
        return population[tournament_fitness[0][0]]
    
    def solve(self, max_attempts: int = 3) -> List[List[int]]:
        """
        Solve using Genetic Algorithm
        
        Args:
            max_attempts: Maximum number of attempts to find perfect solution
            
        Returns:
            List of solutions found
        """
        start_time = time.time()
        self.solutions = []
        self.fitness_history = []
        self.best_fitness_history = []
        
        for attempt in range(max_attempts):
            if self.config.verbose:
                print(f"\nAttempt {attempt + 1}/{max_attempts}")
            
            # Initialize population
            population = [self._create_individual() 
                         for _ in range(self.config.population_size)]
            
            best_solution = None
            best_fitness = float('-inf')
            
            for generation in range(self.config.max_generations):
                # Calculate fitness
                fitnesses = [self._fitness(ind) for ind in population]
                
                # Track best solution
                current_best_fitness = max(fitnesses)
                if current_best_fitness > best_fitness:
                    best_fitness = current_best_fitness
                    best_idx = fitnesses.index(current_best_fitness)
                    best_solution = population[best_idx]
                
                # Store history
                self.fitness_history.append(np.mean(fitnesses))
                self.best_fitness_history.append(best_fitness)
                
                # Check for perfect solution
                if best_fitness == 0:
                    if self.config.verbose:
                        print(f"✅ Perfect solution found at generation {generation}")
                    
                    if self.validate_solution(best_solution):
                        self.solutions = [best_solution]
                        self.solution_count = 1
                        self.execution_time = time.time() - start_time
                        return self.solutions
                
                # Create next generation
                new_population = []
                
                # Elitism: keep best individuals
                elite_indices = np.argsort(fitnesses)[-self.config.elite_size:]
                for idx in elite_indices:
                    new_population.append(population[idx])
                
                # Fill rest with crossover and mutation
                while len(new_population) < self.config.population_size:
                    parent1 = self._tournament_selection(population, fitnesses)
                    parent2 = self._tournament_selection(population, fitnesses)
                    child = self._crossover(parent1, parent2)
                    child = self._mutate(child)
                    new_population.append(child)
                
                population = new_population
                
                # Progress indicator
                if self.config.verbose and generation % 100 == 0:
                    print(f"  Generation {generation}: Best fitness = {best_fitness}")
            
            # If we found a solution in this attempt, return it
            if best_fitness == 0 and self.validate_solution(best_solution):
                self.solutions = [best_solution]
                self.solution_count = 1
                self.execution_time = time.time() - start_time
                return self.solutions
        
        # If no perfect solution found, return best attempt
        if best_solution:
            self.solutions = [best_solution]
            if self.config.verbose:
                print(f"\n⚠️  Best solution has {abs(best_fitness)} conflicts")
        
        self.execution_time = time.time() - start_time
        return self.solutions
    
    def plot_fitness_progress(self):
        """Plot fitness progression over generations"""
        if not self.fitness_history:
            print("No fitness history available. Run solve() first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot average fitness
        ax1.plot(self.fitness_history, 'b-', alpha=0.7, label='Average Fitness')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Average Fitness Over Generations')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot best fitness
        ax2.plot(self.best_fitness_history, 'r-', linewidth=2, label='Best Fitness')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness')
        ax2.set_title('Best Fitness Over Generations')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()


class MinConflictsSolver(NQueensSolver):
    """Min-Conflicts Heuristic solver"""
    
    def __init__(self, n: int, max_steps: int = 10000, max_attempts: int = 10):
        super().__init__(n)
        self.max_steps = max_steps
        self.max_attempts = max_attempts
        self.attempts_made = 0
    
    def _conflicts(self, board: List[int], col: int, row: int) -> int:
        """Count conflicts for a specific position"""
        count = 0
        for other_col in range(self.n):
            if other_col == col:
                continue
            other_row = board[other_col]
            if (other_row == row or 
                abs(other_row - row) == abs(other_col - col)):
                count += 1
        return count
    
    def _total_conflicts(self, board: List[int]) -> int:
        """Calculate total number of conflicts"""
        conflicts = 0
        for i in range(self.n):
            conflicts += self._conflicts(board, i, board[i])
        return conflicts // 2  # Each conflict counted twice
    
    def _find_conflicted_queens(self, board: List[int]) -> List[int]:
        """Find all queens that are in conflict"""
        conflicted = []
        for col in range(self.n):
            if self._conflicts(board, col, board[col]) > 0:
                conflicted.append(col)
        return conflicted
    
    def solve(self) -> List[List[int]]:
        """Solve using Min-Conflicts heuristic with random restarts"""
        start_time = time.time()
        self.solutions = []
        self.attempts_made = 0
        
        for attempt in range(self.max_attempts):
            self.attempts_made += 1
            
            # Initialize with random placement (all queens in different rows)
            board = list(range(self.n))
            random.shuffle(board)
            
            for step in range(self.max_steps):
                # Find conflicted queens
                conflicted = self._find_conflicted_queens(board)
                
                if not conflicted:  # Solution found!
                    if self.validate_solution(board):
                        self.solutions = [board]
                        self.solution_count = 1
                        self.execution_time = time.time() - start_time
                        print(f"✅ Solution found in {self.attempts_made} attempts, {step} steps")
                        return self.solutions
                
                # Choose random conflicted queen
                col = random.choice(conflicted)
                current_row = board[col]
                
                # Find position with minimum conflicts
                min_conflicts = self.n + 1
                best_rows = []
                
                for row in range(self.n):
                    if row == current_row:
                        continue
                    conflicts = self._conflicts(board, col, row)
                    if conflicts < min_conflicts:
                        min_conflicts = conflicts
                        best_rows = [row]
                    elif conflicts == min_conflicts:
                        best_rows.append(row)
                
                # Move to best position (random choice if tie)
                board[col] = random.choice(best_rows)
            
            print(f"Attempt {attempt + 1}: No solution found in {self.max_steps} steps")
        
        self.execution_time = time.time() - start_time
        print(f"⚠️  Failed to find solution after {self.max_attempts} attempts")
        return []


class Benchmark:
    """Benchmark different algorithms"""
    
    @staticmethod
    def compare_algorithms(n_values: List[int] = [4, 8, 12], runs: int = 5):
        """
        Compare performance of different algorithms
        
        Args:
            n_values: List of board sizes to test
            runs: Number of runs per algorithm per board size
        """
        results = {}
        
        for n in n_values:
            print(f"\n{'='*60}")
            print(f"Testing N = {n}")
            print(f"{'='*60}")
            
            n_results = {
                'Backtracking': {'times': [], 'success': []},
                'Genetic': {'times': [], 'success': []},
                'Min-Conflicts': {'times': [], 'success': []}
            }
            
            # Test Backtracking
            print("\nBacktracking Algorithm:")
            for i in range(runs):
                solver = BacktrackingSolver(n)
                start = time.time()
                solutions = solver.solve(find_all=False, max_solutions=1)
                elapsed = time.time() - start
                n_results['Backtracking']['times'].append(elapsed)
                n_results['Backtracking']['success'].append(len(solutions) > 0)
                print(f"  Run {i+1}: {elapsed:.4f}s, {'Success' if solutions else 'Failed'}")
            
            # Test Genetic Algorithm
            print("\nGenetic Algorithm:")
            config = AlgorithmConfig(verbose=False, max_generations=500)
            for i in range(runs):
                solver = GeneticSolver(n, config)
                start = time.time()
                solutions = solver.solve(max_attempts=1)
                elapsed = time.time() - start
                n_results['Genetic']['times'].append(elapsed)
                n_results['Genetic']['success'].append(len(solutions) > 0)
                print(f"  Run {i+1}: {elapsed:.4f}s, {'Success' if solutions else 'Failed'}")
            
            # Test Min-Conflicts
            print("\nMin-Conflicts Algorithm:")
            for i in range(runs):
                solver = MinConflictsSolver(n, max_steps=1000, max_attempts=3)
                start = time.time()
                solutions = solver.solve()
                elapsed = time.time() - start
                n_results['Min-Conflicts']['times'].append(elapsed)
                n_results['Min-Conflicts']['success'].append(len(solutions) > 0)
                print(f"  Run {i+1}: {elapsed:.4f}s, {'Success' if solutions else 'Failed'}")
            
            results[n] = n_results
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"{'BENCHMARK SUMMARY':^80}")
        print(f"{'='*80}")
        
        print(f"\n{'Board Size':<12} {'Algorithm':<15} {'Avg Time (s)':<15} {'Success Rate':<15}")
        print(f"{'-'*60}")
        
        for n, n_results in results.items():
            for algo, data in n_results.items():
                avg_time = np.mean(data['times'])
                success_rate = np.mean(data['success']) * 100
                print(f"{n:<12} {algo:<15} {avg_time:<15.4f} {success_rate:<15.1f}%")
        
        # Plot results
        Benchmark._plot_results(results)
    
    @staticmethod
    def _plot_results(results: Dict):
        """Plot benchmark results"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Prepare data
        n_values = sorted(results.keys())
        algorithms = ['Backtracking', 'Genetic', 'Min-Conflicts']
        colors = ['blue', 'green', 'orange']
        
        # Plot 1: Execution times
        ax1 = axes[0]
        x = np.arange(len(n_values))
        width = 0.25
        
        for idx, algo in enumerate(algorithms):
            times = [np.mean(results[n][algo]['times']) for n in n_values]
            ax1.bar(x + idx*width - width, times, width, label=algo, color=colors[idx])
        
        ax1.set_xlabel('Board Size (N)')
        ax1.set_ylabel('Average Time (seconds)')
        ax1.set_title('Algorithm Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(n_values)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale for better visualization
        
        # Plot 2: Success rates
        ax2 = axes[1]
        for idx, algo in enumerate(algorithms):
            rates = [np.mean(results[n][algo]['success']) * 100 for n in n_values]
            ax2.plot(n_values, rates, 'o-', label=algo, color=colors[idx], linewidth=2)
        
        ax2.set_xlabel('Board Size (N)')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Algorithm Success Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 105)
        
        plt.tight_layout()
        plt.show()


def interactive_demo():
    """Interactive demo with menu"""
    print("="*60)
    print("N-QUEENS PROBLEM SOLVER - INTERACTIVE DEMO")
    print("="*60)
    
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Solve N-Queens with specific algorithm")
        print("2. Compare all algorithms (Benchmark)")
        print("3. Count all solutions for N")
        print("4. Visualize known solutions")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            n = int(input("Enter board size (N): "))
            
            print("\nSelect algorithm:")
            print("1. Backtracking (guaranteed, finds all solutions)")
            print("2. Genetic Algorithm (probabilistic, good for large N)")
            print("3. Min-Conflicts (fast heuristic, good for large N)")
            
            algo_choice = input("Your choice (1-3): ").strip()
            
            if algo_choice == '1':
                print("\nBacktracking Options:")
                print("1. Find first solution only (fast)")
                print("2. Find all solutions (may be slow for N>12)")
                option = input("Your choice (1-2): ").strip()
                
                solver = BacktrackingSolver(n)
                if option == '1':
                    print(f"\nFinding first solution for N={n}...")
                    solutions = solver.solve(find_all=False, max_solutions=1)
                else:
                    print(f"\nFinding all solutions for N={n}...")
                    solutions = solver.solve()
                
                print(f"\nResults:")
                print(f"  Solutions found: {len(solutions)}")
                print(f"  Time taken: {solver.execution_time:.4f}s")
                
                if solutions:
                    show = input("\nVisualize first solution? (y/n): ").strip().lower()
                    if show == 'y':
                        solver.visualize_solution(solutions[0], 
                                                 f"N={n}, Solution 1/{len(solutions)}")
            
            elif algo_choice == '2':
                print("\nGenetic Algorithm Configuration:")
                pop_size = int(input("Population size (default 100): ") or 100)
                max_gen = int(input("Max generations (default 500): ") or 500)
                
                config = AlgorithmConfig(
                    population_size=pop_size,
                    max_generations=max_gen,
                    verbose=True
                )
                
                solver = GeneticSolver(n, config)
                print(f"\nSolving N={n} with Genetic Algorithm...")
                solutions = solver.solve()
                
                if solutions:
                    print(f"\n✅ Solution found in {solver.execution_time:.4f}s")
                    solver.visualize_solution(solutions[0], "Genetic Algorithm Solution")
                    
                    # Show fitness progress
                    show_progress = input("\nShow fitness progression? (y/n): ").strip().lower()
                    if show_progress == 'y':
                        solver.plot_fitness_progress()
                else:
                    print(f"\n⚠️  No perfect solution found. Best attempt had conflicts.")
            
            elif algo_choice == '3':
                print("\nMin-Conflicts Configuration:")
                max_steps = int(input("Max steps per attempt (default 1000): ") or 1000)
                max_attempts = int(input("Max attempts (default 10): ") or 10)
                
                solver = MinConflictsSolver(n, max_steps=max_steps, max_attempts=max_attempts)
                print(f"\nSolving N={n} with Min-Conflicts...")
                solutions = solver.solve()
                
                if solutions:
                    print(f"\n✅ Solution found!")
                    solver.visualize_solution(solutions[0], "Min-Conflicts Solution")
                else:
                    print(f"\n⚠️  No solution found after {solver.attempts_made} attempts")
        
        elif choice == '2':
            print("\nBenchmark Configuration:")
            n_input = input("Enter board sizes (comma separated, e.g., 4,8,12): ").strip()
            n_values = [int(x.strip()) for x in n_input.split(',')]
            runs = int(input("Number of runs per test (default 5): ") or 5)
            
            Benchmark.compare_algorithms(n_values, runs)
        
        elif choice == '3':
            n = int(input("Enter board size (N, max recommended 15): "))
            solver = BacktrackingSolver(n)
            print(f"\nCounting all solutions for N={n}...")
            count = solver.count_solutions()
            print(f"\nTotal solutions for {n}-Queens: {count}")
            
            # Show known sequence
            known_counts = {
                1: 1, 2: 0, 3: 0, 4: 2, 5: 10, 6: 4, 7: 40, 8: 92,
                9: 352, 10: 724, 11: 2680, 12: 14200, 13: 73712,
                14: 365596, 15: 2279184
            }
            
            if n in known_counts:
                expected = known_counts[n]
                if count == expected:
                    print(f"✓ Count matches known sequence ({expected})")
                else:
                    print(f"⚠️  Count differs from known sequence (expected: {expected})")
        
        elif choice == '4':
            print("\nVisualize classic solutions:")
            print("1. 4-Queens (2 solutions)")
            print("2. 8-Queens (92 solutions)")
            print("3. Custom solution")
            
            viz_choice = input("Your choice (1-3): ").strip()
            
            if viz_choice == '1':
                # 4-Queens solutions
                solutions_4 = [
                    [1, 3, 0, 2],  # Solution 1
                    [2, 0, 3, 1]   # Solution 2
                ]
                solver = BacktrackingSolver(4)
                for i, sol in enumerate(solutions_4):
                    solver.visualize_solution(sol, f"4-Queens Solution {i+1}")
            
            elif viz_choice == '2':
                # Classic 8-Queens solution
                classic_8 = [0, 4, 7, 5, 2, 6, 1, 3]
                solver = BacktrackingSolver(8)
                solver.visualize_solution(classic_8, "Classic 8-Queens Solution")
            
            elif viz_choice == '3':
                n = int(input("Enter board size: "))
                print(f"\nEnter solution as {n} numbers (0-indexed rows), comma separated")
                print("Example for 4-Queens: 1,3,0,2")
                sol_input = input("Solution: ").strip()
                solution = [int(x.strip()) for x in sol_input.split(',')]
                
                if len(solution) != n:
                    print(f"Error: Solution must have exactly {n} numbers")
                else:
                    solver = BacktrackingSolver(n)
                    if solver.validate_solution(solution):
                        solver.visualize_solution(solution, "Custom Solution")
                    else:
                        print("Invalid solution! Queens conflict.")
        
        elif choice == '5':
            print("\nThank you for using N-Queens Solver!")
            print("="*60)
            break
        
        else:
            print("Invalid choice. Please try again.")


def main():
    """Main entry point"""
    print("="*60)
    print("N-QUEENS PROBLEM SOLVER")
    print("="*60)
    print("\nChoose mode:")
    print("1. Interactive Demo (recommended)")
    print("2. Quick Solve (command line)")
    print("3. Run Benchmark")
    
    mode = input("\nEnter mode (1-3): ").strip()
    
    if mode == '1':
        interactive_demo()
    elif mode == '2':
        # Quick command line mode
        import sys
        if len(sys.argv) > 1:
            n = int(sys.argv[1])
            algo = sys.argv[2] if len(sys.argv) > 2 else 'backtracking'
            
            if algo.lower() in ['bt', 'backtracking']:
                solver = BacktrackingSolver(n)
                solutions = solver.solve(find_all=False, max_solutions=1)
            elif algo.lower() in ['ga', 'genetic']:
                solver = GeneticSolver(n, AlgorithmConfig(verbose=False))
                solutions = solver.solve()
            elif algo.lower() in ['mc', 'minconflicts']:
                solver = MinConflictsSolver(n)
                solutions = solver.solve()
            else:
                print(f"Unknown algorithm: {algo}")
                return
            
            if solutions:
                print(f"Solution found for N={n}: {solutions[0]}")
                solver.visualize_solution(solutions[0])
            else:
                print(f"No solution found for N={n}")
        else:
            print("Usage: python nqueens.py <N> [algorithm]")
            print("Algorithms: backtracking (bt), genetic (ga), minconflicts (mc)")
    elif mode == '3':
        Benchmark.compare_algorithms([4, 8, 12])
    else:
        print("Invalid mode selected.")


if __name__ == "__main__":
    main()
