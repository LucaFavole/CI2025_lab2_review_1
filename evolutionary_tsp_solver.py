"""
Evolutionary Algorithm (Genetic Algorithm) for TSP - handles both symmetric and asymmetric problems.

Key features:
1. Order-based crossover (OX) - preserves relative order
2. Swap and inversion mutations
3. Tournament selection
4. Elitism (keep best individuals)
5. Adaptive mutation rate
6. 2-opt local search for refinement
"""

import numpy as np
import random
import time
from typing import Tuple, List, Dict


# ============================================================================
# TSP utilities
# ============================================================================

def tour_cost(D: np.ndarray, tour: List[int]) -> float:
    """Calculate tour cost."""
    if not tour:
        return 0.0
    n = len(tour)
    cost = 0.0
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        cost += D[a, b]
    return float(cost)


def nearest_neighbor(D: np.ndarray, start: int = 0) -> List[int]:
    """Greedy nearest neighbor initialization."""
    n = D.shape[0]
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    
    while unvisited:
        current = tour[-1]
        nearest = min(unvisited, key=lambda x: D[current, x])
        tour.append(nearest)
        unvisited.remove(nearest)
    
    return tour


# ============================================================================
# Genetic operators
# ============================================================================

def order_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    """
    Order Crossover (OX) - preserves relative order from parents.
    
    Example:
    P1: [1,2,3,4,5,6,7,8]
    P2: [3,7,5,1,6,8,2,4]
    Select segment from P1: [3,4,5]
    Fill remaining from P2 in order: [7,1,3,4,5,6,8,2]
    """
    n = len(parent1)
    
    # Select random segment from parent1
    start = random.randint(0, n - 2)
    end = random.randint(start + 1, n)
    
    # Copy segment from parent1
    child = [-1] * n
    child[start:end] = parent1[start:end]
    
    # Fill remaining positions from parent2 (preserving order)
    p2_idx = 0
    for i in range(n):
        if child[i] == -1:
            # Find next city from parent2 not already in child
            while parent2[p2_idx] in child:
                p2_idx += 1
            child[i] = parent2[p2_idx]
            p2_idx += 1
    
    return child


def pmx_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    """
    Partially Mapped Crossover (PMX) - good for TSP.
    """
    n = len(parent1)
    
    # Select two random points
    point1 = random.randint(0, n - 2)
    point2 = random.randint(point1 + 1, n)
    
    # Initialize child with -1
    child = [-1] * n
    
    # Copy segment from parent1
    child[point1:point2] = parent1[point1:point2]
    
    # Create mapping from parent1 segment to parent2 segment
    for i in range(point1, point2):
        if parent2[i] not in child:
            # Find position for parent2[i]
            pos = i
            while point1 <= pos < point2:
                # Find where parent1[pos] is in parent2
                pos = parent2.index(parent1[pos])
            child[pos] = parent2[i]
    
    # Fill remaining with parent2
    for i in range(n):
        if child[i] == -1:
            child[i] = parent2[i]
    
    return child


def swap_mutation(tour: List[int], rate: float = 0.01) -> List[int]:
    """Swap two random cities with given probability."""
    tour = tour[:]
    n = len(tour)
    
    if random.random() < rate:
        i, j = random.sample(range(n), 2)
        tour[i], tour[j] = tour[j], tour[i]
    
    return tour


def inversion_mutation(tour: List[int], rate: float = 0.01) -> List[int]:
    """Reverse a random segment with given probability."""
    tour = tour[:]
    n = len(tour)
    
    if random.random() < rate:
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n)
        tour[i:j] = tour[i:j][::-1]
    
    return tour


def scramble_mutation(tour: List[int], rate: float = 0.01) -> List[int]:
    """Randomly shuffle a segment with given probability."""
    tour = tour[:]
    n = len(tour)
    
    if random.random() < rate:
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, min(n, i + 5))  # Small segments
        segment = tour[i:j]
        random.shuffle(segment)
        tour[i:j] = segment
    
    return tour


# ============================================================================
# Selection
# ============================================================================

def tournament_selection(population: List[List[int]], 
                        fitness: List[float], 
                        tournament_size: int = 3) -> List[int]:
    """Select individual using tournament selection."""
    # Select random individuals for tournament
    tournament_idx = random.sample(range(len(population)), tournament_size)
    
    # Return best individual from tournament (minimize cost)
    best_idx = min(tournament_idx, key=lambda i: fitness[i])
    return population[best_idx][:]


def roulette_selection(population: List[List[int]], 
                       fitness: List[float]) -> List[int]:
    """Roulette wheel selection (for minimization, invert fitness)."""
    # Convert to maximization (invert costs)
    max_fitness = max(fitness)
    inverted_fitness = [max_fitness - f + 1 for f in fitness]
    total = sum(inverted_fitness)
    
    # Roulette wheel
    pick = random.uniform(0, total)
    current = 0
    for i, f in enumerate(inverted_fitness):
        current += f
        if current > pick:
            return population[i][:]
    
    return population[-1][:]


# ============================================================================
# 2-opt local search (optional refinement)
# ============================================================================

def two_opt_simple(D: np.ndarray, tour: List[int], max_iterations: int = 100) -> Tuple[List[int], float]:
    """Simple 2-opt for local refinement - handles asymmetric matrices correctly."""
    n = len(tour)
    best = tour[:]
    best_cost = tour_cost(D, best)
    
    # Check if matrix is symmetric
    is_symmetric = np.allclose(D, D.T, rtol=1e-5, atol=1e-8)
    
    improved = True
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        for i in range(n - 1):
            for j in range(i + 2, n):
                if is_symmetric:
                    # Fast symmetric case
                    a, b = best[i], best[i + 1]
                    c, d = best[j], best[(j + 1) % n]
                    
                    delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                    
                    if delta < -1e-9:
                        # Reverse segment
                        best[i + 1:j + 1] = best[i + 1:j + 1][::-1]
                        best_cost += delta
                        improved = True
                        break
                else:
                    # Asymmetric case: calculate full cost difference
                    # Old edges
                    old_cost = D[best[i], best[i + 1]]
                    if j + 1 < n:
                        old_cost += D[best[j], best[j + 1]]
                    else:
                        old_cost += D[best[j], best[0]]
                    
                    # Calculate cost after reversing segment [i+1, j]
                    segment = best[i + 1:j + 1]
                    reversed_segment = segment[::-1]
                    
                    # New edges after reversal
                    new_cost = D[best[i], reversed_segment[0]]
                    if j + 1 < n:
                        new_cost += D[reversed_segment[-1], best[j + 1]]
                    else:
                        new_cost += D[reversed_segment[-1], best[0]]
                    
                    # Cost of reversed segment itself
                    for k in range(len(reversed_segment) - 1):
                        old_cost += D[segment[k], segment[k + 1]]
                        new_cost += D[reversed_segment[k], reversed_segment[k + 1]]
                    
                    delta = new_cost - old_cost
                    
                    if delta < -1e-9:
                        # Apply reversal
                        best[i + 1:j + 1] = reversed_segment
                        best_cost += delta
                        improved = True
                        break
            
            if improved:
                break
    
    return best, best_cost


# ============================================================================
# Main Evolutionary Algorithm
# ============================================================================

def evolutionary_tsp_solve(
    D: np.ndarray,
    population_size: int = None,  # Auto-adjust based on problem size
    generations: int = None,      # Auto-adjust based on problem size
    elite_size: int = 10,
    mutation_rate: float = 0.15,
    tournament_size: int = 5,
    crossover_type: str = 'ox',  # 'ox' or 'pmx'
    local_search: bool = True,
    local_search_freq: int = 10,  # Apply 2-opt every N generations
    adaptive_mutation: bool = True,
    verbose: bool = True
) -> Tuple[List[int], Dict[str, float]]:
    """
    Solve TSP using Evolutionary Algorithm (Genetic Algorithm).
    
    Args:
        D: distance matrix (n x n)
        population_size: number of individuals in population
        generations: number of generations to evolve
        elite_size: number of best individuals to keep unchanged
        mutation_rate: probability of mutation (per individual)
        tournament_size: size of tournament for selection
        crossover_type: 'ox' (Order Crossover) or 'pmx' (PMX)
        local_search: apply 2-opt refinement periodically
        local_search_freq: apply 2-opt every N generations
        adaptive_mutation: increase mutation if no improvement
        verbose: print progress
    
    Returns:
        (best_tour, stats)
    """
    n = D.shape[0]
    
    # OPTIMISATION: Auto-adjust parameters based on problem size
    if population_size is None:
        if n <= 50:
            population_size = 100
        elif n <= 100:
            population_size = 80
        elif n <= 200:
            population_size = 60
        elif n <= 500:
            population_size = 50
        else:  # n > 500
            population_size = 40  # Smaller population for large instances
    
    if generations is None:
        if n <= 50:
            generations = 500
        elif n <= 100:
            generations = 400
        elif n <= 200:
            generations = 300
        elif n <= 500:
            generations = 200  # Fewer generations for 500 cities
        else:  # n > 500
            generations = 150  # Much fewer for 1000 cities
    
    if verbose:
        print(f"{'='*60}")
        print(f"Evolutionary TSP Solver (n={n})")
        print(f"Population: {population_size}, Generations: {generations}")
        print(f"Elite: {elite_size}, Mutation: {mutation_rate:.2f}")
        print(f"Crossover: {crossover_type.upper()}, Local search: {local_search}")
        print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Initialize population
    if verbose:
        print("[Initialization] Creating initial population...")
    
    population = []
    
    # First individual: nearest neighbor
    population.append(nearest_neighbor(D))
    
    # Rest: random permutations with some nearest neighbor variants
    base_tour = list(range(n))
    for i in range(population_size - 1):
        if i < min(10, population_size // 10):
            # Some nearest neighbor from different starts
            start_city = random.randint(0, n - 1)
            population.append(nearest_neighbor(D, start_city))
        else:
            # Random permutations
            tour = base_tour[:]
            random.shuffle(tour)
            population.append(tour)
    
    # Track best solution
    fitness = [tour_cost(D, tour) for tour in population]
    best_idx = np.argmin(fitness)
    best_tour = population[best_idx][:]
    best_cost = fitness[best_idx]
    
    if verbose:
        print(f"  Initial best cost: {best_cost:.2f}\n")
    
    # Evolution statistics
    stagnation_counter = 0
    current_mutation_rate = mutation_rate
    best_costs_history = [best_cost]
    
    # Main evolution loop
    for gen in range(generations):
        # Sort population by fitness (best first)
        sorted_indices = np.argsort(fitness)
        population = [population[i] for i in sorted_indices]
        fitness = [fitness[i] for i in sorted_indices]
        
        # Check for improvement
        if fitness[0] < best_cost:
            best_cost = fitness[0]
            best_tour = population[0][:]
            stagnation_counter = 0
            if verbose and gen % 50 == 0:
                print(f"Gen {gen:4d}: New best = {best_cost:.2f}")
        else:
            stagnation_counter += 1
        
        best_costs_history.append(best_cost)
        
        # Adaptive mutation rate
        if adaptive_mutation and stagnation_counter > 20:
            current_mutation_rate = min(0.5, mutation_rate * 1.5)
        else:
            current_mutation_rate = mutation_rate
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individuals
        new_population.extend(population[:elite_size])
        
        # Generate offspring
        while len(new_population) < population_size:
            # Selection
            parent1 = tournament_selection(population, fitness, tournament_size)
            parent2 = tournament_selection(population, fitness, tournament_size)
            
            # Crossover
            if crossover_type == 'pmx':
                child = pmx_crossover(parent1, parent2)
            else:  # 'ox'
                child = order_crossover(parent1, parent2)
            
            # Mutation (apply multiple types)
            child = inversion_mutation(child, current_mutation_rate)
            child = swap_mutation(child, current_mutation_rate / 2)
            
            new_population.append(child)
        
        # Update population
        population = new_population[:population_size]
        
        # Recalculate fitness
        fitness = [tour_cost(D, tour) for tour in population]
        
        # Local search (2-opt) on best individuals periodically
        if local_search and (gen + 1) % local_search_freq == 0:
            for i in range(min(elite_size, 5)):
                improved_tour, improved_cost = two_opt_simple(D, population[i], max_iterations=50)
                population[i] = improved_tour
                fitness[i] = improved_cost
        
        # Progress
        if verbose and (gen + 1) % 100 == 0:
            avg_cost = np.mean(fitness)
            print(f"Gen {gen+1:4d}: Best={best_cost:.2f}, Avg={avg_cost:.2f}, "
                  f"MutRate={current_mutation_rate:.3f}, Stagnation={stagnation_counter}")
    
    # Final 2-opt refinement on best solution
    if local_search:
        if verbose:
            print(f"\n[Final Refinement] Applying intensive 2-opt...")
        best_tour, best_cost = two_opt_simple(D, best_tour, max_iterations=500)
    
    elapsed = time.time() - start_time
    
    # Statistics
    stats = {
        'final': best_cost,
        'initial': best_costs_history[0],
        'improvement': (best_costs_history[0] - best_cost) / best_costs_history[0] * 100,
        'time': elapsed,
        'generations': generations,
        'population_size': population_size,
        'best_history': best_costs_history
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Initial cost: {stats['initial']:.2f}")
        print(f"Final cost: {stats['final']:.2f}")
        print(f"Improvement: {stats['improvement']:.2f}%")
        print(f"Time: {elapsed:.2f}s")
        print(f"{'='*60}\n")
    
    return best_tour, stats


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    # Test on small problems
    test_files = [
        "lab2/problem_r1_10.npy",
        "lab2/problem_r2_10.npy",
        #"lab2/problem_g_200.npy",
    ]
    
    for filepath in test_files:
        try:
            print(f"\n{'#'*60}")
            print(f"Testing: {filepath}")
            print(f"{'#'*60}\n")
            
            D = np.load(filepath)
            
            tour, stats = evolutionary_tsp_solve(
                D,
                population_size=None,
                generations=None,
                elite_size=10,
                mutation_rate=0.15,
                tournament_size=5,
                crossover_type='ox',
                local_search=True,
                local_search_freq=20,
                adaptive_mutation=True,
                verbose=True
            )
            
        except FileNotFoundError:
            print(f"Skipping {filepath} (not found)")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
# 12262.58