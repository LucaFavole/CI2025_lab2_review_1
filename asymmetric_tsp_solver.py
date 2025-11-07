"""
Asymmetric TSP solver - handles non-symmetric distance matrices and negative values.

Key differences from symmetric solver:
1. No MDS (requires symmetry) → uses PCA or hierarchical clustering
2. Directional 2-opt (respects asymmetry)
3. Proper handling of negative edge weights
4. Alternative clustering methods
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from typing import Tuple, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


# ============================================================================
# TSP utilities for asymmetric problems
# ============================================================================

def tour_cost(D: np.ndarray, perm: List[int]) -> float:
    """Calculate tour cost (works with asymmetric D)."""
    if not perm:
        return 0.0
    n = len(perm)
    cost = 0.0
    for i in range(n):
        a = perm[i]
        b = perm[(i + 1) % n]
        cost += D[a, b]
    return float(cost)


def nearest_neighbor_start(D: np.ndarray, start: int = 0) -> List[int]:
    """Greedy nearest neighbor (asymmetric-aware)."""
    n = D.shape[0]
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    
    while unvisited:
        current = tour[-1]
        # Choose minimum OUTGOING edge from current
        nearest = min(unvisited, key=lambda x: D[current, x])
        tour.append(nearest)
        unvisited.remove(nearest)
    
    return tour


def build_candidate_lists_asymmetric(D: np.ndarray, k: int = 20) -> List[np.ndarray]:
    """Build k-nearest neighbor lists (outgoing edges for asymmetric)."""
    n = D.shape[0]
    candidates = []
    
    for i in range(n):
        # Sort by OUTGOING distance from i
        distances = [(D[i, j], j) for j in range(n) if j != i]
        distances.sort()
        nearest = [j for _, j in distances[:min(k, len(distances))]]
        candidates.append(np.array(nearest, dtype=np.int32))
    
    return candidates


def asymmetric_2opt(D: np.ndarray, perm: List[int], 
                    candidates: List[np.ndarray] = None,
                    max_no_improve: int = 200,
                    max_iterations: int = None) -> Tuple[List[int], float]:
    """
    Asymmetric 2-opt: only considers valid directional swaps.
    
    For asymmetric TSP, we need to be careful:
    - Edge reversal changes directions
    - Must check both D[a,b] and D[b,a]
    """
    n = len(perm)
    
    if max_iterations is None:
        max_iterations = n * 100
    
    if candidates is None:
        k = min(20, max(5, n // 20))
        candidates = build_candidate_lists_asymmetric(D, k=k)
    
    # OPTIMISATION: Précalculer la version symétrisée pour filtrage rapide
    D_sym = (D + D.T) / 2
    
    # Position map
    pos = np.empty(n, dtype=int)
    for i, v in enumerate(perm):
        pos[v] = i
    
    best = perm[:]
    best_cost = tour_cost(D, best)
    no_improve = 0
    iterations = 0
    
    # Progress tracking for large instances
    last_progress_time = time.time()
    progress_interval = 10  # Print every 10 seconds for large instances
    
    while no_improve < max_no_improve and iterations < max_iterations:
        improved = False
        
        # Print progress for large instances
        if n > 300 and (time.time() - last_progress_time) > progress_interval:
            print(f"    [2-opt] iter {iterations}/{max_iterations}, cost={best_cost:.2f}, no_improve={no_improve}/{max_no_improve}")
            last_progress_time = time.time()
        
        for i in range(n):
            if iterations >= max_iterations:
                break
            
            a = best[i]
            b = best[(i + 1) % n]
            
            # Try candidates from a (outgoing)
            for c in candidates[a]:
                j = pos[c]
                
                if j == i or j == (i + 1) % n or abs(j - i) == 1:
                    continue
                
                d = best[(j + 1) % n]
                
                # OPTIMISATION: Filtrage rapide avec D_sym (élimine 90% des swaps en O(1))
                delta_approx = D_sym[a, c] + D_sym[b, d] - D_sym[a, b] - D_sym[c, d]
                
                # Si l'approximation symétrique est mauvaise, skip immédiatement
                if delta_approx >= 0:
                    continue
                
                # Calcul incrémental du delta exact (seulement pour les ~10% prometteurs)
                # Extraire le segment qui sera inversé
                if i < j:
                    segment = best[i + 1 : j + 1]
                else:
                    segment = best[i + 1 :] + best[: j + 1]
                
                # Calcul incrémental du delta
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]

                if len(segment) > 1:
                    for k in range(len(segment) - 1):
                        old_edge = D[segment[k], segment[k + 1]]
                        new_edge = D[segment[-(k+1)], segment[-(k+2)]]
                        delta += new_edge - old_edge
                elif len(segment) == 1:
                    c_node = segment[0]
                    delta = D[a, c_node] + D[c_node, d] - D[a, b] - D[c, d]
                
                if delta < -1e-9:
                    # Perform reversal
                    if i < j:
                        best[i + 1 : j + 1] = best[i + 1 : j + 1][::-1]
                    else:
                        segment_full = best[i + 1 :] + best[: j + 1]
                        segment_reversed = segment_full[::-1]
                        len_tail = n - (i + 1)
                        best[i + 1 :] = segment_reversed[:len_tail]
                        best[: j + 1] = segment_reversed[len_tail:]
                    
                    # Update pos
                    for idx in range(n):
                        pos[best[idx]] = idx
                    
                    best_cost = tour_cost(D, best)  # Recalculate to avoid numerical drift
                    improvements = 1
                    improved = True
                    break
            
            if improved:
                break
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
        
        iterations += 1
    
    return best, best_cost


def perturb_perm(perm: List[int], k: int = 5) -> List[int]:
    """Perturb by reversing k random segments."""
    import random
    result = perm[:]
    n = len(result)
    
    for _ in range(k):
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        result[i:j+1] = result[i:j+1][::-1]
    
    return result


def multi_start_asymmetric_2opt(D: np.ndarray, restarts: int = 10,
                                 initial_perm: List[int] = None,
                                 verbose: bool = False,
                                 timeout: float = None) -> Tuple[List[int], float]:
    """Multi-start 2-opt for asymmetric TSP with timeout support."""
    n = D.shape[0]
    k_candidates = min(20, max(5, n // 20))
    candidates = build_candidate_lists_asymmetric(D, k=k_candidates)
    
    # Adjust iterations based on problem size
    if n <= 100:
        max_iter_per_restart = n * 50
        max_no_improve = 200
    elif n <= 200:
        max_iter_per_restart = n * 30
        max_no_improve = 150
    elif n <= 500:
        max_iter_per_restart = n * 10  # Much more aggressive for 500 cities
        max_no_improve = 50
    else:
        max_iter_per_restart = n * 5
        max_no_improve = 30
    
    max_iter_per_restart = min(max_iter_per_restart, 10000)  # Hard cap
    
    start_time = time.time() if timeout else None
    
    if initial_perm is None:
        best_perm = nearest_neighbor_start(D)
        best_perm, best_cost = asymmetric_2opt(D, best_perm, candidates, 
                                               max_iterations=max_iter_per_restart,
                                               max_no_improve=max_no_improve)
    else:
        best_perm = initial_perm[:]
        best_cost = tour_cost(D, best_perm)
    
    for r in range(restarts):
        # Check timeout
        if timeout and (time.time() - start_time) > timeout:
            if verbose:
                print(f"  Timeout reached after {r} restarts")
            break
        
        if r == 0 and initial_perm is not None:
            perm0 = initial_perm[:]
        elif r == 0:
            perm0 = nearest_neighbor_start(D)
        else:
            perm0 = perturb_perm(best_perm, k=max(3, n // 50))
        
        perm_improved, cost = asymmetric_2opt(D, perm0, candidates,
                                             max_iterations=max_iter_per_restart,
                                             max_no_improve=max_no_improve)
        
        if cost < best_cost:
            best_cost = cost
            best_perm = perm_improved[:]
            if verbose:
                print(f"  Restart {r+1}/{restarts}: improved to {best_cost:.2f}")
    
    return best_perm, best_cost


# ============================================================================
# Clustering for asymmetric matrices
# ============================================================================

def cluster_asymmetric_hierarchical(D: np.ndarray, n_clusters: int, 
                                   verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hierarchical clustering (your current method).
    Handles negative distances by translation.
    """
    if verbose:
        print(f"[Hierarchical] Clustering into {n_clusters} clusters (asymmetric-aware)...")
    
    # Average distance for clustering
    D_avg = (D + D.T) / 2
    
    # Translate to make all distances non-negative
    min_val = np.min(D_avg)
    if min_val < 0:
        D_shifted = D_avg - min_val + 1e-10
        if verbose:
            print(f"  Note: Translated distances by {-min_val:.2f}")
    else:
        D_shifted = D_avg.copy()
    
    np.fill_diagonal(D_shifted, 0)
    
    # Hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(D_shifted)
    
    # PCA for visualization
    D_features = np.column_stack([D, D.T])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(D_features)
    
    if verbose:
        for c in range(n_clusters):
            size = np.sum(labels == c)
            print(f"  Cluster {c}: {size} nodes")
        print()
    
    return labels, coords





# ============================================================================
# Auto-select clusters
# ============================================================================

def auto_select_clusters(D: np.ndarray, max_clusters: int = 20, 
                        verbose: bool = True) -> int:
    """
    Auto-select number of clusters using silhouette score.
    Handles negative distances by translation.
    """
    if verbose:
        print(f"[Auto-cluster] Finding optimal k (testing 2-{max_clusters})...")
    
    from sklearn.metrics import silhouette_score
    
    # Average distance for clustering
    D_avg = (D + D.T) / 2
    
    # Handle negative distances: shift to positive range
    min_val = np.min(D_avg)
    if min_val < 0:
        D_shifted = D_avg - min_val + 1e-10
        if verbose:
            print(f"  Note: Translated distances by {-min_val:.2f} for clustering")
    else:
        D_shifted = D_avg.copy()
    
    # Ensure zero diagonal
    np.fill_diagonal(D_shifted, 0)
    
    scores = []
    K_range = range(2, min(max_clusters + 1, len(D) // 2))
    
    for k in K_range:
        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(D_shifted)
        
        # Silhouette score on translated distances
        score = silhouette_score(D_shifted, labels, metric='precomputed')
        scores.append(score)
        
        if verbose:
            print(f"  k={k:2d}: Silhouette={score:.3f}")
    
    optimal_k = list(K_range)[np.argmax(scores)]
    best_score = np.max(scores)
    
    if verbose:
        print(f"✓ Optimal k: {optimal_k} (Silhouette={best_score:.3f})\n")
    
    return optimal_k


# ============================================================================
# Parallel solving
# ============================================================================

def solve_cluster_asymmetric(args: Tuple[int, np.ndarray, np.ndarray, int]) -> Tuple[int, List[int], float]:
    """
    Solve asymmetric TSP for a single cluster.
    
    OPTIMISATION: Pour petits clusters (≤15 villes), utilise Held-Karp (exact).
    Pour clusters moyens (≤20), convertit cycle en chaîne en coupant pire arête.
    """
    cluster_id, cluster_nodes, D, restarts = args
    
    if len(cluster_nodes) <= 1:
        return cluster_id, [int(x) for x in cluster_nodes], 0.0
    
    # Extract submatrix (preserves asymmetry)
    sub_D = D[np.ix_(cluster_nodes, cluster_nodes)]
    n = len(cluster_nodes)
    
    # OPTIMISATION 1: Résolution exacte pour petits clusters (≤15 villes)
    if n <= 15:
        try:
            from held_karp import held_karp
            perm_local, cost_local = held_karp(sub_D)
            # Held-Karp retourne un cycle, on le garde tel quel pour le merge
            global_tour = [int(cluster_nodes[i]) for i in perm_local]
            return cluster_id, global_tour, cost_local
        except Exception as e:
            # Fallback si held_karp échoue
            print(f"Warning: Held-Karp failed for cluster {cluster_id}, using 2-opt: {e}")
    
    # OPTIMISATION 2: Pour clusters moyens (16-20), résolution exacte possible mais plus lente
    if n <= 20:
        try:
            from held_karp import held_karp
            perm_local, cost_local = held_karp(sub_D)
            global_tour = [int(cluster_nodes[i]) for i in perm_local]
            return cluster_id, global_tour, cost_local
        except:
            pass  # Continue avec 2-opt
    
    # Fallback: Multi-start 2-opt pour grands clusters
    actual_restarts = min(restarts, max(3, n // 2))
    
    try:
        perm_local, cost_local = multi_start_asymmetric_2opt(
            sub_D, 
            restarts=actual_restarts,
            verbose=False
        )
    except Exception as e:
        print(f"Warning: Cluster {cluster_id} failed, using nearest neighbor: {e}")
        perm_local = nearest_neighbor_start(sub_D)
        cost_local = tour_cost(sub_D, perm_local)
    
    # Convert to global indices
    global_tour = [int(cluster_nodes[i]) for i in perm_local]
    
    return cluster_id, global_tour, cost_local


def solve_clusters_parallel_asymmetric(D: np.ndarray, labels: np.ndarray, n_clusters: int,
                                       restarts_per_cluster: int = 20, 
                                       verbose: bool = True) -> Dict[int, Tuple[List[int], float]]:
    """Solve asymmetric TSP for each cluster in parallel."""
    if verbose:
        print(f"[Parallel TSP] Solving {n_clusters} sub-problems (asymmetric)...")
    
    tasks = []
    for c in range(n_clusters):
        cluster_nodes = np.where(labels == c)[0]
        tasks.append((c, cluster_nodes, D, restarts_per_cluster))
    
    cluster_tours = {}
    start_time = time.time()
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(solve_cluster_asymmetric, task): task[0] for task in tasks}
        
        for future in as_completed(futures):
            cluster_id, tour, cost = future.result()
            cluster_tours[cluster_id] = (tour, cost)
            if verbose:
                print(f"  Cluster {cluster_id}: {len(tour)} nodes, cost={cost:.2f}")
    
    elapsed = time.time() - start_time
    if verbose:
        total_cost = sum(c for _, c in cluster_tours.values())
        print(f"✓ Parallel solving done in {elapsed:.2f}s, total intra-cluster cost: {total_cost:.2f}\n")
    
    return cluster_tours


# ============================================================================
# Merging (asymmetric-aware)
# ============================================================================

def merge_clusters_asymmetric(D: np.ndarray, cluster_tours: Dict[int, Tuple[List[int], float]],
                              verbose: bool = True) -> Tuple[List[int], float]:
    """Merge cluster tours (considers both orientations for asymmetric case)."""
    if verbose:
        print("[Merge] Stitching clusters (asymmetric-aware)...")
    
    sorted_clusters = sorted(cluster_tours.items(), key=lambda x: len(x[1][0]), reverse=True)
    merged_tour = list(sorted_clusters[0][1][0])
    remaining = {cid: tour for cid, (tour, _) in sorted_clusters[1:]}
    
    while remaining:
        best_cost_increase = float('inf')
        best_insert_pos = None
        best_cluster_id = None
        best_cluster_orientation = None
        
        for cluster_id, cluster_tour in remaining.items():
            # Try both orientations (important for asymmetric!)
            for orient in [cluster_tour, cluster_tour[::-1]]:
                for i in range(len(merged_tour)):
                    a = merged_tour[i]
                    b = merged_tour[(i + 1) % len(merged_tour)]
                    
                    start_node = orient[0]
                    end_node = orient[-1]
                    
                    # Asymmetric cost calculation
                    cost_increase = D[a, start_node] + D[end_node, b] - D[a, b]
                    
                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_insert_pos = i + 1
                        best_cluster_id = cluster_id
                        best_cluster_orientation = orient
        
        merged_tour = (merged_tour[:best_insert_pos] +
                      list(best_cluster_orientation) +
                      merged_tour[best_insert_pos:])
        
        del remaining[best_cluster_id]
        
        if verbose:
            print(f"  Attached cluster {best_cluster_id} (cost increase: {best_cost_increase:.2f})")
    
    merge_cost = tour_cost(D, merged_tour)
    
    if verbose:
        print(f"✓ Merge complete: {len(merged_tour)} nodes, cost={merge_cost:.2f}\n")
    
    return merged_tour, merge_cost


# ============================================================================
# Main solver
# ============================================================================

def asymmetric_tsp_solve(
    D: np.ndarray,
    max_clusters: int = 20,
    restarts_per_cluster: int = 20,
    final_restarts: int = 10,
    auto_clusters: bool = True,
    n_clusters: int = None,
    clustering_method: str = 'hierarchical',
    verbose: bool = True
) -> Tuple[List[int], Dict[str, float]]:
    """
    Complete asymmetric TSP solver with hierarchical clustering.
    
    Handles:
    - Asymmetric distance matrices (D[i,j] ≠ D[j,i])
    - Negative edge weights
    
    Args:
        D: distance matrix (can be asymmetric, can have negative values)
        max_clusters: max k for auto-selection
        restarts_per_cluster: 2-opt restarts per cluster
        final_restarts: final multi-restart 2-opt
        auto_clusters: auto-select k using silhouette
        n_clusters: manual cluster count
        clustering_method: only 'hierarchical' supported (parameter kept for compatibility)
        verbose: print progress
    
    Returns:
        (final_tour, cost_breakdown)
    """
    n = D.shape[0]
    
    # Make a copy to avoid modifying original
    D = D.copy()
    
    # Ensure diagonal is zero
    np.fill_diagonal(D, 0)
    
    if verbose:
        print(f"{'='*60}")
        print(f"Asymmetric TSP Solver (n={n})")
        is_symmetric = np.allclose(D, D.T, atol=1e-6)
        has_negative = np.any(D < 0)
        print(f"Symmetric: {is_symmetric}, Has negative values: {has_negative}")
        print(f"Clustering method: {clustering_method}")
        print(f"{'='*60}\n")
    
    start_total = time.time()
    
    # Adjust restarts based on problem size to avoid getting stuck
    if n > 400:
        restarts_per_cluster = min(restarts_per_cluster, 30)
        final_restarts = min(final_restarts, 10)
        if verbose:
            print(f"[Note] Large instance (n={n}), reducing restarts:")
            print(f"  restarts_per_cluster: {restarts_per_cluster}")
            print(f"  final_restarts: {final_restarts}\n")
    elif n > 250:
        restarts_per_cluster = min(restarts_per_cluster, 30)
        final_restarts = min(final_restarts, 15)
    
    # Step 1: Select k
    if auto_clusters:
        k = auto_select_clusters(D, max_clusters=max_clusters, verbose=verbose)
    else:
        k = n_clusters if n_clusters else max(2, int(np.sqrt(n / 2)))
        if verbose:
            print(f"[Manual] Using k={k} clusters\n")
    
    # Step 2: Cluster with hierarchical method
    labels, coords = cluster_asymmetric_hierarchical(D, k, verbose=verbose)
    
    # Step 3: Solve clusters
    cluster_tours = solve_clusters_parallel_asymmetric(
        D, labels, k, restarts_per_cluster, verbose=verbose
    )
    intra_cluster_cost = sum(cost for _, cost in cluster_tours.values())
    
    # Step 4: Merge
    merged_tour, merge_cost = merge_clusters_asymmetric(D, cluster_tours, verbose=verbose)
    
    # Step 5: Final refinement
    if verbose:
        print(f"[Final Refinement] Running {final_restarts} asymmetric 2-opt restarts...")
    
    # Set timeout based on problem size
    n = D.shape[0]
    if n <= 200:
        timeout = None  # No timeout for small problems
    elif n <= 500:
        timeout = 300  # 5 minutes for 500 cities
    else:
        timeout = 600  # 10 minutes for larger
    
    final_tour, final_cost = multi_start_asymmetric_2opt(
        D,
        restarts=final_restarts,
        initial_perm=merged_tour,
        verbose=verbose,
        timeout=timeout
    )
    
    total_time = time.time() - start_total
    
    # Results
    cost_breakdown = {
        'intra_cluster': intra_cluster_cost,
        'after_merge': merge_cost,
        'final': final_cost,
        'improvement': (merge_cost - final_cost) / abs(merge_cost) * 100 if merge_cost != 0 else 0,
        'time': total_time,
        'n_clusters': k,
        'method': clustering_method
    }
    
    if verbose:
        print(f"{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Clustering method: {clustering_method}")
        print(f"Number of clusters: {k}")
        print(f"Intra-cluster cost: {intra_cluster_cost:.2f}")
        print(f"After merging: {merge_cost:.2f}")
        print(f"Final (after refinement): {final_cost:.2f}")
        print(f"Improvement from merge: {cost_breakdown['improvement']:.2f}%")
        print(f"Total time: {total_time:.2f}s")
        print(f"{'='*60}\n")
    
    return final_tour, cost_breakdown


# ============================================================================
# Test script
# ============================================================================

if __name__ == "__main__":
    test_files = [
        "lab2/problem_r1_1000.npy",
        #"lab2/problem_r2_1000.npy",
    ]
    
    for filepath in test_files:
        try:
            print(f"\n{'#'*60}")
            print(f"Testing: {filepath}")
            print(f"{'#'*60}\n")
            
            D = np.load(filepath)
            
            tour, stats = asymmetric_tsp_solve(
                D,
                max_clusters=min(100, len(D) // 5),
                restarts_per_cluster=10,
                final_restarts=3,
                auto_clusters=True,
                verbose=True
            )
            
            print(f"\n{'='*60}")
            print(f"RESULTS FOR {filepath}")
            print(f"{'='*60}")
            print(f"Final cost: {stats['final']:8.2f}")
            print(f"Time: {stats['time']:6.2f}s")
            print(f"Clusters: {stats['n_clusters']}")
            print(f"{'='*60}")
            
        except FileNotFoundError:
            print(f"Skipping {filepath} (not found)")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()