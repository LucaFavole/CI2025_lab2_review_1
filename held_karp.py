import numpy as np
from itertools import combinations
from typing import Tuple, List, Optional

def held_karp(distance_matrix: np.ndarray) -> Tuple[List[int], float]:
    """
    Held-Karp exact TSP solver (dynamic programming).
    Accepts an (n x n) distance matrix (asymmetric allowed, diagonal/negative allowed).
    Returns a Hamiltonian cycle as a permutation (starting at 0) and its cost.

    Complexity: O(n^2 * 2^n) time, O(n * 2^n) memory. Practical up to ~20-22 nodes.
    """
    D = np.asarray(distance_matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("distance_matrix must be square")
    n = D.shape[0]
    if n == 0:
        return [], 0.0
    if n == 1:
        return [0], float(D[0, 0])

    # C[(bits, k)] = (cost, parent) where bits is a bitmask of visited nodes, k is last node
    C = {}
    # Initialize with paths that start at 0 and go to k
    for k in range(1, n):
        C[(1 << k, k)] = (D[0, k], 0)

    # Iterate subsets of increasing size (only nodes 1..n-1)
    for subset_size in range(2, n):
        for subset in combinations(range(1, n), subset_size):
            bits = 0
            for bit in subset:
                bits |= 1 << bit
            for k in subset:
                prev_bits = bits & ~(1 << k)
                # evaluate best previous node m that leads to k
                best_cost = float("inf")
                best_parent = -1
                for m in subset:
                    if m == k:
                        continue
                    key = (prev_bits, m)
                    if key not in C:
                        continue
                    cost = C[key][0] + D[m, k]
                    if cost < best_cost:
                        best_cost = cost
                        best_parent = m
                C[(bits, k)] = (best_cost, best_parent)

    # close the tour: consider returning to 0
    bits_all = (1 << n) - 1
    final_bits = bits_all & ~(1 << 0)
    best_cost = float("inf")
    parent = -1
    for k in range(1, n):
        key = (final_bits, k)
        if key not in C:
            continue
        cost = C[key][0] + D[k, 0]
        if cost < best_cost:
            best_cost = cost
            parent = k

    # reconstruct path
    path = []
    bits = final_bits
    last = parent
    for _ in range(n - 1):
        path.append(last)
        key = (bits, last)
        _, prev = C[key]
        bits = bits & ~(1 << last)
        last = prev
    path = [0] + path[::-1]  # start at 0, then visited nodes in order

    return path, float(best_cost)


def solve_from_file(npy_path: str) -> Tuple[List[int], float]:
    """Load distance matrix from .npy file and run Held-Karp."""
    D = np.load(npy_path)
    return held_karp(D)


def solve_solution(obj) -> Tuple[List[int], float]:
    """
    Convenience wrapper that accepts either:
      - a numpy array (distance matrix),
      - or an object with attribute `distance_matrix` (e.g. your helper.Solution).
    Returns (permutation, cost).
    """
    if hasattr(obj, "distance_matrix"):
        D = obj.distance_matrix
    else:
        D = np.asarray(obj)
    return held_karp(D)


# Example usage (in a notebook or script):
# from held_karp import solve_from_file, solve_solution
# perm, cost = solve_from_file("lab2/test_problem.npy")
# print(perm, cost)
if __name__ == "__main__":
    import sys
    print("Held-Karp TSP Solver")
    if len(sys.argv) != 2:
        print("Usage: python held_karp.py <distance_matrix.npy>")
        sys.exit(1)
    npy_path = sys.argv[1]
    print("Loading distance matrix from:", npy_path)
    perm, cost = solve_from_file(npy_path)
    print("Optimal tour:", perm)
    print("Optimal cost:", cost)