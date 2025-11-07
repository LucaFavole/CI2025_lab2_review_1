from typing import List, Optional
import numpy as np
import os
import matplotlib.pyplot as plt

# The plot_cities function (copied from your code for completeness)
def _classical_mds(dist_matrix, n_components=2):
    """Classical MDS (Torgerson) to embed distances into Euclidean coords."""
    D = np.asarray(dist_matrix, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("dist_matrix must be a square 2D array")
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J.dot(D2).dot(J)
    eigvals, eigvecs = np.linalg.eigh(B)  # ascending eigenvalues
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # keep only positive eigenvalues
    pos = eigvals > 0
    k = min(n_components, np.count_nonzero(pos))
    if k == 0:
        # fallback to first components even if non-positive
        k = min(n_components, n)
    L = np.diag(np.sqrt(np.maximum(eigvals[:k], 0.0)))
    V = eigvecs[:, :k]
    return V.dot(L)

def plot_cities(dist_matrix, labels=None, annotate=True, figsize=(9, 7),
                marker='o', s=60, cmap='tab20', title="City map (MDS embedding)",
                groups=None, connect=True, close_groups=False, line_kwargs=None):
    """
    Plot cities as points by embedding the distance matrix into 2D using classical MDS.
    Additional features:
      - groups: optional list of groups. Each group is an iterable of city indices OR city labels.
                Example: [[0,1,2], ['Rome','Milan'], [3,4]]
      - connect: if True, draw lines between consecutive members of each group.
      - close_groups: if True, close the group loop (connect last -> first).
      - line_kwargs: dict passed to matplotlib.plot for group lines (color/linewidth/etc).
    Returns the matplotlib Axes.
    """
    D = np.asarray(dist_matrix, dtype=float)
    n = D.shape[0]
    if labels is None:
        labels = [str(i) for i in range(n)]
    if len(labels) != n:
        raise ValueError("labels length must match dist_matrix size")

    coords = _classical_mds(D, n_components=2)
    x, y = coords[:, 0], coords[:, 1]

    fig, ax = plt.subplots(figsize=figsize)
    cmap_obj = plt.get_cmap(cmap)
    colors = [cmap_obj(i % cmap_obj.N) for i in range(n)]
    scatter = ax.scatter(x, y, c=colors, s=s, marker=marker, edgecolors='k', linewidths=0.4)

    if annotate:
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (x[i], y[i]), xytext=(4, 2), textcoords='offset points', fontsize=9)

    # Draw group lines if requested
    if groups:
        # default line style
        if line_kwargs is None:
            line_kwargs = dict(linestyle='-', linewidth=1.6, alpha=0.8)
        # color cycle for groups
        color_cycle = plt.rcParams.get('axes.prop_cycle').by_key().get('color', None)
        for gi, group in enumerate(groups):
            # resolve indices for this group
            idxs = []
            for item in group:
                if isinstance(item, str):
                    try:
                        idxs.append(labels.index(item))
                    except ValueError:
                        raise ValueError(f"Label '{item}' not found in labels")
                elif isinstance(item, int):
                    if item < 0 or item >= n:
                        raise IndexError(f"Index {item} out of bounds for {n} cities")
                    idxs.append(item)
                else:
                    raise TypeError("Group members must be either int (index) or str (label)")

            if len(idxs) < 2:
                # nothing to connect, optionally mark differently
                continue

            gx = coords[idxs, 0]
            gy = coords[idxs, 1]

            # choose a color: use provided color in line_kwargs if present else cycle
            lk = dict(line_kwargs)  # copy so we can set color per-group
            if 'color' not in lk:
                if color_cycle:
                    lk['color'] = color_cycle[gi % len(color_cycle)]
                else:
                    lk['color'] = cmap_obj(gi % cmap_obj.N)

            # plot the polyline
            if connect:
                ax.plot(gx, gy, **lk)
                if close_groups:
                    ax.plot([gx[-1], gx[0]], [gy[-1], gy[0]], **lk)

            # optionally mark group members with a contrasting marker
            ax.scatter(gx, gy, s=s*1.1, facecolors='none', edgecolors=lk['color'], linewidths=1.2)

    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('MDS X')
    ax.set_ylabel('MDS Y')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return ax

