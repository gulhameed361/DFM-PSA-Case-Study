# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 17:38:38 2025

This method uses a robust inclusion-exclusion algorithm to precisely calculate the hypervolume.
This approach considers each point on the non-dominated front and calculates the unique volume it contributes,
while carefully subtracting any overlapping regions with other points.
This ensures that every part of the objective space is measured exactly once.
This provides a true and reliable measurement of the hypervolume,
which is essential for making robust decisions in a trust-region framework.

@author: Gul Hameed
"""

import numpy as np
from itertools import combinations

# --- CORE UTILITY FUNCTIONS ---

def is_dominated(point, pareto_front):
    """
    Checks if a point is dominated by any point in the Pareto front.
    A point 'p' is dominated by a point 'pf' if all objectives of 'pf' are
    less than or equal to 'p', and at least one objective is strictly less.
    (Minimization problem assumed).
    """
    # Convert to a NumPy array for consistent handling and check if it's empty.
    pareto_np = np.array(pareto_front)
    if pareto_np.size == 0:
        return False

    point_np = np.array(point)

    # Check for dominance (all objectives <= and at least one <)
    dominance_mask = np.all(pareto_np <= point_np, axis=1) & np.any(pareto_np < point_np, axis=1)
    return np.any(dominance_mask)


def update_pareto_front(pareto_front, new_point):
    """
    - If the new point is dominated, the front does not change.
    - If the new point dominates existing points, they are removed.
    - If the new point is non-dominated, it is added.
    """
    new_pareto = []

    # First, check if the new point is dominated by any existing point.
    if is_dominated(new_point, pareto_front):
        return pareto_front

    new_point_np = np.array(new_point, dtype=float)

    # De-duplicate: an equal point is non-dominated but adds no information and
    # inflates |P_k| (which drives the 2^|P_k| inclusion-exclusion cost). Skip it.
    for p in pareto_front:
        if np.allclose(np.array(p, dtype=float), new_point_np):
            return pareto_front

    # If not dominated, filter out any existing points that are now dominated by the new one.
    for p in pareto_front:
        p_np = np.array(p, dtype=float)
        # If new_point dominates p, don't add p to the new front.
        if not (np.all(new_point_np <= p_np) and np.any(new_point_np < p_np)):
            new_pareto.append(p)

    # Finally, add the new point to the updated front.
    new_pareto.append(new_point)

    return new_pareto



def calculate_hypervolume(pareto_front, ref_point):
    """
    Calculates the hypervolume of a Pareto front using the inclusion-exclusion principle.
    This is a corrected, robust algorithm.
    """
    if not pareto_front:
        return 0.0

    points = np.array(pareto_front)
    ref = np.array(ref_point)
    num_points = len(points)

    total_hv = 0.0
    for i in range(1, num_points + 1):
        # Iterate through all subsets of the Pareto front of size i
        for subset_indices in combinations(range(num_points), i):
            subset = points[list(subset_indices)]

            # Find the intersection box for this subset of points
            min_coords = np.max(subset, axis=0)

            # Check if this intersection box has a positive volume
            if np.all(min_coords < ref):
                # Calculate the volume of the intersection box
                volume = np.prod(ref - min_coords)

                # Apply the inclusion-exclusion principle
                if i % 2 == 1:
                    total_hv += volume
                else:
                    total_hv -= volume

    return total_hv

