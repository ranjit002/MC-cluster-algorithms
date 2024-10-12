import random
from collections import deque

import numpy as np
from stat_mech_library.disks import contracted_sets


def init_rand_state(n):
    """
    Returns an nxn array of spins (-1 or +1) randomly distributed
    """
    return np.random.choice([-1, 1], size=(n, n))


def init_align_state(n):
    """
    Returns an nxn array of +1 spins
    """
    return np.ones((n, n), dtype=int)


# Metropolis Algorithm
def metropolis_move(state, j):
    """
    Performs n**2 single Metropolis algorithm updates (i.e. a Monte Carlo step)
    """
    n = len(state)
    # Copy state to avoid changing the global variable "state"
    state_copy = state.copy()

    for _ in range(n**2):
        # Pick random spin on lattice
        i, k = np.random.randint(0, n, 2)

        neighbour_sum = state_copy[i, k] * (
            state_copy[i, (k + 1) % n]
            + state_copy[i, (k - 1) % n]
            + state_copy[(i + 1) % n, k]
            + state_copy[(i - 1) % n, k]
        )

        # If criteria of Metropolis algorithm met then flip the spin
        if neighbour_sum <= 0 or np.random.rand() < np.exp(-2 * j * neighbour_sum):
            state_copy[i, k] *= -1

    return state_copy


# Wolff Algorithm
def find_neighbours(n, list_index):
    """
    Given a list_index for a list with n**2 elements,
    return the indices of the neighbours of the list_index,
    if the list was reshapen into an nxn matrix.
    In this order: left, right, down, up

    Example:
    The neighbours of the 5th entry in the matrix below are: 4, 6, 9, 1
    The neighbours of the 7th entry are: 6, 4, 11, 3
    [[0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15]]
    """
    left = (list_index - list_index % n) + (list_index - 1) % n
    right = (list_index - list_index % n) + (list_index + 1) % n
    down = (list_index + n) % n**2
    up = (list_index - n) % n**2

    return left, right, down, up


def wolff_move(state, j, return_cluster_size=False):
    """
    Perform a single Wolff cluster move on the current state.
    The return_cluster_size bool if set to True returns the size of the cluster grown

    The (nxn) matrix of spins is represented as a list containing the flattened matrix.
    For example for a 4x4 system, the index of each matrix entry is:
    mapping = [
     [0, 1, 2, 3],
     [4, 5, 6, 7],
     [8, 9, 10, 11],
     [12, 13, 14, 15]]

    i.e. the (n, m) matrix entry is mapped onto the (n*n + m) entry in the list.
    """
    n = len(state)

    # Flatten state into a list
    state_copy = state.flatten()

    # Probability of adding a spin (parallel to the seed spin) to the cluster
    flip_prob = max(0, 1 - np.exp(-2 * j))

    # Pick random spin on the lattice (i.e a random index in the list)
    seed_index = np.random.randint(n**2)

    # Change the spin of the seed and store the original value of the spin.
    cluster_spin = state_copy[seed_index]
    state_copy[seed_index] = -cluster_spin

    # Keep track of cluster size
    cluster_size = 1

    # The unvisited deque contains the indices of all spins to be potentialy added to the cluster
    # A deque is being used to preserve the order the indices are added
    unvisited = deque([seed_index])

    while unvisited:
        index = unvisited.pop()

        # Iterating over the neighbours taking into account the periodic boundaries
        for neighbour_index in find_neighbours(n, index):

            # If criteria of Wolff algorithm met (Parellel spin and with probability flip_prob)
            # then flip the spin.
            if (
                state_copy[neighbour_index] == cluster_spin
                and np.random.rand() < flip_prob
            ):
                state_copy[neighbour_index] = -cluster_spin

                # Add the flipped spin to the "univisted" deque,
                # so its neighbours can be potentially added to the cluster
                unvisited.appendleft(neighbour_index)
                cluster_size += 1

    # np.reshape is used to turn the list of n**2 elements back into an nxn matrix
    if not return_cluster_size:
        return np.reshape(state_copy, (n, n))
    else:
        return np.reshape(state_copy, (n, n)), cluster_size


# Swendsen–Wang Algorithm
def sw_move(state, j):
    """
    Perform a single Swendsen-Wang cluster update

    The (nxn) matrix of spins is represented as a list containing the flattened matrix.
    For example for a 4x4 system, the index of each matrix entry is:
    mapping = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]]

    i.e. the (n, m) matrix entry is mapped onto the (n*n + m) entry in the list.
    """
    n = len(state)

    # Flatten state into a list
    state_copy = state.flatten()

    # Probability of forming a bond between neighbouring parallel spins
    flip_prob = max(0, 1 - np.exp(-2 * j))

    # The clusters list will contain tuples of all the lattice points sharing a bond according to the SW algorithm
    clusters = []

    # Iterate through all of the spins in the list
    for index in range(n**2):
        # Form a cluster that contains the current index
        cluster = set([index])

        # Iterating over the neighbours taking into account the periodic boundaries
        for neighbour_index in find_neighbours(n, index):

            # If criteria of SW algorithm met (Paralle spin and with probability flip_prob), then add the spin to the cluster
            # The neighbour_index > index criterion ensures you don't try to form a bond that was already tried to be created
            if (
                neighbour_index > index
                and state_copy[neighbour_index] == state_copy[index]
                and random.random() < flip_prob
            ):
                cluster.add(neighbour_index)

        # A single element cluster shouldn't be flipped, so it's not included to the clusters list
        if len(cluster) > 1:
            clusters.append(cluster)

    # Merge all clusters that share elements, this effectively finds the "full" clusters of spins,
    # Since if two clusters overlap they are treated as one bigger cluster.
    clusters = contracted_sets(clusters)

    # For each cluster attempt to flip the spin with probability 0.5
    for cluster in clusters:
        if random.random() < 1 / 2:
            state_copy[cluster] *= -1

    # np.reshape is used to turn the list of n**2 elements back into an nxn matrix
    return np.reshape(state_copy, (n, n))


"""
I tried to also use a cluster finding algorithm (Hoshen-Kopelman),
available in the scipy library (scipy.ndimage.label) for the SW algorithm.

However, I ran into the issue that periodic boundary conditions aren't supported by this function.
So I tried to manually "fix" the boundary conditions, using the function find_clusters_periodic().

The idea was to use numpy matrix operations to do many of the operations all at once,
rather tham manually visiting each lattice site using a for loop.

However, the code turned out to be much slower than the above implementation.
Below I left my code for this alternative implementation.
"""

from scipy.ndimage import label as cluster


def order_indices(cluster_mat, offset=0):
    """ "
    Given a matrix (cluster_mat) with each cluster indexed with an integer (e.g. 1, 2, 4, 6),
    return a matrix with the clusters indexed by integers in increasing order (1, 2, 3, 4)
    """
    # Keys contains all the cluster indices present in the cluster_mat
    keys = np.unique(cluster_mat)
    # The wanted indices are just integers in increasing order
    values = np.arange(len(keys))

    # Every integer except for 0 is augmented by "offset"
    if offset != 0:
        values = [value + offset for value in values]
        values[0] = 0

    # Form a mapping that maps the keys to the values
    mapping = np.zeros(keys.max() + 1, dtype=int)
    mapping[keys] = values

    # Change all cluster indices to the new reordered indices
    return mapping[cluster_mat]


def find_clusters_periodic(mat):
    """
    Subdivide matrix(mat) into clusters (Complicated by the fact that scipy.ndimage.label() doesn't support periodic boundary conditions)
    The scipy function can only find clusters in a matrix with 0s and 1s (where the 0s are ignored), so the function first finds the clusters of +1,
    then the clusters of -1.
    """
    n = len(mat)

    # Finding all the clusters of spin downs
    bool_mat = mat == -1
    flip_mat_1 = cluster(bool_mat * 1)[0]

    # At the vertical boundaries wherever there are two clusters they need to be made "equivalent"
    # So that the periodic boundary conditions are met.
    vertical_boundary = np.logical_and(bool_mat[0], bool_mat[-1])
    for i in range(n):
        if vertical_boundary[i]:
            indices = (flip_mat_1[0, i], flip_mat_1[-1, i])

            # Replace every instance of the larger cluster index with the smaller cluster index
            min_index = min(indices)
            max_index = max(indices)
            flip_mat_1 = np.where(flip_mat_1 == max_index, min_index, flip_mat_1)

    # Do the same with the horizontal boundary
    horizontal_boundary = np.logical_and(bool_mat[:, 0], bool_mat[:, -1])
    for i in range(n):
        if horizontal_boundary[i]:
            indices = (flip_mat_1[i, 0], flip_mat_1[i, -1])

            min_index = min(indices)
            max_index = max(indices)
            flip_mat_1 = np.where(flip_mat_1 == max_index, min_index, flip_mat_1)
    # Reorder the indices so that they are in increasing order
    flip_mat_1 = order_indices(flip_mat_1)

    # Finding all the clusters of spin ups
    bool_mat = mat == 1
    flip_mat_2 = cluster(bool_mat * 1)[0]

    # Do the same procedure as for the spin downs
    vertical_boundary = np.logical_and(bool_mat[0], bool_mat[-1])
    for i in range(n):
        if vertical_boundary[i]:
            indices = (flip_mat_2[0, i], flip_mat_2[-1, i])

            min_index = min(indices)
            max_index = max(indices)
            flip_mat_2 = np.where(flip_mat_2 == max_index, min_index, flip_mat_2)

    horizontal_boundary = np.logical_and(bool_mat[:, 0], bool_mat[:, -1])
    for i in range(n):
        if horizontal_boundary[i]:
            indices = (flip_mat_2[i, 0], flip_mat_2[i, -1])

            min_index = min(indices)
            max_index = max(indices)
            flip_mat_2 = np.where(flip_mat_2 == max_index, min_index, flip_mat_2)
    # Add an offset to the indices of the spin up clusters, so that they don't "clash" with the indices of the spin down clusters.
    flip_mat_2 = order_indices(flip_mat_2, np.amax(flip_mat_1))

    return flip_mat_1 + flip_mat_2


def sw_move(state, j):
    """
    Perform a single Swendsen-Wang cluster update with specified seed on the state
    """
    n = len(state)
    flip_prob = max(0, 1 - np.exp(-2 * j))

    # Probabilistically remove some parts of the lattice, so they aren't bonded to their neighbours
    rand_mat = np.random.choice([True, False], (n, n), p=[flip_prob, 1 - flip_prob])
    masked_mat = np.where(rand_mat, state, 0)

    # Find the clusters in the masked matrix
    cluster_mat = find_clusters_periodic(masked_mat)

    # For each cluster generate a boolean that dictates whether that cluster should be flipped
    max_cluster_index = np.amax(cluster_mat)
    flip_vec = np.random.choice([True, False], max_cluster_index + 1)

    # The 0 positions should be left unflipped
    flip_vec[0] = False

    return np.where(flip_vec[cluster_mat], -state, state)
