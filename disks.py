import random
from collections import deque

import numpy as np
from matplotlib.pyplot import Circle
from networkx import Graph, add_path, connected_components


def init_disk_no_overlap_state(num_of_disks, radius):
    """
    Returns list of random non-overlapping points,
    where each coordinate is between 0 and 1
    (each point has radius = radius)
    """
    # If too many disks or radius of disks too large, can't form a non overlapping state
    # Raise OverflowError Error
    if num_of_disks * 4 * radius**2 > 1 / 2:
        raise OverflowError

    # Pick a random coordinate to form a disk at. Fit the rest of the disks around this point.
    coordinates = np.random.rand(1, 2)

    # Keep attempting to add a disk until you have num_of_disks
    while len(coordinates) < num_of_disks:
        potential_disk = np.random.rand(1, 2)

        # If the proposed disk doesn't overlap with any other then add it to coordinates list
        if not any(wrap_distance(coordinates, potential_disk) < 2 * radius):
            coordinates = np.vstack((coordinates, potential_disk))

    return coordinates


def init_disk_rand_state(num_of_disks):
    """
    Returns a list of random points,
    where each coordinate is between 0 and 1 (each point has radius = radius)
    """
    return np.random.rand(num_of_disks, 2)


def draw_disks(ax, state, radius, draw_index=False):
    """
    Given a list of points (state), draw a circle with given radius on the ax
    The draw_index bool controls whether a number on each disk should be drawn
    """
    num_of_disks = len(state)

    for index in range(num_of_disks):
        coords = state[index]

        # Draw circle at coords and draw index label
        ax.add_patch(Circle(coords, radius))
        if draw_index:
            ax.text(coords[0], coords[1], index, va="center", ha="center")

    # Remove axes ticks
    ax.set_xticks([])
    ax.set_yticks([])


def wrap_distance(x0, x1, n=1):
    """
    Given a list of points x0 and a point x1 (x1 must have x, y cooridnates between 0 and 1)
    Return the distance between them, if they are on a N*N square grid with periodic boundary conditions.
    By default the shape of the grid is 1*1
    """
    # Find the Euclidean x, y distance between the points
    delta = np.abs(x0 - x1)

    # Since the points are on a NxN grid,
    # if the Euclidean distance between two points (delta) is greater than N/2,
    # you can go around the periodic boundary and get a smaller distance (delta - N).
    delta = np.where(delta > n / 2, delta - n, delta)

    # Pythagoras' theorem
    return np.sqrt((delta**2).sum(axis=-1))


def contracted_sets(cluster_list):
    """
    Given a list of tuples with some elements,
    Return a list where you merged all the tuples that share at least one element.
    """
    # Create a graph
    G = Graph()

    # Add each cluster to the graph
    for cluster in cluster_list:
        add_path(G, cluster)

    # Find the connected componenets in the graph (i.e. the contracted sets)
    contracted_clusters = list(connected_components(G))

    return [list(cluster) for cluster in contracted_clusters]


def find_overlaps(state, disk_index, radius):
    """
    Given a list of cordinates of disks with radius = radius,
    return the indices of the disks that overlap with the disk (disk_index)
    or the version of the disk that is point reflected by (0, 0)
    """
    # Find distance of all disks from the disk given by "disk_index"
    distances1 = wrap_distance(state, state[disk_index])

    # Find distance of all disks from the point relected version (about (0,0)) of "disk_index"
    distances2 = wrap_distance(1 - state, state[disk_index])

    # Find the overlaps i.e where the separation is less than twice the radius
    cluster = set(np.where(distances1 < 2 * radius)[0])
    cluster = cluster.union(set(np.where(distances2 < 2 * radius)[0]))

    # Remove the disk_index
    cluster.discard(disk_index)

    return cluster


def hard_disks_move(state, radius):
    """
    Evolving the state according to the non interacting disks algorithm
    """
    num_of_disks = len(state)

    # Pick a random point in the matrix to be the pivot
    # Recentre the coordinates so that the pivot is at the centre
    pivot = np.random.rand(1, 2)
    recentered_state = (state - pivot) % 1

    # The clusters list will contain tuples with disk indices that are in the same cluster
    # Where the notion of cluster is defined in the disk model algorithm
    clusters = []

    # Find all the clusters of overlapping disks
    for index in range(num_of_disks):
        # Find indices of overlapping disks,
        # by checking if distance between two disks is less than twice their radius
        # Coordinates truncated to avoid finding the distance between two particle more than once.
        # i.e. if you checked for overlap between disk 1 and 2,
        # you don't check overlap between disk 2 and 1.
        truncated_list = recentered_state[index:]
        distances1 = wrap_distance(truncated_list, recentered_state[index])
        distances2 = wrap_distance(1 - truncated_list, recentered_state[index])

        cluster = set(index + np.where(distances1 < 2 * radius)[0])
        cluster = cluster.union(set(index + np.where(distances2 < 2 * radius)[0]))

        # Add cluster to list of clusters
        clusters.append(cluster)

    # Merge the clusters that share an element
    clusters = contracted_sets(clusters)

    # For each cluster invert the positions of the disks in the cluster with probabiity 0.5
    for cluster in clusters:
        if random.choice([True, False]):
            recentered_state[cluster] = 1 - recentered_state[cluster]

    return recentered_state


def interacting_disk_move(state, radius, beta_potential):
    """
    Evolving configuration according to the interacting disks algorithm
    State is a list of coordinates of the disks in a 1x1 grid
    Radius is the radius of each disk
    beta_potential is a function of both separation of the disks and their radius (in this order),
    which is used to find the potential energy of a configuration.
    """
    num_of_disks = len(state)

    # Pick a random point in the matrix to be the pivot
    # Recentre the matrix so that the pivot is at the centre
    pivot = np.random.rand(1, 2)
    recentered_state = (state - pivot) % 1

    # The clusters list will contain tuples with disk indices that are in the same cluster
    # Where the notion of cluster is defined in the disk model algorithm
    seed_index = np.random.randint(num_of_disks)
    recentered_state[seed_index] = 1 - recentered_state[seed_index]

    # Store the the disks that have been flipped (to avoid reflipping them)
    # and those that could potentiallly be flipped
    flipped = set([seed_index])
    stack = deque([seed_index])

    while stack:
        i = stack.pop()

        # Find the neighbours of the disk currently being "visited", that haven't been visisted.
        neighbours = find_overlaps(recentered_state, i, radius)
        neighbours = neighbours - flipped

        for j in neighbours:
            # Find the separation before and after the proposed flip (point reflection)
            current_sep = wrap_distance(1 - recentered_state[i], recentered_state[j])
            proposed_sep = wrap_distance(recentered_state[i], recentered_state[j])

            # Find the energy change if you carry out the proposed flip (point reflection)
            beta_deltaE = beta_potential(proposed_sep, radius) - beta_potential(
                current_sep, radius
            )

            # Need to prevent exp() overflow,
            # which can happen if beta_deltaE is sufficiently large and negative
            if beta_deltaE < 0:
                prob = 0
            else:
                prob = max(0, 1 - np.exp(-beta_deltaE))

            # Point reflect the particle with probability "prob"
            if random.random() < prob:
                recentered_state[j] = 1 - recentered_state[j]

                # Add the particle that was just flipped to the "stack" deque
                stack.appendleft(j)
                flipped.add(j)

    return recentered_state
