import numpy as np
from collections import deque
import numpy.random


def draw_sample(samples):
    """Draw a random sample from a given list.

    Args:
        samples: List of samples.

    Returns:
        Randomly selected sample.

    """
    idx = int(numpy.random.randint(low=0, high=len(samples)-1, size=1))
    return samples[idx]


def sample_particles(n_samples, sample_params):
    """Multi distribution functions to generate random scales and 3D coordinates of (spherical) samples.

    Args:
        n_samples: Number of samples.
        sample_params: Dictionary containing size sampling parameters.
        base_sample_limit: Positional norm limit in x,y,z for the first sample.

    Returns:
        List of sample scales and positions.
    """
    sample_3d = sample_params['mesh_position3d']
    base_sample_limit = sample_params['mesh_base_sample_limit'] / sample_params['mesh_unit']
    max_agglomerates = sample_params['mesh_max_agglomerates']
    size_dist = sample_params['mesh_size_dist']
    position_algo = sample_params['particle_placement']

    if size_dist == 'bimodal':
        bimodal_params = sample_params['bimodal_params']
        mu1 = bimodal_params['mean1']
        mu2 = bimodal_params['mean2']
        stddev1 = bimodal_params['stddev1']
        stddev2 = bimodal_params['stddev2']
        p = bimodal_params['p']
        sizes = bimodal_dist(mu1, mu2, stddev1, stddev2, p, n_samples)

    elif size_dist == 'gaussian':
        gauss_params = sample_params['gaussian_params']
        mean = gauss_params['mean']
        stddev = gauss_params['stddev']
        sizes = gaussian_dist(mean, stddev, n_samples)

    elif size_dist == 'lognormal':
        logn_params = sample_params['lognormal_params']
        mu = logn_params['mu']
        sigma = logn_params['sigma']
        sizes = lognormal_dist(mu, sigma, n_samples)


    samples = []
    agglomerates = random_split(sizes, max_agglomerates)
    for agglomerate in agglomerates:
        radii = list(agglomerate)
        # Sample the particle positions
        if position_algo == 'random':
            samples.extend(place_random(radii, base_sample_limit, sample_3d))
        else:
            samples.extend(place_poisson(radii, base_sample_limit, sample_3d, samples))
    return np.asarray(samples)


def gaussian_dist(mu, sigma, n_samples):
    return np.random.normal(size=n_samples) * sigma + mu


def lognormal_dist(logn_mu, logn_std, n_samples):
    return np.random.lognormal(logn_mu, logn_std, n_samples)


def bimodal_dist(mu1, mu2, sigma1, sigma2, p1, num_samples):
  """
  Samples from a bimodal distribution.

  Args:
      mu1: Mean of the first mode.
      mu2: Mean of the second mode.
      sigma1: Standard deviation of the first mode.
      sigma2: Standard deviation of the second mode.
      p1: Probability of sampling from the first mode.
      num_samples: Number of samples to draw.

  Returns:
      A numpy array of samples.
  """

  # Sample from a Bernoulli distribution to decide which mode to sample from
  choices = np.random.choice([0, 1], size=num_samples, p=[1 - p1, p1])

  # Sample from the first mode
  samples_from_mode1 = np.random.normal(mu1, sigma1, size=np.sum(choices == 0))

  # Sample from the second mode
  samples_from_mode2 = np.random.normal(mu2, sigma2, size=np.sum(choices == 1))

  # Combine samples from both modes
  samples = np.concatenate((samples_from_mode1, samples_from_mode2))
  np.random.shuffle(samples)
  return samples

def random_split(arr, n):
    """
    Randomly splits a NumPy array into n splits with random sizes.

    Args:
        arr: The NumPy array to split.
        n: The number of splits.

    Returns:
        A list of n NumPy arrays, representing the splits.

    Raises:
        ValueError: If the array cannot be split into n pieces without exceeding or falling below n elements.
    """

    total_size = arr.shape[0]
    if total_size % n != 0:
        raise ValueError("Array size must be divisible by n for equal-sized splits.")

    # Generate random split sizes, ensuring they sum to total size
    split_sizes = np.random.randint(1, total_size // n + 1, size=n)
    remaining = total_size - np.sum(split_sizes[:-1])
    split_sizes[-1] = remaining

    # Create an empty list to store the splits
    splits = []

    # Split the array based on the random sizes
    start = 0
    for size in split_sizes:
        split = arr[start:start + size]
        splits.append(split)
        start += size

    return splits


def place_random(radiuses, base_sample_limit, sample_3d):
    return generate_uniform_sample(radiuses, base_sample_limit, sample_3d=sample_3d, size=(len(radiuses), 3))


def place_poisson(radiuses, base_sample_limit, sample_3d, agglo_samples):
    """Sample 3D points using multi radius Poisson disk sampling.

    Args:
        radiuses: List of radii for shapes.
        base_sample_limit: Positional norm limit in x,y,z
        sample_3d: Flag to sample in 3D or 2D.

    Returns:
        List of sampled points.
    """
    radius_queue = deque(radiuses)
    num_to_samples = len(radiuses)
    samples = []

    # Starting point
    seed = generate_uniform_sample(radius_queue.pop(), base_sample_limit, size=(1,3), sample_3d=sample_3d)
    if not np.any([
        _is_overlap(*seed[0], *sample)
        for sample in agglo_samples
    ]): samples.append(seed[0])
    else:
        return place_poisson(radiuses, base_sample_limit, sample_3d, agglo_samples)

    while len(samples) < num_to_samples:
        cur_sample_index = np.random.randint(low=0, high=len(samples))
        rng_sample = samples[cur_sample_index]

        while len(radius_queue) > 0:
            r_candidate = radius_queue[-1]
            candidate = _find_candidate(rng_sample, r_candidate, samples,
                                        sample_3d, agglo_samples)

            if candidate is None:
                break

            samples.append([*candidate, r_candidate])
            radius_queue.pop()

    return samples


def generate_uniform_sample(radius, base_sample_limit,  size, sample_3d=True,):
    """Generate a seed point for sampling.

    Args:
        radius: Radius of the sample.
        base_sample_limit: Positional norm limit in x,y,z for the first sample.
        sample_3d: Flag to generate seed in 3D or 2D.

    Returns:
        Seed point coordinates.
    """

    coords = np.random.uniform(low=-base_sample_limit, high=base_sample_limit, size=size)
    if not sample_3d and size == 3:
        coords[2] = 0.0
    if not sample_3d:
        coords[:,2]= 0.0
    return np.hstack((coords, np.asarray(radius).reshape(-1,1))).tolist()


def _generate_neighbour_candidates(x, y, z, r1, r2, sample_3d=True, freq=16):
    """Generate neighbor candidates on the surface of a sphere.

    Args:
        x, y, z: Coordinates of the center of the sphere.
        r1: Radius of the existing sample.
        r2: Radius of the candidate sample.
        sample_3d: Flag to generate candidates in 3D or 2D.
        freq: Frequency of points on the sphere.

    Returns:
        List of candidate points.
    """
    radius = r1 + r2
    theta = np.linspace(0, 2 * np.pi, freq)
    phi = np.linspace(0, np.pi, freq)
    theta, phi = np.meshgrid(theta, phi)
    x_sphere = (radius * np.sin(phi) * np.cos(theta) +
                x if sample_3d else radius * np.cos(theta) + x)
    y_sphere = (radius * np.sin(phi) * np.sin(theta) +
                y if sample_3d else radius * np.sin(theta) + y)
    z_sphere = radius * np.cos(phi) + z if sample_3d else theta * 0

    return list(zip(x_sphere.flatten(), y_sphere.flatten(),
                    z_sphere.flatten()))


def _is_overlap(x1, y1, z1, r1, x2, y2, z2, r2):
    return _distance(x1, y1, z1, x2, y2, z2) < r1 + r2


def _distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def _find_candidate(sample, r_candidate, samples, sample_3d, agglo_samples):
    """Find a valid candidate point for sampling.

    Args:
        sample: Coordinates and radius of the existing sample..
        r_candidate: Radius of the candidate sample.
        samples: List of already found samples.
        sample_3d: Flag to find a candidate in 3D or 2D.

    Returns:
        Coordinates and radius of the candidate sample if found, None otherwise.
    """
    candidates = _generate_neighbour_candidates(*sample,
                                                r_candidate,
                                                sample_3d=sample_3d)
    rindices = np.random.permutation(len(candidates))

    for ridx in rindices:
        if not np.any([
            _is_overlap(*sample, *candidates[ridx], r_candidate)
            for sample in [*samples, *agglo_samples]
        ]):
            return candidates[ridx]