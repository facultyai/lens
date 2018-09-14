from tdigest.tdigest import TDigest, Centroid


def tdigest_from_centroids(seq):
    """Create a TDigest from a list of centroid means and weights tuples

    Parameters
    ----------

    seq : iterable
        List of tuples of length 2 that contain the centroid mean and weight
        from a TDigest.
    """

    tdigest = TDigest()

    for mean, weight in seq:
        tdigest.C.insert(mean, Centroid(mean, weight))
        tdigest.n += weight

    return tdigest


def centroids_from_tdigest(tdigest):
    """Return centroid means and weights from a TDigest instance"""

    if not isinstance(tdigest, TDigest):
        raise ValueError("Argument must be a TDigest instance")

    means = [c.mean for c in tdigest.C.values()]
    counts = [c.count for c in tdigest.C.values()]

    return means, counts
