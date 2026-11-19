"""Route-A neighbor fast paths versus the pre-vectorization scalar contract."""
import numpy as np
import pytest

from models.instance_route_a_geometry import BOUNDARY_ROUNDOFF_M, NeighborIndex


def scalar_anchor(index, points, radius=.005):
    """Frozen reference: the original per-query selection and arithmetic."""
    anchors = np.full(len(points), -1, np.int64)
    distance = np.full(len(points), np.inf, np.float64)
    if index.all_tree is None or not len(points):
        return anchors, distance
    d, rows = index.all_tree.query(points, k=min(2, len(index.points)))
    if len(index.points) == 1:
        d, rows = d[:, None], rows[:, None]
    for i in range(len(points)):
        first_distance = float(d[i, 0])
        if first_distance > radius + BOUNDARY_ROUNDOFF_M:
            continue
        choices = np.asarray([rows[i, 0]], np.int64)
        if d.shape[1] > 1 and d[i, 1] == first_distance:
            choices = np.asarray(index.all_tree.query_ball_point(
                points[i], np.nextafter(first_distance, np.inf)), np.int64)
        actual = np.linalg.norm(index.points[choices] - points[i], axis=1)
        winner = np.lexsort((index.source_ids[choices], actual))[0]
        anchors[i], distance[i] = int(choices[winner]), float(actual[winner])
    return anchors, distance


def scalar_association(index, points, components, radius=.010):
    """Frozen reference including the original 3/4 cutoff tie fallback."""
    indices = np.full((len(points), 3), -1, np.int64)
    weights = np.zeros((len(points), 3), np.float64)
    d1, d3 = np.full(len(points), np.nan), np.full(len(points), np.nan)
    counts = np.zeros(len(points), np.int64)
    component_missing = np.ones(len(points), bool)
    for component in np.unique(components):
        if int(component) not in index.groups:
            continue
        query_rows = np.flatnonzero(components == component)
        source_rows, tree = index.groups[int(component)]
        component_missing[query_rows] = False
        k = min(4, len(source_rows))
        distances, nearest = tree.query(points[query_rows], k=k)
        if k == 1:
            distances, nearest = distances[:, None], nearest[:, None]
        for local_row, query_row in enumerate(query_rows):
            selected = source_rows[np.asarray(nearest[local_row]).reshape(-1)]
            dist = np.linalg.norm(index.points[selected] - points[query_row], axis=1)
            if k == 4 and distances[local_row, 2] == distances[local_row, 3]:
                selected = source_rows[np.asarray(tree.query_ball_point(
                    points[query_row], np.nextafter(distances[local_row, 2], np.inf)), np.int64)]
                dist = np.linalg.norm(index.points[selected] - points[query_row], axis=1)
            order = np.lexsort((index.source_ids[selected], dist))[:3]
            selected, dist = selected[order], dist[order]
            d1[query_row] = dist[0]
            if len(dist) >= 3:
                d3[query_row] = dist[2]
            keep = dist <= radius + BOUNDARY_ROUNDOFF_M
            selected, dist = selected[keep], dist[keep]
            count = len(selected)
            counts[query_row] = count
            indices[query_row, :count] = selected
            if not count:
                continue
            if dist[0] == 0:
                weights[query_row, 0] = 1.
            else:
                reciprocal = 1. / dist
                weights[query_row, :count] = reciprocal / reciprocal.sum()
    return {'indices': indices, 'weights': weights, 'valid': counts > 0,
            'counts': counts, 'd1': d1, 'd3': d3,
            'component_missing': component_missing}


def assert_identical(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_identical(actual[key], expected[key])
    elif isinstance(expected, tuple):
        for left, right in zip(actual, expected):
            assert_identical(left, right)
    else:
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        # Include exact float64 rounding, NaN payloads, and signed zero.
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('source_count', [0, 1, 2, 3, 4, 33, 2048])
def test_neighbors_match_scalar_bits_for_random_mixed_and_missing_components(source_count):
    rng = np.random.default_rng(100 + source_count)
    source = rng.uniform(-.04, .04, (source_count, 3))
    source_ids = rng.permutation(source_count) * 7 + 11
    source_components = rng.integers(1, 4, source_count)
    points = rng.uniform(-.04, .04, (257, 3))
    components = rng.integers(1, 5, len(points))
    count = min(20, source_count)
    points[:count], components[:count] = source[:count], source_components[:count]
    index = NeighborIndex(source, source_ids, source_components)
    assert_identical(index.nearest_anchor(points), scalar_anchor(index, points))
    assert_identical(index.associate(points, components), scalar_association(index, points, components))
    assert_identical(index.nearest_anchor(points[:0]), scalar_anchor(index, points[:0]))
    assert_identical(index.associate(points[:0], components[:0]),
                     scalar_association(index, points[:0], components[:0]))


@pytest.mark.parametrize('radius', [.005, .010])
def test_neighbors_keep_boundary_ties_duplicate_coordinates_and_exact_point_precedence(radius):
    source = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
                       [radius, 0., 0.], [-radius, 0., 0.], [0., radius, 0.], [0., -radius, 0.],
                       [0., 0., radius], [0., 0., -radius]], np.float64)
    ids = np.array([40, 10, 30, 20, 90, 60, 80, 50, 70, 100])
    points = np.array([[0., 0., 0.], [radius, 0., 0.],
                       [2 * radius + .5e-12, 0., 0.], [2 * radius + 2e-12, 0., 0.]])
    index = NeighborIndex(source, ids, np.ones(len(source), np.int64))
    components = np.ones(len(points), np.int64)
    assert_identical(index.nearest_anchor(points, radius), scalar_anchor(index, points, radius))
    assert_identical(index.associate(points, components, radius),
                     scalar_association(index, points, components, radius))
    selected = index.associate(points, components, radius)
    assert ids[selected['indices'][0]].tolist() == [10, 20, 30]
    np.testing.assert_array_equal(selected['weights'][0], [1., 0., 0.])
    assert selected['valid'][2] and not selected['valid'][3]


def test_neighbors_resolve_internal_ties_without_scalar_ball_query():
    class CountingTree:
        def __init__(self, tree):
            self.tree, self.calls = tree, 0
        def query(self, *args, **kwargs):
            return self.tree.query(*args, **kwargs)
        def query_ball_point(self, *args, **kwargs):
            self.calls += 1
            return self.tree.query_ball_point(*args, **kwargs)

    source = np.array([[.001, 0., 0.], [.003, 0., 0.], [-.003, 0., 0.], [.004, 0., 0.]])
    ids = np.array([100, 50, 25, 10])
    index = NeighborIndex(source, ids, np.ones(4, np.int64))
    rows, tree = index.groups[1]
    counted = CountingTree(tree)
    index.groups[1] = rows, counted
    index.all_tree = CountingTree(index.all_tree)
    points = np.zeros((5000, 3))
    association = index.associate(points, np.ones(len(points), np.int64))
    assert ids[association['indices'][0]].tolist() == [100, 25, 50]
    assert counted.calls == 0  # 2/3 tie can be resolved inside the batched short sort.
    anchors, distance = index.nearest_anchor(points)
    assert np.all(anchors == 0) and np.all(distance == .001)
    assert index.all_tree.calls == 0


def test_contact_ball_matches_tree_at_boundary_duplicate_ids_and_large_components():
    import math
    from models.instance_route_a_geometry import NeighborIndex, HEIGHT_LIMIT_M
    from suction20.instance_geometry import BOUNDARY_ROUNDOFF_M
    radius = math.hypot(.022, HEIGHT_LIMIT_M) + 2 * BOUNDARY_ROUNDOFF_M
    rng = np.random.default_rng(23)
    for count, tied in ((200, False), (200, True), (32769, False)):
        points = rng.uniform(-.05, .05, (count, 3))
        points[:3] = [[radius, 0., 0.], [np.nextafter(radius, 0.), 0., 0.],
                      [np.nextafter(radius, np.inf), 0., 0.]]
        ids = rng.permutation(count)
        if tied:
            ids[1] = ids[0]
        index = NeighborIndex(points, ids, np.zeros(count, int))
        rows, tree = index.groups[0]
        for center in (np.zeros(3), np.array([.00123, .00456, -.00789])):
            found = rows[np.array(tree.query_ball_point(center, radius), dtype=np.intp)]
            ordered = found[np.argsort(ids[found], kind='stable')]
            np.testing.assert_array_equal(index.support_ball(center, 0), ordered)
        del index._tied_source_ids  # transient geometry prepared before this optimization
        assert len(index.support_ball(points[0], 0))
