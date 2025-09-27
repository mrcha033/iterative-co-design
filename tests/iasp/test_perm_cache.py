from icd.runtime.perm_cache import load_entry, save_entry, should_invalidate


def test_save_and_load_entry(tmp_path):
    path = tmp_path / "cache.json"
    clusters = [[0, 1], [2, 3]]
    entry = save_entry(path, [1, 0, 3, 2], "hash", clusters, meta={"clusters": 2})
    assert path.exists()
    loaded = load_entry(path)
    assert loaded is not None
    assert loaded.signature == entry.signature
    assert not should_invalidate(loaded, clusters)
    assert should_invalidate(loaded, [[0, 2], [1, 3]])

