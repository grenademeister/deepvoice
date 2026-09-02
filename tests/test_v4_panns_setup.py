def test_panns_prepares_runtime_label_file(tmp_path, monkeypatch):
    from models.panns import prepare_labels
    source = tmp_path / 'model' / 'class_labels_indices.csv'
    source.parent.mkdir(parents=True); source.write_text('index,display_name\n0,Music\n')
    home = tmp_path / 'home'; home.mkdir(); monkeypatch.setattr('pathlib.Path.home', lambda: home)
    target = prepare_labels(source.parent)
    assert target == home / 'panns_data' / 'class_labels_indices.csv'
    assert target.read_text() == source.read_text()
