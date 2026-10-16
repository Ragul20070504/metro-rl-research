"""Forecasting exports must show actual held-out metrics, not tuning settings."""
import json

import matplotlib.pyplot as plt
import numpy as np

from scripts import make_paper_assets as assets


def test_forecast_plot_and_table_read_nested_model_split_metrics(monkeypatch):
    fc = json.loads((assets.RESULTS_DIR / "forecasting_metrics.json").read_text())
    captured = {}

    def save(fig, stem, sources, detail):
        captured["counts"] = [len(ax.patches) for ax in fig.axes]
        captured["detail"] = detail
        plt.close(fig)

    monkeypatch.setattr(assets, "save", save)
    monkeypatch.setattr(assets, "save_table", lambda frame, *args: captured.update(table=frame))
    assets.figure03_forecasting()
    assets.forecasting_table(fc)
    assert captured["counts"] == [3, 3]
    for model, splits in fc["metrics"].items():
        assert captured["detail"]["test_rmse"][model] == splits["test"]["point"]["rmse"]
    assert np.allclose(captured["detail"]["coverage"], [
        fc["calibration"]["coverage_before"], fc["calibration"]["coverage_after"],
        fc["metrics"]["xgb_quantile"]["test"]["probabilistic"]["picp_80"]])
    table = captured["table"]
    assert len(table) == 9 and set(table.split) == {"train", "val", "test"}
    assert table.loc[table.model == "xgb_quantile", "pinball_mean"].notna().all()
    assert "n_candidates" not in table.columns


def test_missing_binaries_do_not_turn_the_recorded_study_size_negative():
    from scripts.build_manuscript import Study, manuscript, supplementary

    study = Study()
    study.policies = 0
    text = manuscript(study)
    assert "**Main-study learned-policy identities in stored cells:** 50; available original binaries: 0" in text
    assert "**Trained confirmatory policies:** -8" not in text
    assert "The code is committed now" not in text
    supplement = supplementary(study)
    assert "scripts/analyse.py" not in supplement
    assert "rolling empirical pool" in text or "rolling pool" in text
    assert "not an exact permutation" in text


def test_document_exports_keep_images_and_native_math(tmp_path, monkeypatch):
    import shutil
    from zipfile import ZipFile
    import pytest
    from scripts import build_manuscript as builder

    if not shutil.which('pandoc') or not shutil.which('xelatex'):
        pytest.skip('Document export requires Pandoc and XeLaTeX')
    plt.imsave(tmp_path / 'figure.png', np.ones((2, 2, 3)))
    monkeypatch.setattr(builder, 'DOCS', tmp_path)
    builder.export_documents('# Export check\n\n$x^2$\n\n![Figure](figure.png)\n')
    assert (tmp_path / 'paper_final.pdf').read_bytes().startswith(b'%PDF-')
    assert r'\includegraphics' in (tmp_path / 'paper_final.tex').read_text()
    with ZipFile(tmp_path / 'paper_final.docx') as document:
        assert document.testzip() is None
        assert any(name.startswith('word/media/') for name in document.namelist())
        assert b'<m:oMath' in document.read('word/document.xml')
