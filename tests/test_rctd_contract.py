from pathlib import Path
from types import SimpleNamespace

from segbench import evaluate


def test_rctd_command_freezes_sparse_profile_cutoffs(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="expected test stop")

    monkeypatch.setattr(evaluate.subprocess, "run", fake_run)
    result = evaluate.run_rctd(
        cell_h5ad=Path("spatial.h5ad"),
        reference_h5ad=Path("reference.h5ad"),
        celltype_col="lineage",
        outdir=tmp_path,
        rscript="Rscript",
        exclude_celltypes=[],
        reference_min_umi=10,
    )

    command = (tmp_path / "rctd_cmd.txt").read_text()
    assert "--umi-min 10" in command
    assert "--umi-min-sigma 10" in command
    assert "--reference-min-umi 10" in command
    assert result["rctd_status"] == "failed(rc=1)"
