import json
from pathlib import Path
from typing import Any, cast


def test_burgers_notebook_is_clean_and_compilable() -> None:
    """Verify the example notebook has clean outputs and valid Python cells."""
    notebook_path = Path(__file__).parents[2] / "notebooks" / "burgers_example.ipynb"
    notebook = cast(dict[str, Any], json.loads(notebook_path.read_text(encoding="utf-8")))

    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell["execution_count"] is None and not cell["outputs"] for cell in code_cells)
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"{notebook_path.name}:cell-{index}", "exec")

    notebook_text = notebook_path.read_text(encoding="utf-8")
    notebook_source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert all(command in notebook_source for command in ("phijax-train", "phijax-predict", "phijax-evaluate"))
    assert "colab.research.google.com" in notebook_source
    assert "~callbacks.rich_progress_bar" not in notebook_source
    assert '"logger=console"' not in notebook_source
    assert "model.net.hidden=" not in notebook_source
    assert "trainer.max_steps=" not in notebook_source
    assert '"callbacks.model_checkpoint.every_n_steps=null"' in notebook_source
    assert "--xla_gpu_exclude_nondeterministic_ops --xla_gpu_autotune_level=0" in notebook_source
    assert "~callbacks.lr_monitor" not in notebook_source
    assert "subprocess.Popen" in notebook_source
    assert "PROJECT_ROOT" not in notebook_text
    assert "hydra.run.dir" not in notebook_text
