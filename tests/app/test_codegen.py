"""Tests for _codegen_base, _memthick_codegen, and _pana_codegen (T8)."""
from __future__ import annotations

import ast
import json

import pytest

from cryocat.app.suite.pages._codegen_base import (
    Verbatim,
    format_value,
    format_dict,
    format_kwargs,
    render_slurm_wrapper,
    parse_sbatch_text,
)
from cryocat.app.suite.pages._memthick_codegen import (
    render_pipeline_py,
    render_pipeline_ipynb,
    render_pipeline_ipynb_json,
)
from cryocat.app.suite.pages._pana_codegen import render_analysis_py


# ── format_value ─────────────────────────────────────────────────────────────

class TestFormatValue:
    def test_str(self):
        assert format_value("hello") == repr("hello")

    def test_int(self):
        assert format_value(42) == "42"

    def test_float(self):
        assert format_value(3.14) == repr(3.14)

    def test_bool_true(self):
        assert format_value(True) == "True"

    def test_bool_false(self):
        assert format_value(False) == "False"

    def test_none(self):
        assert format_value(None) == "None"

    def test_tuple(self):
        assert format_value((1, 2)) == "(1, 2)"

    def test_dict_delegates_to_format_dict(self):
        result = format_value({"a": 1})
        assert result == format_dict({"a": 1})

    def test_verbatim_emits_bare_identifier(self):
        assert format_value(Verbatim("my_var")) == "my_var"

    def test_verbatim_no_quotes(self):
        result = format_value(Verbatim("segmentation_path"))
        assert '"' not in result and "'" not in result


# ── render_pipeline_py ────────────────────────────────────────────────────────

class TestRenderPipelinePy:
    _MINIMAL = {"segmentation_map": "/x/seg.mrc"}

    def test_is_valid_python(self):
        ast.parse(render_pipeline_py(self._MINIMAL))

    def test_imports_memthick(self):
        py = render_pipeline_py(self._MINIMAL)
        assert "from cryocat.analysis import memthick" in py

    def test_path_param_hoisted(self):
        py = render_pipeline_py(self._MINIMAL)
        assert "segmentation_path = '/x/seg.mrc'" in py
        assert "segmentation_map=segmentation_path" in py

    def test_analyzer_included_when_set(self):
        py = render_pipeline_py({
            **self._MINIMAL,
            "analyzer": {"smooth_sigma_intensity_profiles": 0.7},
        })
        ast.parse(py)
        assert "IntensityProfileAnalyzer" in py
        assert "analyzer=analyzer" in py

    def test_analyzer_omitted_when_absent(self):
        py = render_pipeline_py(self._MINIMAL)
        assert "IntensityProfileAnalyzer" not in py
        assert "analyzer" not in py

    def test_membrane_labels_rendered_as_dict(self):
        py = render_pipeline_py({
            **self._MINIMAL,
            "membrane_labels": {"outer": 1, "inner": 2},
        })
        ast.parse(py)
        assert "'outer': 1" in py
        assert "'inner': 2" in py


# ── render_analysis_py ────────────────────────────────────────────────────────

class TestRenderAnalysisPy:
    _MINIMAL = {"template_list": "/data/templates.csv"}

    def test_is_valid_python(self):
        ast.parse(render_analysis_py(self._MINIMAL))

    def test_imports_pana(self):
        py = render_analysis_py(self._MINIMAL)
        assert "from cryocat.analysis import pana" in py

    def test_path_param_hoisted(self):
        py = render_analysis_py(self._MINIMAL)
        assert "template_list = '/data/templates.csv'" in py
        assert "template_list=template_list" in py

    def test_scalar_kwarg_inlined(self):
        py = render_analysis_py({**self._MINIMAL, "n_workers": 4})
        ast.parse(py)
        assert "n_workers=4" in py

    def test_none_values_excluded(self):
        py = render_analysis_py({**self._MINIMAL, "wedge_path": None})
        ast.parse(py)
        assert "wedge_path" not in py


# ── render_pipeline_ipynb ─────────────────────────────────────────────────────

class TestRenderPipelineIpynb:
    _KWARGS = {
        "segmentation_map": "/x/seg.mrc",
        "analyzer": {"smooth_sigma_intensity_profiles": 0.7},
    }

    def test_nbformat_4(self):
        nb = render_pipeline_ipynb(self._KWARGS)
        assert nb["nbformat"] == 4

    def test_first_cell_is_markdown(self):
        nb = render_pipeline_ipynb(self._KWARGS)
        assert nb["cells"][0]["cell_type"] == "markdown"

    def test_at_least_three_code_cells(self):
        nb = render_pipeline_ipynb(self._KWARGS)
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) >= 3

    def test_json_serialisable(self):
        nb = render_pipeline_ipynb(self._KWARGS)
        text = json.dumps(nb)
        parsed = json.loads(text)
        assert parsed["nbformat"] == 4

    def test_ipynb_json_round_trip(self):
        text = render_pipeline_ipynb_json(self._KWARGS)
        parsed = json.loads(text)
        assert parsed["nbformat"] == 4
        assert any(c["cell_type"] == "markdown" for c in parsed["cells"])


# ── render_slurm_wrapper ──────────────────────────────────────────────────────

class TestRenderSlurmWrapper:
    def test_shebang_first(self):
        out = render_slurm_wrapper("/scratch/run.py")
        assert out.startswith("#!/bin/bash\n")

    def test_python_line_last(self):
        out = render_slurm_wrapper("/scratch/run.py")
        assert out.rstrip().endswith("python /scratch/run.py")

    def test_sbatch_lines_before_module_load(self):
        out = render_slurm_wrapper(
            "/scratch/run.py",
            cluster_params={"--mem": "32G"},
            module_loads=["cryocat/1.0"],
        )
        sbatch_idx = out.index("#SBATCH")
        module_idx = out.index("module load")
        python_idx = out.index("python /scratch/run.py")
        assert sbatch_idx < module_idx < python_idx

    def test_sbatch_long_flag(self):
        out = render_slurm_wrapper("/run.py", cluster_params={"--mem": "32G"})
        assert "#SBATCH --mem=32G" in out

    def test_sbatch_short_flag(self):
        out = render_slurm_wrapper("/run.py", cluster_params={"-N": 1})
        assert "#SBATCH -N 1" in out

    def test_module_load_lines(self):
        out = render_slurm_wrapper("/run.py", module_loads=["cryocat/1.0", "cuda/12.1"])
        assert "module load cryocat/1.0" in out
        assert "module load cuda/12.1" in out

    def test_no_params_minimal_output(self):
        out = render_slurm_wrapper("/run.py")
        assert out.splitlines() == ["#!/bin/bash", "python /run.py"]


# ── parse_sbatch_text ─────────────────────────────────────────────────────────

class TestParseSbatchText:
    def test_long_form_key_value(self):
        result = parse_sbatch_text("--mem=32G")
        assert result == {"--mem": "32G"}

    def test_short_form_key_value(self):
        result = parse_sbatch_text("-N 4")
        assert result == {"-N": "4"}

    def test_skips_blank_lines(self):
        result = parse_sbatch_text("--mem=32G\n\n-N 4")
        assert len(result) == 2

    def test_skips_comment_lines(self):
        result = parse_sbatch_text("# comment\n--mem=32G")
        assert result == {"--mem": "32G"}

    def test_empty_string(self):
        assert parse_sbatch_text("") == {}

    def test_none_input(self):
        assert parse_sbatch_text(None) == {}

    def test_multiple_directives(self):
        text = "--time=01:00:00\n--mem=16G\n-N 2"
        result = parse_sbatch_text(text)
        assert result["--time"] == "01:00:00"
        assert result["--mem"] == "16G"
        assert result["-N"] == "2"
