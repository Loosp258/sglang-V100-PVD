"""Keep the native V launcher ahead of SGLang's transitive torch imports."""

import ast
import tomllib
from pathlib import Path


def test_native_launcher_imports_cuvs_before_sglang():
    source = Path(__file__).resolve().parents[3] / "python/pvd_cagra_server.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    imports = [
        ("import", alias.name)
        for node in module.body
        if isinstance(node, ast.Import)
        for alias in node.names
    ] + [
        ("from", node.module)
        for node in module.body
        if isinstance(node, ast.ImportFrom)
    ]
    assert ("import", "cuvs.neighbors.cagra") in imports
    assert ("from", "sglang.srt.disaggregation.pvd.server") in imports
    native_line = next(
        node.lineno
        for node in module.body
        if isinstance(node, ast.Import)
        and any(alias.name == "cuvs.neighbors.cagra" for alias in node.names)
    )
    sglang_line = next(
        node.lineno
        for node in module.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "sglang.srt.disaggregation.pvd.server"
    )
    assert native_line < sglang_line
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and node.lineno < native_line
        and (
            any(alias.name.startswith("sglang") for alias in node.names)
            if isinstance(node, ast.Import)
            else (node.module or "").startswith("sglang")
        )
        for node in module.body
    )


def test_native_launcher_is_in_wheel_and_exposes_console_script():
    pyproject = Path(__file__).resolve().parents[3] / "python/pyproject.toml"
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert config["tool"]["setuptools"]["py-modules"] == ["pvd_cagra_server"]
    assert config["project"]["scripts"]["pvd-cagra-server"] == (
        "pvd_cagra_server:main"
    )
