from pathlib import Path

from scpc.scans.fingerprint import implementation_runtime_fingerprint, source_tree_sha256


def _source_tree(root: Path, content: str = "VALUE = 1\n") -> Path:
    package = root / "scpc"
    (package / "nested").mkdir(parents=True)
    (package / "__init__.py").write_text(content, encoding="utf-8")
    (package / "nested" / "module.py").write_text("OTHER = 2\n", encoding="utf-8")
    return package


def test_source_hash_is_path_order_independent_and_content_sensitive(tmp_path) -> None:
    first = _source_tree(tmp_path / "first")
    second = _source_tree(tmp_path / "second")
    assert source_tree_sha256(first) == source_tree_sha256(second)

    (second / "nested" / "module.py").write_text("OTHER = 3\n", encoding="utf-8")
    assert source_tree_sha256(first) != source_tree_sha256(second)


def test_runtime_fingerprint_is_deterministic_for_explicit_environment(tmp_path) -> None:
    package = _source_tree(tmp_path)
    versions = {"numpy": "2.0", "scipy": "1.13"}
    first = implementation_runtime_fingerprint(
        package_root=package,
        package_versions=versions,
        python_version="3.12.4",
        python_implementation="CPython",
        platform_string="Linux-test",
        machine="x86_64",
    )
    second = implementation_runtime_fingerprint(
        package_root=package,
        package_versions=dict(reversed(list(versions.items()))),
        python_version="3.12.4",
        python_implementation="CPython",
        platform_string="Linux-test",
        machine="x86_64",
    )
    assert first["strict_sha256"] == second["strict_sha256"]
    assert first["source_tree_sha256"] == source_tree_sha256(package)


def test_source_dependency_and_platform_changes_invalidate_fingerprint(tmp_path) -> None:
    package = _source_tree(tmp_path)
    baseline = implementation_runtime_fingerprint(
        package_root=package,
        package_versions={"numpy": "2.0"},
        python_version="3.12.4",
        python_implementation="CPython",
        platform_string="Linux-A",
        machine="x86_64",
    )
    changed_dependency = implementation_runtime_fingerprint(
        package_root=package,
        package_versions={"numpy": "2.1"},
        python_version="3.12.4",
        python_implementation="CPython",
        platform_string="Linux-A",
        machine="x86_64",
    )
    changed_platform = implementation_runtime_fingerprint(
        package_root=package,
        package_versions={"numpy": "2.0"},
        python_version="3.12.4",
        python_implementation="CPython",
        platform_string="Linux-B",
        machine="aarch64",
    )
    (package / "__init__.py").write_text("VALUE = 9\n", encoding="utf-8")
    changed_source = implementation_runtime_fingerprint(
        package_root=package,
        package_versions={"numpy": "2.0"},
        python_version="3.12.4",
        python_implementation="CPython",
        platform_string="Linux-A",
        machine="x86_64",
    )

    assert baseline["strict_sha256"] != changed_dependency["strict_sha256"]
    assert baseline["strict_sha256"] != changed_platform["strict_sha256"]
    assert baseline["strict_sha256"] != changed_source["strict_sha256"]
