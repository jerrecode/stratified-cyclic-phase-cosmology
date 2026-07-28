import hashlib

from scpc.numerics.provenance import build_output_inventory, sha256_file


def test_sha256_file_matches_known_digest(tmp_path) -> None:
    output = tmp_path / "result.dat"
    output.write_bytes(b"SCPC verification output\n")
    expected = hashlib.sha256(output.read_bytes()).hexdigest()
    assert sha256_file(output) == expected


def test_output_inventory_records_relative_paths_sizes_and_hashes(tmp_path) -> None:
    first = tmp_path / "b.txt"
    second = tmp_path / "a.txt"
    first.write_text("beta", encoding="utf-8")
    second.write_text("alpha", encoding="utf-8")

    inventory = build_output_inventory([first, second], relative_to=tmp_path)

    assert [item["path"] for item in inventory] == ["a.txt", "b.txt"]
    assert inventory[0]["size_bytes"] == len("alpha".encode())
    assert inventory[0]["sha256"] == sha256_file(second)
