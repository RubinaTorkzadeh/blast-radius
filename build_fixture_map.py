import ast
import gzip
import json
from collections import defaultdict
from pathlib import Path

REPO = Path("/tmp/sqlglot_tmp")
TESTS = json.loads(Path("sample/tests.json").read_text(encoding="utf-8"))["tests"]


def main() -> None:
    tests_by_file: dict[str, list[dict]] = defaultdict(list)
    test_by_stem: dict[str, list[int]] = defaultdict(list)
    for t in TESTS:
        tests_by_file[t["file"]].append(t)
        method_name = t["id"].split("::")[-1]
        if method_name.startswith("test_"):
            test_by_stem[method_name[5:]].append(t["number"])

    fixture_to_tests: dict[str, set[int]] = defaultdict(set)

    all_fixtures = [
        p.as_posix()[len("/tmp/sqlglot_tmp/"):]
        for p in (REPO / "tests" / "fixtures").rglob("*")
        if p.is_file()
    ]

    test_method_strings: dict[int, set[str]] = defaultdict(set)
    for tfile, tlist in tests_by_file.items():
        full_path = REPO / tfile
        if not full_path.exists():
            continue
        tree = ast.parse(full_path.read_text(encoding="utf-8", errors="replace"))
        methods_by_name = {t["id"].split("::")[-1]: t["number"] for t in tlist}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in methods_by_name:
                tnum = methods_by_name[node.name]
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        test_method_strings[tnum].add(sub.value)

    opt_nums = {t["number"] for t in tests_by_file.get("tests/test_optimizer.py", [])}

    for fix in all_fixtures:
        p = Path(fix)
        stem = p.stem
        if fix.startswith("tests/fixtures/optimizer/"):
            for num in test_by_stem.get(stem, []):
                if num in opt_nums:
                    fixture_to_tests[fix].add(num)
            for num in opt_nums:
                if any(
                    stem == s
                    or f"/{stem}." in s
                    or s.endswith(f"/{stem}")
                    or (len(stem) >= 6 and stem in s)
                    for s in test_method_strings.get(num, set())
                ):
                    fixture_to_tests[fix].add(num)
        else:
            for tnum, strs in test_method_strings.items():
                if any(fix in s or p.name in s or stem == s for s in strs):
                    fixture_to_tests[fix].add(tnum)

    serializable = {k: sorted(v) for k, v in fixture_to_tests.items() if v}
    with gzip.open("fixture_map.json.gz", "wt", encoding="utf-8") as f:
        json.dump(serializable, f, separators=(",", ":"))
    print(f"Mapped {len(serializable)} fixtures in fixture_map.json.gz")


if __name__ == "__main__":
    main()
