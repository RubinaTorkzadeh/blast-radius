import ast
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path("/tmp/sqlglot_tmp")
TESTS = json.loads(Path("sample/tests.json").read_text(encoding="utf-8"))["tests"]


def compute_dialect_descendants(known_dialects: set[str]) -> dict[str, set[str]]:
    # Map dialect stem -> direct parent dialect stems by parsing sqlglot/dialects/*.py
    parents_of: dict[str, set[str]] = defaultdict(set)
    for stem in known_dialects:
        p = REPO / "sqlglot" / "dialects" / f"{stem}.py"
        if not p.exists():
            continue
        tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        imported_dialects: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                # e.g. from sqlglot.dialects.hive import Hive
                parts = node.module.split(".")
                if len(parts) >= 3 and parts[-2] == "dialects" and parts[-1] in known_dialects:
                    for alias in node.names:
                        imported_dialects[alias.asname or alias.name] = parts[-1]
                elif node.module == "sqlglot.dialects":
                    for alias in node.names:
                        if alias.name in known_dialects:
                            imported_dialects[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = None
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                        if base.value.id in imported_dialects:
                            parents_of[stem].add(imported_dialects[base.value.id])
                    if base_name and base_name in imported_dialects:
                        parents_of[stem].add(imported_dialects[base_name])

    children_of: dict[str, set[str]] = defaultdict(set)
    for child, pars in parents_of.items():
        for par in pars:
            children_of[par].add(child)

    descendants: dict[str, set[str]] = {}
    for d in known_dialects:
        seen = {d}
        stack = [d]
        while stack:
            cur = stack.pop()
            for ch in children_of.get(cur, set()):
                if ch not in seen:
                    seen.add(ch)
                    stack.append(ch)
        descendants[d] = seen
    return descendants


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
    dialect_aware_opt_tests: set[int] = set()
    per_dialect_opt_tests: dict[str, set[int]] = defaultdict(set)
    per_dialect_suite_tests: dict[str, set[int]] = defaultdict(set)

    known_dialects = {
        p.stem for p in (REPO / "sqlglot" / "dialects").glob("*.py")
        if p.stem not in {"__init__", "dialect"}
    }
    descendants = compute_dialect_descendants(known_dialects)

    dialect_token_re = re.compile(
        r"\b(" + "|".join(sorted(known_dialects)) + r")\b",
        re.I,
    )

    for tfile, tlist in tests_by_file.items():
        full_path = REPO / tfile
        if not full_path.exists():
            continue
        tree = ast.parse(full_path.read_text(encoding="utf-8", errors="replace"))
        methods_by_name = {t["id"].split("::")[-1]: t["number"] for t in tlist}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in methods_by_name:
                tnum = methods_by_name[node.name]
                uses_generic_dialect = False
                mentioned_dialects: set[str] = set()
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        test_method_strings[tnum].add(sub.value)
                        low = sub.value.lower()
                        if low in known_dialects:
                            mentioned_dialects.add(low)
                        else:
                            for m in dialect_token_re.finditer(low):
                                mentioned_dialects.add(m.group(1).lower())
                    elif isinstance(sub, ast.keyword) and sub.arg in {"dialect", "read", "write"}:
                        uses_generic_dialect = True

                if tfile == "tests/test_optimizer.py":
                    if mentioned_dialects:
                        dialect_aware_opt_tests.add(tnum)
                        for d in mentioned_dialects:
                            per_dialect_opt_tests[d].add(tnum)
                    elif uses_generic_dialect:
                        dialect_aware_opt_tests.add(tnum)
                        for d in known_dialects:
                            per_dialect_opt_tests[d].add(tnum)
                elif tfile.startswith("tests/dialects/"):
                    stem_name = Path(tfile).stem
                    if stem_name.startswith("test_") and stem_name[5:] in known_dialects:
                        per_dialect_suite_tests[stem_name[5:]].add(tnum)
                    if mentioned_dialects:
                        for d in mentioned_dialects:
                            per_dialect_suite_tests[d].add(tnum)
                    elif tfile == "tests/dialects/test_dialect.py" and node.name == "test_dialect":
                        for d in known_dialects:
                            per_dialect_suite_tests[d].add(tnum)

    # Propagate descendant dialect suites (e.g., hive -> spark -> databricks)
    expanded_per_dialect_suite: dict[str, set[int]] = defaultdict(set)
    for d, subs in descendants.items():
        for sub_d in subs:
            expanded_per_dialect_suite[d].update(per_dialect_suite_tests.get(sub_d, set()))

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
            txt = (REPO / fix).read_text(encoding="utf-8", errors="replace")
            found_in_sql = {m.group(1).lower() for m in dialect_token_re.finditer(txt)}
            if found_in_sql:
                dialect_aware_opt_tests.update(fixture_to_tests[fix])
                for d in found_in_sql:
                    per_dialect_opt_tests[d].update(fixture_to_tests[fix])
        else:
            for tnum, strs in test_method_strings.items():
                if any(fix in s or p.name in s or stem == s for s in strs):
                    fixture_to_tests[fix].add(tnum)

    payload = {
        "fixtures": {k: sorted(v) for k, v in fixture_to_tests.items() if v},
        "dialect_aware_opt": sorted(dialect_aware_opt_tests),
        "per_dialect_opt": {k: sorted(v) for k, v in per_dialect_opt_tests.items() if v},
        "per_dialect_suite": {k: sorted(v) for k, v in expanded_per_dialect_suite.items() if v},
        "dialect_descendants": {k: sorted(v) for k, v in descendants.items()},
    }
    with gzip.open("fixture_map.json.gz", "wt", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(
        f"Mapped {len(payload['fixtures'])} fixtures, "
        f"{len(dialect_aware_opt_tests)} dialect-aware opt tests, "
        f"descendants sample: hive->{sorted(descendants.get('hive', []))}"
    )


if __name__ == "__main__":
    main()
