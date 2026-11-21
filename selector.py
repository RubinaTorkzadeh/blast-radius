import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

DATA = Path("/data")
COV_PATH = Path("/app/coverage_map.json.gz")
AST_PATH = Path("/app/ast_map.json.gz")
DIALECT_PATH = Path("/app/dialect_map.json.gz")
MAX_WORKERS = 8
WORKER_COST = 2.0

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def normalize_code_lines(lines: list[str]) -> list[str]:
    """Strip blank lines and pure # comment lines to detect comment/whitespace-only diffs."""
    out: list[str] = []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if "#" in s and '"' not in s and "'" not in s:
            s = s.split("#", 1)[0].rstrip()
        if s:
            out.append(s)
    return out


def parse_patch(patch_text: str) -> dict[str, set[int]]:
    """Return {file_path: set_of_changed_line_numbers} for files with substantive changes."""
    file_lines: dict[str, set[int]] = defaultdict(set)
    added_raw: dict[str, list[str]] = defaultdict(list)
    removed_raw: dict[str, list[str]] = defaultdict(list)

    cur_file: str | None = None
    old_line = 0
    new_line = 0
    in_hunk = False

    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            in_hunk = False
            parts = line.split()
            if len(parts) >= 4 and parts[3].startswith("b/"):
                cur_file = parts[3][2:]
            else:
                cur_file = None
        elif line.startswith("+++ "):
            target = line[4:].strip()
            if target != "/dev/null":
                cur_file = target[2:] if target.startswith("b/") else target
        elif line.startswith("--- "):
            target = line[4:].strip()
            if cur_file is None and target != "/dev/null":
                cur_file = target[2:] if target.startswith("a/") else target
        elif line.startswith("@@ ") and cur_file is not None:
            m = HUNK_RE.match(line)
            if m:
                old_line = int(m.group(1))
                new_line = int(m.group(3))
                in_hunk = True
                file_lines[cur_file].add(max(1, old_line))
                file_lines[cur_file].add(max(1, new_line))
        elif in_hunk and cur_file is not None:
            if line.startswith("+") and not line.startswith("+++"):
                file_lines[cur_file].add(max(1, new_line))
                added_raw[cur_file].append(line[1:])
                new_line += 1
            elif line.startswith("-") and not line.startswith("---"):
                file_lines[cur_file].add(max(1, old_line))
                removed_raw[cur_file].append(line[1:])
                old_line += 1
            else:
                old_line += 1
                new_line += 1

    filtered: dict[str, set[int]] = {}
    for fpath, lns in file_lines.items():
        if fpath.endswith(".py"):
            if normalize_code_lines(added_raw[fpath]) == normalize_code_lines(removed_raw[fpath]):
                continue
        filtered[fpath] = lns
    return filtered


def split_lpt(tests: list[dict], worker_count: int) -> tuple[float, list[list[int]]]:
    workers: list[list[int]] = [[] for _ in range(worker_count)]
    loads = [0.0] * worker_count
    for test in sorted(tests, key=lambda t: t["seconds"], reverse=True):
        lightest = loads.index(min(loads))
        workers[lightest].append(test["number"])
        loads[lightest] += test["seconds"]
    total_cost = max(loads) + WORKER_COST * worker_count
    return total_cost, workers


def choose_best_workers(selected_tests: list[dict]) -> list[list[int]]:
    if not selected_tests:
        return []
    max_k = min(MAX_WORKERS, len(selected_tests))
    best_cost = float("inf")
    best_workers: list[list[int]] = []
    for k in range(1, max_k + 1):
        cost, workers = split_lpt(selected_tests, k)
        if cost < best_cost:
            best_cost = cost
            best_workers = workers
    return [w for w in best_workers if w]


def is_doc_only(path: str) -> bool:
    p = path.lower()
    return (
        p.endswith((".md", ".rst", ".txt"))
        or p.startswith(("docs/", ".github/"))
        or p in {"license", "changelog.md", "readme.md"}
    )


def module_name_for_path(fpath: str) -> str:
    if fpath.endswith("/__init__.py"):
        return fpath[:-12].replace("/", ".")
    if fpath.endswith(".py"):
        return fpath[:-3].replace("/", ".")
    return fpath


def main() -> None:
    tests_raw = json.loads((DATA / "tests.json").read_text(encoding="utf-8"))["tests"]
    by_num = {t["number"]: t for t in tests_raw}
    all_nums = set(by_num.keys())

    tests_by_file: dict[str, set[int]] = defaultdict(set)
    for t in tests_raw:
        tests_by_file[t["file"]].add(t["number"])

    with gzip.open(COV_PATH, "rt", encoding="utf-8") as f:
        raw_cov: dict[str, dict[str, list[int]]] = json.load(f)

    cov_map: dict[str, dict[str, list[int]]] = {}
    for k, v in raw_cov.items():
        cov_map[k] = v
        if k.startswith("sqlglot/sqlglot/"):
            cov_map[k[len("sqlglot/"):]] = v

    with gzip.open(AST_PATH, "rt", encoding="utf-8") as f:
        ast_data = json.load(f)
    func_ranges: dict[str, list[list[int]]] = ast_data["func_ranges"]
    imports_map: dict[str, list[str]] = ast_data["imports"]

    with gzip.open(DIALECT_PATH, "rt", encoding="utf-8") as f:
        dialect_map: dict[str, list[int]] = json.load(f)

    direct_importers: dict[str, set[int]] = defaultdict(set)
    for src_file, imported_mods in imports_map.items():
        if src_file in tests_by_file:
            for mod in imported_mods:
                direct_importers[mod].update(tests_by_file[src_file])

    file_all_tests: dict[str, set[int]] = {}
    for fpath, line_dict in cov_map.items():
        s: set[int] = set()
        for tlist in line_dict.values():
            s.update(tlist)
        file_all_tests[fpath] = s

    fast_cross_dialect_suites: set[int] = set()
    for tf in [
        "tests/test_transpile.py",
        "tests/test_optimizer.py",
        "tests/test_parser.py",
        "tests/test_expressions.py",
    ]:
        fast_cross_dialect_suites.update(tests_by_file.get(tf, set()))

    dialect_runner_suite: set[int] = set()
    for tf in ["tests/dialects/test_dialect.py", "tests/test_transpile.py"]:
        dialect_runner_suite.update(tests_by_file.get(tf, set()))

    optimizer_suites: set[int] = set()
    for tf in [
        "tests/test_optimizer.py",
        "tests/test_executor.py",
        "tests/test_transpile.py",
        "tests/test_lineage.py",
    ]:
        optimizer_suites.update(tests_by_file.get(tf, set()))

    for patch_path in sorted((DATA / "commits").glob("*.patch")):
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
        changed = parse_patch(patch_text)

        selected_nums: set[int] = set()
        for fpath, lines in changed.items():
            if is_doc_only(fpath):
                continue
            elif fpath in tests_by_file:
                selected_nums.update(tests_by_file[fpath])
            elif fpath.startswith("tests/fixtures/optimizer/"):
                selected_nums.update(optimizer_suites)
            elif fpath.startswith("tests/fixtures/"):
                selected_nums.update(dialect_runner_suite)
                selected_nums.update(fast_cross_dialect_suites)
                stem_tokens = set(re.split(r"[^a-z0-9]+", fpath.lower())) - {"tests", "fixtures", "sql", "json", ""}
                for tfile, tnums in tests_by_file.items():
                    if any(tok in tfile.lower() for tok in stem_tokens if len(tok) >= 3):
                        selected_nums.update(tnums)
            elif fpath in cov_map:
                f_tests = file_all_tests.get(fpath, set())
                stem = Path(fpath).stem
                mod = module_name_for_path(fpath)
                is_dialect_family = (
                    fpath.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/"))
                    and stem in dialect_map
                )
                if is_dialect_family:
                    selected_nums.update(f_tests)
                    selected_nums.update(dialect_map[stem])
                    selected_nums.update(fast_cross_dialect_suites)
                    selected_nums.update(direct_importers.get(mod, set()))
                elif fpath.startswith("sqlglot/optimizer/"):
                    selected_nums.update(f_tests)
                    selected_nums.update(optimizer_suites)
                    selected_nums.update(direct_importers.get(mod, set()))
                elif len(f_tests) <= 350:
                    selected_nums.update(f_tests)
                    selected_nums.update(direct_importers.get(mod, set()))
                else:
                    line_dict = cov_map[fpath]
                    spans = func_ranges.get(fpath, [])
                    expanded_lines: set[int] = set()
                    module_level_touched = False

                    for ln in lines:
                        enclosing = [
                            (s, e) for (s, e) in spans
                            if (s - 2) <= ln <= (e + 2)
                        ]
                        if enclosing:
                            min_len = min(e - s for (s, e) in enclosing)
                            for (s, e) in enclosing:
                                if (e - s) <= min_len + 5:
                                    expanded_lines.update(range(s, e + 1))
                        else:
                            module_level_touched = True

                    if module_level_touched:
                        selected_nums.update(f_tests)
                        selected_nums.update(direct_importers.get(mod, set()))
                    else:
                        matched_any = False
                        for eln in expanded_lines:
                            tlist = line_dict.get(str(eln))
                            if tlist:
                                selected_nums.update(tlist)
                                matched_any = True
                        if not matched_any:
                            selected_nums.update(f_tests)
            else:
                selected_nums.update(all_nums)

        selected_tests = [by_num[n] for n in selected_nums]
        workers = choose_best_workers(selected_tests)
        print(json.dumps({"commit": patch_path.stem, "workers": workers}))


if __name__ == "__main__":
    main()
