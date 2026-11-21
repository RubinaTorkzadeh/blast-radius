import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

DATA = Path("/data")
COV_PATH = Path("/app/coverage_map.json.gz")
AST_PATH = Path("/app/ast_map.json.gz")
DIALECT_PATH = Path("/app/dialect_map.json.gz")
FIXTURE_PATH = Path("/app/fixture_map.json.gz")
MAX_WORKERS = 8
WORKER_COST = 2.0

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

STRESS_TEST_IDS = {
    "tests/test_parser.py::TestParser::test_max_nodes",
    "tests/test_generator.py::TestGenerator::test_generator_memory",
}

HEAVY_OPTIMIZER_ONLY_IDS = {
    "tests/test_optimizer.py::TestOptimizer::test_scope",
    "tests/test_optimizer.py::TestOptimizer::test_tpcds",
    "tests/test_optimizer.py::TestOptimizer::test_optimize_in_nulls",
    "tests/test_optimizer.py::TestOptimizer::test_merge_subqueries",
    "tests/test_optimizer.py::TestOptimizer::test_canonicalize",
    "tests/test_optimizer.py::TestOptimizer::test_qualify_struct_star_long_cte_chain",
}


def normalize_code_lines(lines: list[str]) -> list[str]:
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
    workers: list[list[dict]] = [[] for _ in range(worker_count)]
    loads = [0.0] * worker_count
    for test in sorted(tests, key=lambda t: t["seconds"], reverse=True):
        lightest = loads.index(min(loads))
        workers[lightest].append(test)
        loads[lightest] += test["seconds"]

    # Local-search refinement: move or swap items off the heaviest worker
    if worker_count > 1:
        improved = True
        while improved:
            improved = False
            heavy_idx = loads.index(max(loads))
            heavy_load = loads[heavy_idx]
            for other_idx in range(worker_count):
                if other_idx == heavy_idx:
                    continue
                other_load = loads[other_idx]
                for i, t_a in enumerate(workers[heavy_idx]):
                    sec_a = t_a["seconds"]
                    # Try single move
                    if max(heavy_load - sec_a, other_load + sec_a) < heavy_load - 1e-6:
                        workers[heavy_idx].pop(i)
                        workers[other_idx].append(t_a)
                        loads[heavy_idx] -= sec_a
                        loads[other_idx] += sec_a
                        improved = True
                        break
                    # Try 1-for-1 swap
                    for j, t_b in enumerate(workers[other_idx]):
                        sec_b = t_b["seconds"]
                        new_heavy = heavy_load - sec_a + sec_b
                        new_other = other_load - sec_b + sec_a
                        if max(new_heavy, new_other) < heavy_load - 1e-6:
                            workers[heavy_idx][i] = t_b
                            workers[other_idx][j] = t_a
                            loads[heavy_idx] = new_heavy
                            loads[other_idx] = new_other
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    total_cost = max(loads) + WORKER_COST * worker_count
    return total_cost, [[t["number"] for t in w] for w in workers]


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


def lookup_function_level(
    fpath: str,
    lines: set[int],
    cov_map: dict[str, dict[str, list[int]]],
    func_ranges: dict[str, list[list[int]]],
    file_all_tests: dict[str, set[int]],
    direct_importers: dict[str, set[int]],
    mod: str,
) -> set[int]:
    line_dict = cov_map[fpath]
    spans = func_ranges.get(fpath, [])
    expanded_lines: set[int] = set()
    module_level_touched = False

    for ln in lines:
        enclosing = [(s, e) for (s, e) in spans if (s - 2) <= ln <= (e + 2)]
        if enclosing:
            min_len = min(e - s for (s, e) in enclosing)
            for (s, e) in enclosing:
                if (e - s) <= min_len + 5:
                    expanded_lines.update(range(s, e + 1))
        else:
            module_level_touched = True

    if module_level_touched:
        return file_all_tests.get(fpath, set()) | direct_importers.get(mod, set())

    matched: set[int] = set()
    for eln in expanded_lines:
        tlist = line_dict.get(str(eln))
        if tlist:
            matched.update(tlist)
    return matched if matched else file_all_tests.get(fpath, set())


def main() -> None:
    tests_raw = json.loads((DATA / "tests.json").read_text(encoding="utf-8"))["tests"]
    by_num = {t["number"]: t for t in tests_raw}
    all_nums = set(by_num.keys())
    stress_nums = {t["number"] for t in tests_raw if t["id"] in STRESS_TEST_IDS}
    heavy_opt_only_nums = {t["number"] for t in tests_raw if t["id"] in HEAVY_OPTIMIZER_ONLY_IDS}

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

    with gzip.open(FIXTURE_PATH, "rt", encoding="utf-8") as f:
        fix_payload = json.load(f)
    fixture_map: dict[str, list[int]] = fix_payload["fixtures"]
    dialect_aware_opt: set[int] = set(fix_payload["dialect_aware_opt"])
    per_dialect_opt: dict[str, list[int]] = fix_payload.get("per_dialect_opt", {})

    mod_to_file: dict[str, str] = {}
    for src_file in imports_map:
        mod_to_file[module_name_for_path(src_file)] = src_file

    direct_importers: dict[str, set[int]] = defaultdict(set)
    for src_file, imported_mods in imports_map.items():
        if src_file in tests_by_file:
            tnums = tests_by_file[src_file]
            for mod in imported_mods:
                direct_importers[mod].update(tnums)
                mid_file = mod_to_file.get(mod)
                if mid_file:
                    for sub_mod in imports_map.get(mid_file, []):
                        direct_importers[sub_mod].update(tnums)

    file_all_tests: dict[str, set[int]] = {}
    for fpath, line_dict in cov_map.items():
        s: set[int] = set()
        for tlist in line_dict.values():
            s.update(tlist)
        file_all_tests[fpath] = s

    cross_dialect_suites: set[int] = set(dialect_aware_opt)
    for tf in [
        "tests/dialects/test_dialect.py",
        "tests/test_transpile.py",
    ]:
        cross_dialect_suites.update(tests_by_file.get(tf, set()))
    cross_dialect_suites -= (stress_nums | heavy_opt_only_nums)

    optimizer_suites: set[int] = set()
    for tf in [
        "tests/test_optimizer.py",
        "tests/test_executor.py",
        "tests/test_transpile.py",
        "tests/test_lineage.py",
    ]:
        optimizer_suites.update(tests_by_file.get(tf, set()))
    optimizer_suites -= stress_nums

    for patch_path in sorted((DATA / "commits").glob("*.patch")):
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
        changed = parse_patch(patch_text)

        selected_nums: set[int] = set()
        for fpath, lines in changed.items():
            if is_doc_only(fpath):
                continue
            elif fpath in tests_by_file:
                selected_nums.update(tests_by_file[fpath])
            elif fpath in fixture_map:
                selected_nums.update(fixture_map[fpath])
            elif fpath.startswith("tests/fixtures/optimizer/"):
                selected_nums.update(tests_by_file.get("tests/test_optimizer.py", set()) - stress_nums)
            elif fpath.startswith("tests/fixtures/"):
                selected_nums.update(cross_dialect_suites)
                stem_tokens = set(re.split(r"[^a-z0-9]+", fpath.lower())) - {"tests", "fixtures", "sql", "json", ""}
                for tfile, tnums in tests_by_file.items():
                    if any(tok in tfile.lower() for tok in stem_tokens if len(tok) >= 3):
                        selected_nums.update(tnums - stress_nums)
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
                    selected_nums.update(set(dialect_map[stem]) - stress_nums)
                    selected_nums.update(
                        (tests_by_file.get("tests/dialects/test_dialect.py", set())
                         | tests_by_file.get("tests/test_transpile.py", set())
                         | set(per_dialect_opt.get(stem, dialect_aware_opt)))
                        - (stress_nums | heavy_opt_only_nums)
                    )
                    selected_nums.update(direct_importers.get(mod, set()) - stress_nums)
                elif fpath.startswith("sqlglot/optimizer/"):
                    selected_nums.update(f_tests)
                    selected_nums.update(optimizer_suites)
                    selected_nums.update(direct_importers.get(mod, set()) - stress_nums)
                elif fpath.startswith("sqlglot/executor/"):
                    selected_nums.update(
                        lookup_function_level(fpath, lines, cov_map, func_ranges, file_all_tests, direct_importers, mod)
                    )
                    selected_nums.update(direct_importers.get(mod, set()) - stress_nums)
                elif len(f_tests) <= 350:
                    selected_nums.update(f_tests)
                    selected_nums.update(direct_importers.get(mod, set()) - stress_nums)
                else:
                    selected_nums.update(
                        lookup_function_level(fpath, lines, cov_map, func_ranges, file_all_tests, direct_importers, mod)
                    )
            else:
                selected_nums.update(all_nums)

        selected_tests = [by_num[n] for n in selected_nums]
        workers = choose_best_workers(selected_tests)
        print(json.dumps({"commit": patch_path.stem, "workers": workers}))


if __name__ == "__main__":
    main()
