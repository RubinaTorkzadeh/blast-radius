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

PARSER_STRESS_ID = "tests/test_parser.py::TestParser::test_max_nodes"
GENERATOR_STRESS_ID = "tests/test_generator.py::TestGenerator::test_generator_memory"
HEAVY_EXECUTOR_IDS = {
    "tests/test_executor.py::TestExecutor::test_execute_tpcds",
    "tests/test_executor.py::TestExecutor::test_execute_tpch",
}


def parse_patch(patch_text: str) -> dict[str, set[int]]:
    file_lines: dict[str, set[int]] = defaultdict(set)
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
                new_line += 1
            elif line.startswith("-") and not line.startswith("---"):
                file_lines[cur_file].add(max(1, old_line))
                old_line += 1
            else:
                old_line += 1
                new_line += 1

    return dict(file_lines)


def split_lpt(tests: list[dict], worker_count: int) -> tuple[float, list[list[int]]]:
    workers: list[list[dict]] = [[] for _ in range(worker_count)]
    loads = [0.0] * worker_count
    for test in sorted(tests, key=lambda t: t["seconds"], reverse=True):
        lightest = loads.index(min(loads))
        workers[lightest].append(test)
        loads[lightest] += test["seconds"]

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
                    if max(heavy_load - sec_a, other_load + sec_a) < heavy_load - 1e-6:
                        workers[heavy_idx].pop(i)
                        workers[other_idx].append(t_a)
                        loads[heavy_idx] -= sec_a
                        loads[other_idx] += sec_a
                        improved = True
                        break
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
) -> set[int]:
    line_dict = cov_map.get(fpath, {})
    spans = func_ranges.get(fpath, [])
    expanded_lines: set[int] = set()
    for ln in lines:
        enclosing = [(s, e) for (s, e) in spans if (s - 15) <= ln <= (e + 15)]
        for (s, e) in enclosing:
            expanded_lines.update(range(s, e + 1))
        expanded_lines.update(range(max(1, ln - 15), ln + 16))

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

    parser_stress_nums = {t["number"] for t in tests_raw if t["id"] == PARSER_STRESS_ID}
    gen_stress_nums = {t["number"] for t in tests_raw if t["id"] == GENERATOR_STRESS_ID}
    heavy_exec_nums = {t["number"] for t in tests_raw if t["id"] in HEAVY_EXECUTOR_IDS}
    giant_nums = parser_stress_nums | gen_stress_nums | heavy_exec_nums

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

    all_dialect_suites: set[int] = set()
    core_unit_suites: set[int] = set()
    for tf, tnums in tests_by_file.items():
        if tf.startswith("tests/dialects/"):
            all_dialect_suites.update(tnums - giant_nums)
        elif tf not in {"tests/test_optimizer.py", "tests/test_executor.py"}:
            core_unit_suites.update(tnums - giant_nums)

    opt_all = tests_by_file.get("tests/test_optimizer.py", set())
    exec_all = tests_by_file.get("tests/test_executor.py", set())
    exec_fast = exec_all - heavy_exec_nums

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
                selected_nums.update(all_dialect_suites)
                selected_nums.update(core_unit_suites)
                selected_nums.update(opt_all)
            elif fpath.startswith("tests/fixtures/"):
                selected_nums.update(all_dialect_suites)
                selected_nums.update(core_unit_suites)
                selected_nums.update(opt_all)
            elif fpath.startswith("sqlglot/"):
                stem = Path(fpath).stem
                mod = module_name_for_path(fpath)
                selected_nums.update(all_dialect_suites)
                selected_nums.update(direct_importers.get(mod, set()) - giant_nums)
                if stem in dialect_map:
                    selected_nums.update(set(dialect_map[stem]) - giant_nums)

                if fpath.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")):
                    selected_nums -= exec_all
                    selected_nums.update(dialect_aware_opt - giant_nums)
                    selected_nums.update(tests_by_file.get("tests/test_transpile.py", set()) - giant_nums)
                elif fpath.startswith(("sqlglot/optimizer/", "sqlglot/executor/")):
                    selected_nums.update(core_unit_suites)
                    selected_nums.update(opt_all)
                    selected_nums.update(exec_all)
                elif fpath in {
                    "sqlglot/expressions.py",
                    "sqlglot/schema.py",
                    "sqlglot/planner.py",
                    "sqlglot/errors.py",
                    "sqlglot/__init__.py",
                    "sqlglot/helper.py",
                    "sqlglot/tokens.py",
                    "sqlglot/typing.py",
                }:
                    selected_nums.update(core_unit_suites)
                    selected_nums.update(opt_all)
                    selected_nums.update(exec_all)
                    if fpath == "sqlglot/tokens.py":
                        selected_nums.update(parser_stress_nums)
                elif fpath == "sqlglot/parser.py":
                    selected_nums.update(core_unit_suites)
                    selected_nums.update(dialect_aware_opt - giant_nums)
                    selected_nums.update(parser_stress_nums)
                    func_hits = lookup_function_level(fpath, lines, cov_map, func_ranges, file_all_tests)
                    selected_nums.update(func_hits - giant_nums)
                    selected_nums.update(func_hits & heavy_exec_nums)
                elif fpath == "sqlglot/generator.py":
                    selected_nums.update(core_unit_suites)
                    selected_nums.update(dialect_aware_opt - giant_nums)
                    selected_nums.update(gen_stress_nums)
                    func_hits = lookup_function_level(fpath, lines, cov_map, func_ranges, file_all_tests)
                    selected_nums.update(func_hits - giant_nums)
                    selected_nums.update(func_hits & heavy_exec_nums)
                else:
                    selected_nums.update(core_unit_suites)
                    selected_nums.update(opt_all)
                    selected_nums.update(exec_all)
            else:
                selected_nums.update(all_nums)

        selected_tests = [by_num[n] for n in selected_nums]
        workers = choose_best_workers(selected_tests)
        print(json.dumps({"commit": patch_path.stem, "workers": workers}))


if __name__ == "__main__":
    main()
