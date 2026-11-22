import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

DATA = Path("/data")
FIXTURE_PATH = Path("/app/fixture_map.json.gz")
SAMPLE_TESTS_PATH = Path("/app/sample_tests.json")
MAX_WORKERS = 8
WORKER_COST = 2.0

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

PARSER_STRESS_ID = "tests/test_parser.py::TestParser::test_max_nodes"
GENERATOR_STRESS_ID = "tests/test_generator.py::TestGenerator::test_generator_memory"
TPCDS_EXEC_ID = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
TPCH_EXEC_IDS = {
    "tests/test_executor.py::TestExecutor::test_execute_tpch",
    "tests/test_executor.py::TestExecutor::test_optimized_tpch",
}


def parse_patch_files(patch_text: str) -> set[str]:
    files: set[str] = set()
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4 and parts[3].startswith("b/"):
                files.add(parts[3][2:])
        elif line.startswith("+++ "):
            target = line[4:].strip()
            if target != "/dev/null":
                files.add(target[2:] if target.startswith("b/") else target)
        elif line.startswith("--- "):
            target = line[4:].strip()
            if target != "/dev/null":
                files.add(target[2:] if target.startswith("a/") else target)
    return files


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


def main() -> None:
    tests_raw = json.loads((DATA / "tests.json").read_text(encoding="utf-8"))["tests"]
    by_num = {t["number"]: t for t in tests_raw}
    id_to_num = {t["id"]: t["number"] for t in tests_raw}
    all_nums = set(by_num.keys())

    sample_num_to_id: dict[int, str] = {}
    if SAMPLE_TESTS_PATH.exists():
        sample_tests = json.loads(SAMPLE_TESTS_PATH.read_text(encoding="utf-8"))["tests"]
        sample_num_to_id = {t["number"]: t["id"] for t in sample_tests}

    def remap_nums(raw_nums: list[int] | set[int]) -> set[int]:
        out: set[int] = set()
        for n in raw_nums:
            tid = sample_num_to_id.get(n)
            if tid and tid in id_to_num:
                out.add(id_to_num[tid])
            elif n in by_num:
                out.add(n)
        return out

    parser_stress_nums = {t["number"] for t in tests_raw if t["id"] == PARSER_STRESS_ID}
    gen_stress_nums = {t["number"] for t in tests_raw if t["id"] == GENERATOR_STRESS_ID}
    tpcds_nums = {t["number"] for t in tests_raw if t["id"] == TPCDS_EXEC_ID}
    tpch_nums = {t["number"] for t in tests_raw if t["id"] in TPCH_EXEC_IDS}
    giant_nums = parser_stress_nums | gen_stress_nums | tpcds_nums | tpch_nums

    tests_by_file: dict[str, set[int]] = defaultdict(set)
    for t in tests_raw:
        tests_by_file[t["file"]].add(t["number"])

    with gzip.open(FIXTURE_PATH, "rt", encoding="utf-8") as f:
        fix_payload = json.load(f)
    fixture_map: dict[str, set[int]] = {k: remap_nums(v) for k, v in fix_payload["fixtures"].items()}

    opt_all = tests_by_file.get("tests/test_optimizer.py", set())
    base_fast_all = all_nums - giant_nums

    for patch_path in sorted((DATA / "commits").glob("*.patch")):
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
        changed_files = parse_patch_files(patch_text)

        selected_nums: set[int] = set()
        for fpath in changed_files:
            if is_doc_only(fpath):
                continue
            elif fpath in tests_by_file:
                selected_nums.update(tests_by_file[fpath])
            elif fpath.startswith("tests/fixtures/optimizer/"):
                selected_nums.update(opt_all)
                if fpath in fixture_map:
                    selected_nums.update(fixture_map[fpath])
            elif fpath.startswith("tests/fixtures/"):
                selected_nums.update(base_fast_all)
            elif fpath.startswith("sqlglot/"):
                selected_nums.update(base_fast_all)
                is_leaf_dialect = (
                    fpath != "sqlglot/dialects/dialect.py"
                    and fpath.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/"))
                )
                if is_leaf_dialect:
                    if fpath.startswith("sqlglot/parsers/"):
                        selected_nums.update(parser_stress_nums)
                    if fpath.startswith("sqlglot/generators/"):
                        selected_nums.update(gen_stress_nums)
                else:
                    selected_nums.update(tpcds_nums | tpch_nums)
                    if fpath in {"sqlglot/parser.py", "sqlglot/tokens.py"} or fpath.startswith("sqlglot/expressions"):
                        selected_nums.update(parser_stress_nums)
                    if fpath == "sqlglot/generator.py" or fpath.startswith("sqlglot/expressions"):
                        selected_nums.update(gen_stress_nums)
            else:
                selected_nums.update(all_nums)

        selected_tests = [by_num[n] for n in selected_nums]
        workers = choose_best_workers(selected_tests)
        print(json.dumps({"commit": patch_path.stem, "workers": workers}))


if __name__ == "__main__":
    main()
