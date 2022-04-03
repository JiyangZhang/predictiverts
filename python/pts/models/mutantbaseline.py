from pathlib import Path
from typing import List
from seutil import IOUtils
from pts.Macros import Macros
import collections
from sklearn.metrics import recall_score


def build_line_num_to_test_map(project_name, all_covered=False):
    pit_report_data = IOUtils.load(
        Macros.repos_results_dir / project_name / f"{project_name}-default-mutation-report.json")
    mutant_line_num_to_killingTest_per_file = collections.defaultdict(dict)
    for pit_report_data_item in pit_report_data:
        if pit_report_data_item["status"] == "KILLED":
            filename = pit_report_data_item["sourceFile"]  # xx.java
            linenumber = pit_report_data_item["lineNumber"]  # 777
            full_killed_tests: List[List[str]] = pit_report_data_item["killingTests"]
            if all_covered and "succeedingTests" in pit_report_data_item and pit_report_data_item["succeedingTests"]:
                full_killed_tests += pit_report_data_item["succeedingTests"]
            killed_test_set = set([kt[0].split('.')[-1] for kt in full_killed_tests])
            if linenumber in mutant_line_num_to_killingTest_per_file[filename]:
                mutant_line_num_to_killingTest_per_file[filename][linenumber] |= killed_test_set
            else:
                mutant_line_num_to_killingTest_per_file[filename][linenumber] = killed_test_set
        if all_covered and pit_report_data_item["status"] == "SURVIVED":
            filename = pit_report_data_item["sourceFile"]  # xx.java
            linenumber = pit_report_data_item["lineNumber"]  # 777
            full_survived_tests: List[List[str]] = pit_report_data_item["succeedingTests"]
            survived_test_set = set([kt[0].split('.')[-1] for kt in full_survived_tests])
            if linenumber in mutant_line_num_to_killingTest_per_file[filename]:
                mutant_line_num_to_killingTest_per_file[filename][linenumber] |= survived_test_set
            else:
                mutant_line_num_to_killingTest_per_file[filename][linenumber] = survived_test_set
    return mutant_line_num_to_killingTest_per_file


def run_mutant_baseline_model(eval_data_dir: Path, output_dir: Path, project_name, search_span=10, use_deleted=False, all_covered=False):
    if all_covered and (output_dir/"killed-all-mutant-linenumber-to-test.json").exists():
        mutant_line_num_to_killingTest_per_file = IOUtils.load(output_dir/"killed-all-mutant-linenumber-to-test.json")
    elif not all_covered and (output_dir/"killed-mutant-linenumber-to-test.json").exists():
        mutant_line_num_to_killingTest_per_file = IOUtils.load(output_dir / "killed-all-mutant-linenumber-to-test.json")
    else:
        mutant_line_num_to_killingTest_per_file = build_line_num_to_test_map(project_name, all_covered)
        for k, v in mutant_line_num_to_killingTest_per_file.items():
            for l, t in v.items():
                v[l] = list(t)
        if all_covered:
            IOUtils.dump(output_dir / "killed-all-mutant-linenumber-to-test.json", dict(mutant_line_num_to_killingTest_per_file))

    baseline_model_res_dict = {}
    eval_data = IOUtils.load(eval_data_dir)
    for eval_data_item in eval_data:
        not_covered_file_num = 0
        selected_tests = set()
        deleted_lines_per_file = {}
        if use_deleted:
            deleted_lines_eval_data = IOUtils.load(
                Macros.eval_data_dir / "raw-eval-data-adding-deleted" / f"{project_name}.json")
            current_commit = eval_data_item["commit"].split('-')[0]
            for d in deleted_lines_eval_data:
                if d["commit"] == current_commit:
                    deleted_lines_per_file = d["deleted_line_number_list_per_file"]
        line_num_per_file = eval_data_item["diff_line_number_list_per_file"]
        for filename, line_num_list in line_num_per_file.items():
            if use_deleted:
                if filename in deleted_lines_per_file:
                    line_num_list += deleted_lines_per_file[filename]
            if "src/test" in filename:
                continue
            filename = filename.split("/")[-1]
            if filename not in mutant_line_num_to_killingTest_per_file:
                not_covered_file_num += 1
                continue
            for line in line_num_list:
                min_bound_line = max(1, line - search_span)
                max_bound_line = line + search_span
                for l in range(min_bound_line, max_bound_line + 1):
                    history_lines = [int(l) for l in mutant_line_num_to_killingTest_per_file[filename].keys()]
                    if l in history_lines:
                        selected_tests |= set(mutant_line_num_to_killingTest_per_file[filename][str(l)])
        all_tests = eval_data_item["passed_test_list"] + eval_data_item["failed_test_list"]
        all_test_labels = [0 for _ in range(len(eval_data_item["passed_test_list"]))] + [1 for _ in range(len(eval_data_item["failed_test_list"]))]
        baseline_model_res = [0 for _ in range(len(all_tests))]
        for index, test in enumerate(all_tests):
            if test in selected_tests:
                baseline_model_res[index] = 1
        # end for
        if sum(baseline_model_res) == 0:
            recall_rate = 0
        else:
            recall_rate = recall_score(all_test_labels, baseline_model_res)

        baseline_model_res_dict[eval_data_item["commit"]] = {
            "baseline-select-rate": len(selected_tests) / len(all_tests),
            "baseline-recall": recall_rate,
            "not-covered-file": not_covered_file_num/len(line_num_per_file)
        }
    if use_deleted:
        IOUtils.dump(output_dir / f"baseline-del-{search_span}-model-eval-results.json", baseline_model_res_dict)
    elif all_covered:
        IOUtils.dump(output_dir / f"baseline-all-mutant-{search_span}-model-eval-results.json", baseline_model_res_dict)
    else:
        IOUtils.dump(output_dir / f"baseline-{search_span}-model-eval-results.json", baseline_model_res_dict)
