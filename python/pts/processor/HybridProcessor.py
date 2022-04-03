import collections
import random
from pathlib import Path
from typing import List, Dict

from seutil import LoggingUtils, IOUtils, BashUtils

from pts.Environment import Environment
from pts.Macros import Macros
from pts.data import diff_utils
from pts.processor.data_utils.SubTokenizer import SubTokenizer
from pts.collector.mutation.rtstool_tests_collector import getTestsFromSTARTS

class HybridProcessor:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.repos_downloads_dir = Macros.repos_downloads_dir
        self.repos_result_dir = Macros.repos_results_dir
        self.TRAIN_RATIO = 0.8
        self.VAL_RATIO = 0.1
        self.TEST_RATIO = 0.1

    def process_train(self, proj_names: List[str]):
        """
        Create the dataset to train the hybrid model
        Only consider Killed mutants
        """
        # proj_names: List = listdir(self.repos_result_dir)  # a list of project name
        proj_dirs: List = [Path(self.repos_result_dir / proj) for proj in proj_names]  # a list of project dirs
        for proj, proj_dir in zip(proj_names, proj_dirs):
            # Get all test class names, with the methods in them
            test_class_2_methods = collections.defaultdict(list)
            test_class_file_list = BashUtils.run(f"find {Macros.repos_downloads_dir}/{proj}_ekstazi/.ekstazi/ "
                                                 f"-name \"*Test*.clz\"").stdout.split("\n")
            collected_results_dir: Path = proj_dir / "collector"
            method_dict = IOUtils.load(collected_results_dir / "method-data.json")
            for cf in test_class_file_list:
                if cf != "":
                    class_name = cf.split('/')[-1].split('.')[-2]
                    for m in method_dict:
                        if m["class_name"] == class_name:
                            test_class_2_methods[class_name].append(m["name"])
                # end if
            # end for
            IOUtils.dump(collected_results_dir / "test2meth.json", test_class_2_methods)
            all_test_classes = list(test_class_2_methods.keys())
            processed_data = list()
            processed_valid_data = list()
            valid_data_id = 0
            output_dir = self.output_dir / proj.split('_')[1] / "hybrid"
            IOUtils.mk_dir(output_dir)

            objs = IOUtils.load_json_stream(collected_results_dir / "mutant-data-rts-tool.json")
            for obj in objs:
                if obj["status"] == "TIME_OUT":
                    continue
                # get diff sequence
                old_code = obj["old_code"].strip().split()
                new_code = obj["new_code"].strip().split()
                edit_seq, _, _ = diff_utils.compute_code_diffs(old_code, new_code)
                killed_test_set = set()
                if obj["succeedingTests"] == "All":
                    continue
                elif obj["succeedingTests"] == "Remaining":
                    assert obj["status"] == "KILLED"
                    # extract killed test
                    for t in obj["killingTests"]:
                        if t[0] == t[1] and t[0] != "":
                            tc = t[0].split('.')[-1]
                            if tc in all_test_classes:
                                killed_test_set.add(tc)
                        elif t[0] == "":
                            try:
                                tc = t[1].split(".")[-2]
                            except IndexError:
                                tc = t[1]
                            if tc in all_test_classes:
                                killed_test_set.add(tc)
                        else:
                            try:
                                tc = t[0].split(".")[-1]
                            except IndexError:
                                tc = t[0]
                            if tc in all_test_classes:
                                killed_test_set.add(tc)
                        # end if
                    # end for
                    # Decide train or valid data
                    pb = random.uniform(0, 1)
                    if pb > 0.08:
                        starts_tests = set(obj["STARTS"])
                        lived_test_class = starts_tests.difference(killed_test_set)
                        for positive_test in killed_test_set:
                            pos_test_input = SubTokenizer.sub_tokenize_java_like(positive_test) + \
                                             SubTokenizer.sub_tokenize_java_like(
                                                 " ".join(test_class_2_methods[positive_test]))
                            for negative_test in lived_test_class:
                                neg_test_input = SubTokenizer.sub_tokenize_java_like(negative_test) + \
                                                 SubTokenizer.sub_tokenize_java_like(
                                                     " ".join(test_class_2_methods[negative_test]))
                                data_point = {
                                    "code_diff": edit_seq,
                                    "pos_test_code": pos_test_input,
                                    "neg_test_code": neg_test_input
                                }
                                processed_data.append(data_point)
                            # end for
                        # end for
                    else:
                        # valid data
                        starts_tests = set(obj["STARTS"])
                        lived_test_class = starts_tests.difference(killed_test_set)
                        for positive_test in killed_test_set:
                            pos_test_input = SubTokenizer.sub_tokenize_java_like(positive_test) + \
                                             SubTokenizer.sub_tokenize_java_like(
                                                 " ".join(test_class_2_methods[positive_test]))
                            for negative_test in lived_test_class:
                                neg_test_input = SubTokenizer.sub_tokenize_java_like(negative_test) + \
                                                 SubTokenizer.sub_tokenize_java_like(
                                                     " ".join(test_class_2_methods[negative_test]))
                                processed_valid_data.append(
                                    {
                                        "id": valid_data_id,
                                        "code_diff": edit_seq,
                                        "pos_test_code": pos_test_input,
                                        "neg_test_code": neg_test_input
                                    }
                                )
                        # end for
                    # end if
                # end if
            # end for
            print(f"In total there are {len(processed_data)} data point for training")
            print(f"In total there are {len(processed_valid_data)} data point for validation")
            IOUtils.dump(output_dir / "train.json", processed_data)
            IOUtils.dump(output_dir / "valid.json", processed_valid_data)

    def process_eval(self, src_data_dir: Path, proj: str,  train_sha=""):
        """
        Prepare eval data for the model from the real-world shas.
        for augumented data
        Note:
            * will filter those shas without any code diff
            * will filter those shas that ekstazi and starts select nothing
        """
        shas_data = IOUtils.load(src_data_dir)
        data_list = list()
        pos_num = 0
        neg_num = 0
        discard_sha = 0
        empty_sha = 0
        ekstazi_selected_num = list()
        starts_selected_num = list()
        for sha in shas_data:
            if sha["commit"] == train_sha:
                discard_sha += 1
                continue
            if "diff_per_file" not in sha:
                discard_sha += 1
                bad_sha = sha["commit"]
                self.logger.warning(f"This sha {bad_sha} can not get diff per file!")
                continue
            if len(sha["ekstazi_test_list"]) == 0 or len(sha["starts_test_list"]) == 0:
                self.logger.info(f"Discard non-bytecode changes sha.")
                empty_sha += 1
                continue
            failed_test_clss: List = list(set([t.split('.')[-1] for t in sha["failed_test_list"]]))
            # passed_test_clss: List = list(set([t.split('.')[-1] for t in sha["passed_test_list"]]))
            # ekstazi_selected: List = list(set([t.split('.')[-1] for t in sha["ekstazi_test_list"]]))
            # starts_selected: List = list(set([t.split('.')[-1] for t in sha["starts_test_list"]]))

            # Get all candidate tests
            changed_classes = ["/"+ch_file.split('/')[-1].split('.')[-2]+".class" for ch_file in sha["changed_files"]]
            candidate_tests = []
            with IOUtils.cd(f"{Macros.repos_downloads_dir}/{proj}_starts"):
                for changed_class in changed_classes:
                    candidate_tests.extend(getTestsFromSTARTS(changed_class))
            candidate_tests = list(set(candidate_tests))
            assert set(candidate_tests).issuperset(set(failed_test_clss))

            changed_files_code = [(c.split('/')[-1], sha["diff_per_file"][c]) for c in sha["changed_files"]]

            # Currently only use code diff bcause we do not know what context to useTODO: use new method body
            if sha["diff_code"] == "":
                self.logger.warning("Discard this sha {}.".format(sha["commit"]))
                discard_sha += 1
                continue
            collected_results_dir = Macros.repos_results_dir / proj / "collector"
            test_class_2_methods = IOUtils.load(collected_results_dir / "test2meth.json")

            passed_test_clss = set(candidate_tests).difference(set(failed_test_clss))

            if len(failed_test_clss) > 0:
                for cls, code in changed_files_code:
                    if "java" not in cls:
                        continue
                    # get edit sequence
                    diff_list = code.split("\n")[:-1]
                    old_code = " ".join([dif[1:].strip() for dif in diff_list if dif[0] == '-']).split()
                    new_code = " ".join([dif[1:].strip() for dif in diff_list if dif[0] == '+']).split()
                    edit_seq, _, _ = diff_utils.compute_code_diffs(old_code, new_code)
                    for ftc in failed_test_clss:
                        # get test code
                        pos_test_input = SubTokenizer.sub_tokenize_java_like(ftc) + \
                                         SubTokenizer.sub_tokenize_java_like(
                                             " ".join(test_class_2_methods[ftc]))
                        data_point = {
                            "sha": sha["commit"],
                            "label": 1,
                            "code_diff": edit_seq,
                            "pos_test_code": pos_test_input,
                            "neg_test_code": pos_test_input
                        }
                        data_list.append(data_point)
                        pos_num += 1
                    # end for
                    for ptc in passed_test_clss:
                        # get test code
                        neg_test_input = SubTokenizer.sub_tokenize_java_like(ptc) + \
                                         SubTokenizer.sub_tokenize_java_like(
                                             " ".join(test_class_2_methods[ptc]))
                        data_point = {
                            "sha": sha["commit"],
                            "label": 0,
                            "code_diff": edit_seq,
                            "pos_test_code": neg_test_input,
                            "neg_test_code": neg_test_input
                        }
                        data_list.append(data_point)
                        neg_num += 1
                    # end for
                # end for

            # # add negative examples
            # if passed_test_clss:
            #     for ptc in passed_test_clss:
            #         if ptc in ekstazi_selected:
            #             ekstazi_label = 1
            #         else:
            #             ekstazi_label = 0
            #         if ptc in starts_selected:
            #             starts_label = 1
            #         else:
            #             starts_label = 0
            #         for cls, code in zip(changed_clss, changed_code):
            #             diff_list = code.split("\n")[:-1]
            #             old_dif = " ".join([dif[1:].strip() for dif in diff_list if dif[0] == '-'])
            #             new_dif = " ".join([dif[1:].strip() for dif in diff_list if dif[0] == '+'])
            #             mutated_code: str = old_dif + ' ' + new_dif
            #             if mutated_code == "":
            #                 self.logger.warning("Can not get diff: {}".format(sha["diff_code"]))
            #             data_point = {
            #                 "label": 0,
            #                 "input": [cls, mutated_code, ptc],
            #                 "ekstazi_label": ekstazi_label,
            #                 "starts_label": starts_label,
            #                 "sha": sha["commit"]
            #             }
            #             data_list.append(data_point)
            #         # end for
            #     # end for
        self.logger.info(f"In total there are {len(data_list)} number of data point for test. {pos_num} are positive and {neg_num} are negative")
        print(f"Discard {discard_sha} SHAs in total, {empty_sha} SHAs have no bytecode changes.")
        remain_shas = len(shas_data) - 1 - discard_sha - empty_sha
        IOUtils.dump(self.output_dir / proj.split('_')[1] / f"test.json", data_list, IOUtils.Format.jsonNoSort)
