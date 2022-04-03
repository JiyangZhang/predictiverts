from random import sample, shuffle
import random
from pathlib import Path
from typing import List, Dict
import ipdb
from collections import defaultdict
from seutil import LoggingUtils, IOUtils, BashUtils

from pts.Environment import Environment
from pts.Macros import Macros
from pts.data import diff_utils
from pts.processor.RankProcess import RankProcessor
from pts.processor.data_utils.SubTokenizer import SubTokenizer
from pts.data.diff_utils import extract_code_diffs, removeComments

SEED = 14

class CodeBertProcessor:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.repos_downloads_dir = Macros.repos_downloads_dir
        self.repos_result_dir = Macros.repos_results_dir

    def process_train_tools(self, proj: str):
        """
        Create the dataset to train the codeBert model, this function uses the tests selected by tools as the positive
        examples.
        Ignore time-out mutants;
        Features: code diff, test file (truncated)
        """
        # Load required data files
        proj_dir = Path(self.repos_result_dir / proj)
        collected_results_dir: Path = proj_dir / "collector"
        methods_list = IOUtils.load(collected_results_dir / "method-data.json")
        objs = IOUtils.load_json_stream(collected_results_dir / "mutant-data-rts-tool.json")
        context_objs = IOUtils.load_json_stream(collected_results_dir / "mutant-data.json")
        if (collected_results_dir / "method-data.json").exists():
            test_class_2_methods = IOUtils.load(collected_results_dir / "test2methods.json")
            print(f"{proj} has {len(test_class_2_methods.keys())} test classes in total.")
        else:
            raise FileNotFoundError("Can not find test2method.json.")
        all_test_classes = list(test_class_2_methods.keys())
        test_class_num = len(all_test_classes)

        for data_type in ["Ekstazi"]:
            mutated_class_set = set()
            processed_train_data = list()
            output_dir = self.output_dir / proj.split('_')[1] / data_type
            IOUtils.mk_dir(output_dir)

            for obj, cobj in zip(objs, context_objs):
                assert obj["description"] == cobj["description"]
                if obj["status"] == "TIME_OUT":
                    continue
                # Decide train or valid data
                if obj["mutatedClass"] in mutated_class_set:
                    continue
                else:
                    mutated_class_set.add(obj["mutatedClass"])
                # get mutated class name
                mutated_clss = obj["mutatedClass"].split('.')[-1]
                mutated_method = obj["mutatedMethod"]

                code_diffs = obj["new_code"]
                context = cobj["context"]

                killed_test_set = set(obj[data_type])
                lived_test_class = set(all_test_classes).difference(killed_test_set)
                for positive_test in killed_test_set:
                    pos_test_class = positive_test
                    pos_test_methods = []
                    try:
                        test_methods_id = test_class_2_methods[positive_test]
                        for m_id in test_methods_id:
                            test_method_code = methods_list[m_id]["code"]
                            pos_test_methods.append(test_method_code)
                    except KeyError:
                        pos_test_methods.append("None")
                    for negative_test in lived_test_class:
                        neg_test_class = negative_test
                        neg_test_methods = []
                        try:
                            test_methods_id = test_class_2_methods[negative_test]
                            for m_id in test_methods_id:
                                test_method_code = methods_list[m_id]["code"]
                                neg_test_methods.append(test_method_code)
                        except KeyError:
                            neg_test_methods.append("None")
                        data_point = {
                            "changed_class_name": mutated_clss,
                            "changed_method_name": mutated_method.lstrip(),
                            "code_diff": code_diffs.lstrip(),
                            "context": context.lstrip(),
                            "abstract_code_diff": obj["mutator"],
                            "pos_test_class": pos_test_class,
                            "pos_test_methods": pos_test_methods,
                            "neg_test_class": neg_test_class,
                            "neg_test_methods": neg_test_methods
                        }
                        processed_train_data.append(data_point)
                        # end if
                    # end for
                # end for
            # end for
            print(f"{data_type}")
            data_size = len(processed_train_data)
            valid_size = int(0.2 * data_size)
            random.Random(SEED).shuffle(processed_train_data)
            processed_valid_data = processed_train_data[: valid_size]

            print(f"In total there are {len(processed_train_data)} data point for training")
            print(f"In total there are {len(processed_valid_data)} data point for validation")
            shuffle(processed_train_data)
            shuffle(processed_valid_data)
            IOUtils.dump(output_dir / "train.json", processed_train_data, IOUtils.Format.jsonNoSort)
            IOUtils.dump(output_dir / "valid.json", processed_valid_data, IOUtils.Format.jsonNoSort)

    def process_train_fail(self, proj: str):
        """
        Create the dataset to train the rank model, this function uses the failed tests as the positive
        examples for the codeBert model.
        """
        # Load required data files
        proj_dir = Path(self.repos_result_dir / proj)  # a list of project dirs
        collected_results_dir = proj_dir / "collector"
        methods_list = IOUtils.load(collected_results_dir / "method-data.json")
        objs = IOUtils.load_json_stream(collected_results_dir / "mutant-data-rts-tool.json")
        context_objs = IOUtils.load_json_stream(collected_results_dir / "mutant-data.json")
        if (collected_results_dir / "method-data.json").exists():
            test_class_2_methods = IOUtils.load(collected_results_dir / "test2methods.json")
        else:
            raise FileNotFoundError("Can not find test2methods.json.")
        all_test_classes = list(test_class_2_methods.keys())
        test_class_num = len(all_test_classes)

        # initialize the list to store the data
        processed_train_data = []
        output_dir = self.output_dir / proj.split('_')[1]
        if not output_dir.exists():
            IOUtils.mk_dir(output_dir)

        for obj, cobj in zip(objs, context_objs):
            assert obj["description"] == cobj["description"]
            if obj["status"] == "TIME_OUT":
                continue

            mutator = obj["mutator"]
            killed_test_set = set()
            if obj["succeedingTests"] == "All":
                continue
            elif obj["status"] == "KILLED":
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
                # extract pass tests
                lived_test_class = set(all_test_classes).difference(killed_test_set)
                # get mutated class name
                mutated_clss = obj["mutatedClass"].split('.')[-1]
                mutated_method = obj["mutatedMethod"]
                # extract code diff
                code_diffs = ' '.join(obj["new_code"].lstrip().split())
                context = ' '.join(cobj["context"].lstrip().split())
                for positive_test in killed_test_set:
                    pos_test_input = positive_test
                    pos_test_methods = []
                    try:
                        test_methods_id = test_class_2_methods[positive_test]
                        for m_id in test_methods_id:
                            test_method_code = ' '.join(methods_list[m_id]["code"].lstrip().split())
                            pos_test_methods.append(test_method_code)
                    except KeyError:
                        pos_test_methods.append("None")

                    sample_size = min(len(lived_test_class), 20)  # predefined maximum living test classes
                    sampled_lived_test_class = sample(lived_test_class, sample_size)
                    for negative_test in sampled_lived_test_class:
                        neg_test_input = negative_test
                        neg_test_methods = []
                        try:
                            test_methods_id = test_class_2_methods[negative_test]
                            for m_id in test_methods_id:
                                test_method_code = ' '.join(methods_list[m_id]["code"].lstrip().split())
                                neg_test_methods.append(test_method_code)
                        except KeyError:
                            neg_test_methods.append("None")
                        data_point = {
                            "changed_class_name": mutated_clss.lstrip(),
                            "changed_method_name": mutated_method.lstrip(),
                            "code_diff": code_diffs.lstrip(),
                            "context": context,
                            "abstract_code_diff": mutator,
                            "pos_test_class": pos_test_input,
                            "pos_test_methods": pos_test_methods,
                            "neg_test_class": neg_test_input,
                            "neg_test_methods": neg_test_methods
                        }
                        processed_train_data.append(data_point)
                    # end for
                # end for
            # end for
        random.Random(SEED).shuffle(processed_train_data)
        data_size = len(processed_train_data)
        valid_size = int(0.2*data_size)
        processed_valid_data = processed_train_data[: valid_size]
        processed_train_data = processed_train_data[valid_size: ]
        print(f"In total there are {len(processed_train_data)} data point for training")
        print(f"In total there are {len(processed_valid_data)} data point for validation")
        shuffle(processed_train_data)
        shuffle(processed_valid_data)
        IOUtils.dump(output_dir / "fail-train.json", processed_train_data, IOUtils.Format.jsonNoSort)
        IOUtils.dump(output_dir / "fail-valid.json", processed_valid_data, IOUtils.Format.jsonNoSort)

    def process_eval(self, src_data_dir: Path, proj: str, train_sha=""):
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
        ekstazi_selected_num = list()
        starts_selected_num = list()

        # Iterate data to process
        for sha in shas_data:
            # do sanity check
            if "diff_per_file" not in sha:
                bad_sha = sha["commit"]
                self.logger.warning(f"This sha {bad_sha} does not have diff per file. Please fix the bug.")
                discard_sha += 1
                continue

            # load data files
            proj_dir = Path(self.repos_result_dir / proj / sha["commit"].split('-')[0])  # a list of project dirs
            collected_results_dir = proj_dir / "collector"
            methods_list = IOUtils.load(collected_results_dir / "method-data.json")
            if (collected_results_dir / "method-data.json").exists():
                test_class_2_methods = IOUtils.load(collected_results_dir / "test2methods.json")
            else:
                raise FileNotFoundError("Can not find test2methods.json.")

            failed_test_clss: List = sha["failed_test_list"]
            passed_test_clss: List = sha["passed_test_list"]
            ekstazi_selected: List = sha["ekstazi_test_list"]
            starts_selected: List = sha["starts_test_list"]

            # tuple of (ChangedFile.java, added_code)
            changed_files_code = [(c.split('/')[-1], sha["diff_per_file"][c]) for c in sha["diff_per_file"]]

            num_test_class = len(failed_test_clss) + len(passed_test_clss)
            num_changed_file = len(changed_files_code)

            ekstazi_selected_num.append(len(ekstazi_selected))
            starts_selected_num.append(len(starts_selected))

            if len(failed_test_clss) > 0:
                for cls, code in changed_files_code:
                    # Ignore non java files
                    if "java" not in cls:
                        continue
                    code_diffs = ' '.join(code.lstrip().split())
                    # get mutator
                    mutator_type = RankProcessor.abstract_mutated_type(code)
                    for ftc in failed_test_clss:
                        # get labels from RTS tools
                        if ftc in ekstazi_selected:
                            ekstazi_label = 1
                        else:
                            ekstazi_label = 0
                        if ftc in starts_selected:
                            starts_label = 1
                        else:
                            starts_label = 0
                        # get test code
                        pos_test_input = ftc
                        mutated_clss = cls.split('.')[-2]
                        mutated_method = []
                        test_methods = []
                        try:
                            test_methods_id = test_class_2_methods[ftc]
                            for m_id in test_methods_id:
                                test_method_code = ' '.join(methods_list[m_id]["code"].lstrip().split())
                                test_methods.append(test_method_code)
                        except KeyError:
                            test_methods.append("None")

                        data_point = {
                            "sha": sha["commit"],
                            "label": 1,
                            "changed_class_name": mutated_clss,
                            "changed_method_name": mutated_method,
                            "context": code_diffs,
                            "abstract_code_diff": mutator_type,
                            "pos_test_class": pos_test_input,
                            "pos_test_methods": test_methods,
                            "neg_test_class": pos_test_input,
                            "neg_test_methods": test_methods,
                            "ekstazi_label": ekstazi_label,
                            "starts_label": starts_label,
                            "num_test_class": num_test_class,
                            "num_changed_files": num_changed_file
                        }
                        data_list.append(data_point)
                        pos_num += 1
                    # end for
                    for ptc in passed_test_clss:
                        # get labels from RTS tools
                        if ptc in ekstazi_selected:
                            ekstazi_label = 1
                        else:
                            ekstazi_label = 0
                        if ptc in starts_selected:
                            starts_label = 1
                        else:
                            starts_label = 0
                        # get test code
                        neg_test_input = ptc
                        mutated_clss = cls.split('.')[-2]
                        mutated_method = []
                        test_methods = []
                        try:
                            test_methods_id = test_class_2_methods[ptc]
                            for m_id in test_methods_id:
                                test_method_code = ' '.join(methods_list[m_id]["code"].lstrip().split())
                                test_methods.append(test_method_code)
                        except KeyError:
                            test_methods.append("None")
                        data_point = {
                            "sha": sha["commit"],
                            "label": 0,
                            "changed_method_name": mutated_method,
                            "changed_class_name": mutated_clss,
                            "context": code_diffs,
                            "abstract_code_diff": mutator_type,
                            "pos_test_class": neg_test_input,
                            "pos_test_methods": test_methods,
                            "neg_test_class": neg_test_input,
                            "neg_test_methods": test_methods,
                            "ekstazi_label": ekstazi_label,
                            "starts_label": starts_label,
                            "num_test_class": num_test_class,
                            "num_changed_files": num_changed_file
                        }
                        data_list.append(data_point)
                        neg_num += 1
                    # end for
                # end for

        self.logger.info(
            f"In total there are {len(data_list)} number of data point for test. {pos_num} are positive and"
            f" {neg_num} are negative. In total there are {len(shas_data)} shas, {discard_sha} shas are kicked out.")
        IOUtils.dump(self.output_dir / proj.split('_')[1] / f"test.json", data_list, IOUtils.Format.jsonNoSort)

    @staticmethod
    def abstract_mutated_type(new_code: str) -> List[str]:
        new_code = new_code.split()
        math_mutators = {"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", ">>>"}
        mutator_types_list = set()
        for code_piece in new_code:
            if "return" in code_piece and ("true" in code_piece or "false" in code_piece):
                mutator_types_list.add("BooleanFalseReturnValsMutator")
            if "<" in code_piece or "<=" in code_piece or ">" in code_piece or ">=" in code_piece:
                mutator_types_list.add("ConditionalsBoundaryMutator")
            if "==" in code_piece or "!=" in code_piece:
                mutator_types_list.add("NegateConditionalsMutator")
            for piece in code_piece:
                if '!' in piece:
                    mutator_types_list.add("NegateConditionalsMutator")
            if len(set(code_piece).intersection(math_mutators)) > 0:
                mutator_types_list.add("MathMutator")
            if "++" in code_piece or "--" in code_piece or "+=" in code_piece or "-=" in code_piece:
                mutator_types_list.add("IncrementsMutator")
            if "/*" in " ".join(code_piece):
                mutator_types_list.add("EmptyObjectReturnValsMutator")
        # for code_piece in deleted_code:
        #     if "return" in code_piece:
        #         mutator_types_list.add("NullReturnValsMutator")
        #     if "();" in code_piece:
        #         mutator_types_list.add("VoidMethodCallMutator")

        if len(mutator_types_list) == 0:
            # print(f"No mutator extracted!")
            # print(old_code + new_code)
            mutator_types_list.add("VoidMethodCallMutator")
        # print(len(mutator_types_list))
        return list(mutator_types_list)[:1]

    def process_triplet_labels(self, proj_names: List[str]):
        """
        Process the data to create three categories for the data:
        Ekstazi-selected-failed: positive
        Ekstazi-selected-passed: netural
        Ekstazi-not-selected: negative
        """
        proj_dirs: List = [Path(self.repos_result_dir / proj) for proj in proj_names]  # a list of project dirs
        for proj, proj_dir in zip(proj_names, proj_dirs):
            output_dir = self.output_dir / proj.split('_')[1]
            collected_results_dir = proj_dir / "collector"
            if (collected_results_dir / "method-data.json").exists():
                test_class_2_methods = IOUtils.load(collected_results_dir / "test2meth.json")
            else:
                raise FileNotFoundError("Can not find test2meth.json.")
            all_test_classes = list(test_class_2_methods.keys())
            test_class_num = len(all_test_classes)

            # initialize the list to store the data
            processed_train_data = []  # with large margin  fail v.s. passed & not selected
            processed_train_add_data = []  # with small margin  fail v.s. selected passed; selected passed v.s. not selected
            processed_valid_data = []
            data_type = "Ekstazi"
            objs = IOUtils.load_json_stream(collected_results_dir / "mutant-data-rts-tool.json")
            for obj in objs:
                pb = random.uniform(0, 1)
                if obj["status"] == "TIME_OUT":
                    continue
                # get diff features: mutated_class_name, mutated_method_name, mutation_type
                mutator = obj["mutator"]
                killed_test_set = set()
                if obj["succeedingTests"] == "All":
                    continue
                elif obj["status"] == "KILLED":
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
                    # extract tests select by tools but not fail
                    tool_selected_tests = set(obj[data_type])
                    tool_selected_fail_tests = tool_selected_tests.intersection(killed_test_set)
                    selected_pass_tests = tool_selected_tests.difference(tool_selected_fail_tests)
                    # extract passed tests and not selected
                    fail_plus_selected_test = tool_selected_tests.union(killed_test_set)
                    lived_test_class = set(all_test_classes).difference(fail_plus_selected_test)
                    # get mutated class name: fully qualified
                    mutated_clss = SubTokenizer.sub_tokenize_java_like(obj["mutatedClass"].split('.')[-1])
                    mutated_method = SubTokenizer.sub_tokenize_java_like(obj["mutatedMethod"])
                    # extract code diff
                    code_diffs = SubTokenizer.sub_tokenize_java_like(obj["new_code"])
                    for positive_test in killed_test_set:
                        pos_test_input = SubTokenizer.sub_tokenize_java_like(positive_test)
                        sample_size = min(len(lived_test_class), 20)  # predefined maximum living test classes
                        sampled_lived_test_class = sample(lived_test_class, sample_size)
                        for negative_test in sampled_lived_test_class:
                            neg_test_input = SubTokenizer.sub_tokenize_java_like(negative_test)
                            data_point = {
                                "changed_class_name": mutated_clss,
                                "changed_method_name": mutated_method,
                                "code_diff": code_diffs,
                                "abstract_code_diff": mutator,
                                "pos_test_class": pos_test_input,
                                "neg_test_class": neg_test_input
                            }
                            pb = random.uniform(0, 1)
                            if pb > 0.1:
                                processed_train_data.append(data_point)
                            else:
                                processed_valid_data.append(data_point)
                        # end for
                    # end for
                    # Additional training data for the small margin
                    for positive_test in killed_test_set:
                        pos_test_input = SubTokenizer.sub_tokenize_java_like(positive_test)
                        for negative_test in selected_pass_tests:
                            neg_test_input = SubTokenizer.sub_tokenize_java_like(negative_test)
                            data_point = {
                                "changed_class_name": mutated_clss,
                                "changed_method_name": mutated_method,
                                "code_diff": code_diffs,
                                "abstract_code_diff": mutator,
                                "pos_test_class": pos_test_input,
                                "neg_test_class": neg_test_input
                            }
                            processed_train_add_data.append(data_point)
                        # end for
                    # end for
                    for positive_test in selected_pass_tests:
                        pos_test_input = SubTokenizer.sub_tokenize_java_like(positive_test)
                        sample_size = min(len(lived_test_class), 20)  # predefined maximum living test classes
                        sampled_lived_test_class = sample(lived_test_class, sample_size)
                        for negative_test in sampled_lived_test_class:
                            neg_test_input = SubTokenizer.sub_tokenize_java_like(negative_test)
                            data_point = {
                                "changed_class_name": mutated_clss,
                                "changed_method_name": mutated_method,
                                "code_diff": code_diffs,
                                "abstract_code_diff": mutator,
                                "pos_test_class": pos_test_input,
                                "neg_test_class": neg_test_input
                            }
                            processed_train_add_data.append(data_point)
                        # end for
                    # end for
                # end for
            print(f"In total there are {len(processed_train_data)} data point for training")
            print(f"In total there are {len(processed_train_add_data)} data point for small margin training")
            print(f"In total there are {len(processed_valid_data)} data point for validation")
            shuffle(processed_train_data)
            shuffle(processed_valid_data)
            IOUtils.dump(output_dir / "triplet-train.json", processed_train_data, IOUtils.Format.jsonNoSort)
            IOUtils.dump(output_dir / "triplet-add-train.json", processed_train_add_data, IOUtils.Format.jsonNoSort)
            IOUtils.dump(output_dir / "triplet-valid.json", processed_valid_data, IOUtils.Format.jsonNoSort)
