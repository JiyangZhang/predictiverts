from typing import *
import random
from os import listdir
from pathlib import Path
from seutil import LoggingUtils, IOUtils, BashUtils
import traceback

from pts.Environment import Environment
from pts.Macros import Macros
from pts.processor.data_utils.SubTokenizer import SubTokenizer


class MatchModelProcessor:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self):
        self.data = list()
        self.output_dir = Macros.data_dir / "model-data"
        pass

    def process(self, default=True):
        proj_names: List = listdir(Macros.repos_results_dir)  # a list of project name
        proj_dirs: List = [Path(Macros.repos_results_dir / proj) for proj in proj_names]  # a list of project dirs
        IOUtils.mk_dir(self.output_dir/"match-model")
        for proj, proj_dir in zip(proj_names, proj_dirs):
            try:
                if default:
                    reports = IOUtils.load(proj_dir / f"{proj}-default-mutation-report.json")
                else:
                    reports: List[dict] = IOUtils.load(proj_dir / f"{proj}-all-mutation-report.json")
                self.logger.info(f"Currenlty working on {proj}, in total {len(reports)} mutations")
                for mut in reports:
                    mutated_method = mut["mutatedClass"] + "." + mut["mutatedMethod"]
                    if isinstance(mut["succeedingTests"], list):
                        for t in mut["succeedingTests"]:
                            test = t[1] if t[1] != "" else t[0]
                            SubTokenizer.sub_tokenize_java_like(" ".join(mutated_method.split(".")))
                            self.data.append(
                                {
                                    "mutated_method": SubTokenizer.sub_tokenize_java_like(" ".join(mutated_method.split("."))),
                                    "test": SubTokenizer.sub_tokenize_java_like(" ".join(test.split("."))),
                                    "label": 0
                                }
                            )
                        # end for
                    # end if
                    if isinstance(mut["killingTests"], list):
                        for t in mut["killingTests"]:
                            test = t[1] if t[1] != "" else t[0]
                            self.data.append(
                                {
                                    "mutated_method": SubTokenizer.sub_tokenize_java_like(" ".join(mutated_method.split("."))),
                                    "test": SubTokenizer.sub_tokenize_java_like(" ".join(test.split("."))),
                                    "label": 1
                                }
                            )
                        # end for
                    # end if
            except FileNotFoundError:
                print("Can not find the file.")
                self.logger.warning(f"Collection data for project {proj} failed, error was: {traceback.format_exc()}")
                continue
            except RuntimeError:
                print(RuntimeError)
                self.logger.warning(f"Collection data for project {proj} failed, error was: {traceback.format_exc()}")
                continue
        # end for
        output_dir = self.output_dir/"match-model"
        IOUtils.dump(output_dir/f"data.json", self.data, IOUtils.Format.jsonPretty)
        
