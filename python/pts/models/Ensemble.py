from seutil import BashUtils, IOUtils
from pts.main import Macros

from pathlib import Path
from typing import *
from operator import add

class Ensemble:

    IR_BASELINES = ["BM25Baseline", "TFIDFBaseline"]

    def __init__(self, sub_models: List[str]):
        self.sub_models = sub_models

    def max_abs_normalize(self, scores: List):
        """Normalize the scores"""
        max_score = max(scores)
        return [s/max_score for s in scores]

    def ensembling(self, project: str):
        """Combine the output scores of different sub models to make the final prediction.
        Use the results in modelResults dir of sub-models and store in the model_data_dir.
        You need to move the results from the model_data_dir to modelResults.
        """
        ensemble_model_name = "-".join(self.sub_models)
        IOUtils.mk_dir(Macros.results_dir / "modelResults" / project.split('_')[1] / ensemble_model_name)
        ensemble_model_results = IOUtils.load(Macros.results_dir / "modelResults" / project.split('_')[1] /
                                              self.sub_models[0] / "per-sha-result.json")
        for sha_mutant in ensemble_model_results:
            sha_mutant["prediction_scores"] = [0 for _ in range(len(sha_mutant["prediction_scores"]))]

        for m in self.sub_models:
            model_results_file = Macros.results_dir / "modelResults" / project.split('_')[1] / m / "per-sha-result.json"
            model_results_list = IOUtils.load(model_results_file)

            for i, sha_mutant in enumerate(model_results_list):
                if ensemble_model_results[i]["commit"] != sha_mutant["commit"]:
                    raise Exception(f"The sha-mutant does not match, please check, models are {self.sub_models} at {project}.")       
                accumulate_scores = ensemble_model_results[i]["prediction_scores"]
                model_scores = sha_mutant["prediction_scores"]
                if m in self.IR_BASELINES:
                    model_scores = self.max_abs_normalize(model_scores)
                new_accumulated_scores = list(map(add, accumulate_scores, model_scores))
                ensemble_model_results[i]["prediction_scores"] = new_accumulated_scores
            # end for
        # end for
        for res in ensemble_model_results:
            normalized_scores = [score/len(self.sub_models) for score in res["prediction_scores"]]
            res["prediction_scores"] = normalized_scores
        # end for
        IOUtils.mk_dir(Macros.model_data_dir / "rank-model" / project.split('_')[1] / ensemble_model_name / "results")
        IOUtils.dump(Macros.model_data_dir / "rank-model" / project.split('_')[1] / ensemble_model_name / "results" / "per-sha-result.json", ensemble_model_results)
