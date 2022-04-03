from pathlib import Path
from typing import List
import torch
import numpy as np
from seutil import IOUtils
from operator import add, mul

from pts.Macros import *
from pts.models.rank_model.TestSelectionModel import load_model, threshold_valid_loss
from pts.models.rank_model.model_runner import max_abs_normalize
from pts.models.CodeBertRTS.CodeBert import aggregate_eval_data


def get_loss(labels: List, preds: List):
    """Return the loss of prediction"""
    pred = np.array(preds)
    label = np.array(labels)
    m = len(preds)
    loss = np.zeros((m, 1))
    loss[pred != label] = 1
    sum_loss = np.average(loss).item()
    return sum_loss


class AdamBoost:

    IR_BASELINES = ["BM25Baseline", "TFIDFBaseline"]

    def __init__(self, model_data_dir: Path, sub_models: List[str]):
        self.sub_models = sub_models
        self.model_data_dir = model_data_dir
        return

    def AdaBoost_run(self, project: str):
        ensembled_models = self.AdaBoost_train()
        self.AdaBoost_eval(ensembled_models, project)

    def AdaBoost_train(self):
        """Run all the sub models to get the training loss, which will be used as the weight to combine the models."""
        ensembled_models = []
        for model in self.sub_models:
            loaded_model = load_model(self.model_data_dir/model/"saved_models"/"best_model")

            train_batches = loaded_model.create_pair_batches(mode="train")
            loaded_model.eval()

            predictions = []
            labels = []
            with torch.no_grad():
                for batch_data in train_batches:
                    pos_score, neg_score = loaded_model.forward(batch_data)
                    scores = torch.cat((pos_score.unsqueeze(1), neg_score.unsqueeze(1)), -1)  # BS, 2
                    label, pred = threshold_valid_loss(scores)
                    labels.extend(label)
                    predictions.extend(pred)
            loss = get_loss(labels, predictions)
            weight = 0.5*np.log(1.0/loss -1)
            ensembled_models.append((model, weight))

        return ensembled_models

    def AdaBoost_eval(self, ensembled_models: List, project: str):
        """Predicts labels for the (diff, test) pairs."""
        ensemble_model_name = "boosting"
        # Initialize the results
        ensemble_model_results = IOUtils.load(Macros.results_dir / "modelResults" / project.split('_')[1] /
                                              self.sub_models[0] / "per-sha-result.json")
        for sha_mutant in ensemble_model_results:
            sha_mutant["prediction_scores"] = [0 for _ in range(len(sha_mutant["prediction_scores"]))]

        # ensemble the models results
        for model, weight in ensembled_models:
            model_results_file = Macros.results_dir / "modelResults" / project.split('_')[1] / model / "per-sha-result.json"
            model_results_list = IOUtils.load(model_results_file)
            for i, sha_mutant in enumerate(model_results_list):
                if ensemble_model_results[i]["commit"] != sha_mutant["commit"]:
                    raise Exception("The sha-mutant does not match, please check.")
                accumulate_scores = ensemble_model_results[i]["prediction_scores"]
                model_scores = sha_mutant["prediction_scores"]
                if model in self.IR_BASELINES:
                    model_scores = max_abs_normalize(model_scores)
                model_scores = list(map(lambda x: x*weight, model_scores))
                new_accumulated_scores = list(map(add, accumulate_scores, model_scores))
                ensemble_model_results[i]["prediction_scores"] = new_accumulated_scores
            # end for
        # end for

        # Normalize the scores
        for res in ensemble_model_results:
            normalized_scores = [score / len(ensembled_models) for score in res["prediction_scores"]]
            res["prediction_scores"] = normalized_scores
        # end for

        IOUtils.mk_dir(Macros.model_data_dir / "rank-model" / project.split('_')[1] / ensemble_model_name / "results")
        IOUtils.dump(Macros.model_data_dir / "rank-model" / project.split('_')[
            1] / ensemble_model_name / "results" / "per-sha-result.json", ensemble_model_results)
        IOUtils.dump(Macros.model_data_dir / "rank-model" / project.split('_')[1] / ensemble_model_name / "boosting-model.json",
                     ensembled_models, IOUtils.Format.json)

