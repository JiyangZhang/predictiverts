"""
Extract k methods in the test file.
Extract the changed lines for each file with its context, radius of 5 lines.
Get embeddings for k methods and the context, use attention module to get representation
use mlp to predict scores.
"""
import sys
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import random
from seutil import IOUtils, LoggingUtils
import torch.nn as nn
from sklearn.metrics import recall_score, precision_score, f1_score
from collections import defaultdict

from pts.models.rank_model.utils import DiffTestBatchData, DiffPairBatchData, read_data_from_file, hinge_loss, \
    compute_score, find_missing_tests
from pts.models.CodeBertRTS.CodeTransformer import CodeTransformer
from pts.Macros import *
from pts.models.rank_model.TestSelectionModel import threshold_valid_loss
from pts.models.CodeBertRTS.utils import BertBatchData, BertTestBatchData
from pts.main import proj_logs

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")


def aggregate_eval_data(test_data_list):
    """ Helper function to aggregate test data from each sha."""
    sha_2_data = defaultdict(list)
    for test_data_sha in test_data_list:
        sha_2_data[test_data_sha["sha"]].append(test_data_sha)

    print(f"In total {len(sha_2_data)} shas in eval...")

    return sha_2_data

def load_model(model_path):
    """Loads a pretrained model from model_path."""
    print('Loading model from: {}'.format(model_path))
    sys.stdout.flush()
    if torch.cuda.is_available():
        model = torch.load(model_path)
        model.torch_device_name = 'gpu'
        model.cuda()
        for c in model.children():
            c.cuda()
    else:
        model = torch.load(model_path, map_location='cpu')
        model.torch_device_name = 'cpu'
        model.cpu()
        for c in model.children():
            c.cpu()
    return model


def eval_model(model_saving_dir: Path, model_data_dir: Path):
    """Caller function to load model and run model evaluation."""
    model = load_model(model_saving_dir)
    model.mode = "test"
    test_data_dir = model_data_dir / "test.json"
    model.model_results_dir = model_data_dir / "results"
    model.run_evaluation(test_data_file=test_data_dir)

class CodeBert(torch.nn.Module):
    def __init__(self, config_file, model_data_dir):
        super(CodeBert, self).__init__()

        # load config file
        base_config_file = Macros.config_dir / config_file
        self.config = IOUtils.load(base_config_file, IOUtils.Format.jsonPretty)
        self.model_data_dir = model_data_dir
        self.train_add_data_file = None

        self.train_data_file = self.model_data_dir / "train.json"
        self.valid_data_file = self.model_data_dir / "valid.json"
        self.test_data_file = self.model_data_dir / "test.json"
        # load hyper-param
        self.load_hyper_params()

        # set up logging
        logging_file = self.model_data_dir / "model.log"
        LoggingUtils.setup(filename=str(logging_file))
        self.logger = LoggingUtils.get_logger(__name__, LoggingUtils.INFO)
        self.logger.info("===> initializing the model ...")

        # set up sub-modules
        self.diff_test_model = CodeTransformer(self.config, self.hidden_size, self.cross_feature_size, self.encoder_layers,
                                             self.dropout, self.num_heads)
        self.logger.info(repr(self.diff_test_model))  # feature_dim = self.config["cross_feature_dim"] + self.config["test_feature_dim"]
        self.output_layer = nn.Linear(self.last_layer_dim, 1)
        self.rankLoss = nn.MarginRankingLoss(1.0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.sigmoid = nn.Sigmoid()
        # Create dir for the model
        IOUtils.mk_dir(self.model_data_dir / "saved_models")
        IOUtils.mk_dir(self.model_data_dir / "results")
        self.model_save_dir = self.model_data_dir / "saved_models"
        self.model_results_dir = self.model_data_dir / "results"

    def load_hyper_params(self):
        self.learning_rate = self.config["learning_rate"]
        self.embed_size = self.config["embedding_size"]
        self.hidden_size = self.config["hidden_size"]
        self.encoder_layers = self.config["layers"]
        self.max_epoch = self.config["max_epochs"]
        self.patience = self.config["patience"]
        self.dropout = self.config["dropout"]
        self.num_heads = self.config["num_heads"]
        self.cross_feature_size = self.config["cross_feature_size"]
        self.last_layer_dim = self.config["last_layer_dim"]
        if "triplet" in self.config:
            self.margin = self.config["margin_large"]
            self.small_margin = self.config["margin_small"]
        else:
            self.margin = self.config["margin"]
        self.torch_device_name = self.config["device_name"]
        self.diff_features: List = self.config["diff_features"]
        self.test_features: List = self.config["test_features"]
        self.batch_size = self.config["batch_size"]

    def get_device(self):
        """Returns the proper device."""
        if self.torch_device_name == 'gpu':
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def forward(self, batch_data, additional_features=None):
        pos_output, neg_output = self.diff_test_model.forward(batch_data, self.get_device())  # BS, output_size
        # pos_logits = self.sigmoid(self.output_layer(pos_output))
        # neg_logits = self.sigmoid(self.output_layer(neg_output))
        pos_logits = self.sigmoid(pos_output)
        neg_logits = self.sigmoid(neg_output)

        # convert feature vectors to probabilities
        if self.mode == "test":
            return pos_logits, neg_logits
        return pos_logits[:, 0], neg_logits[:, 0]

    def create_pair_batches(self, mode="train", batch_size=32, shuffle=True, dataset=None):
        """
        Create batches for Bert data.
        inputs: test class + test methods (truncated) + changed code with context (truncated)
        """
        if "batch_size" in self.__dict__.keys():
            batch_size = self.batch_size
        batches = []
        if mode == "train":
            objs = IOUtils.load_json_stream(self.train_data_file)
            dataset = []
            for obj in objs:
                dataset.append(obj)
            if shuffle:
                random.shuffle(dataset)
        elif mode == "valid":
            objs = IOUtils.load_json_stream(self.valid_data_file)
            dataset = []
            for obj in objs:
                dataset.append(obj)
            if shuffle:
                random.shuffle(dataset)
        elif mode == "test":
            if not dataset:
                dataset = read_data_from_file(self.test_data_file)

        curr_idx = 0
        while curr_idx < len(dataset):
            start_idx = curr_idx
            end_idx = min(start_idx + batch_size, len(dataset))

            labels = []
            ekstazi_labels = []
            starts_labels = []

            batch_code_diff = []
            batch_pos_test = []
            batch_neg_test = []

            for i in range(start_idx, end_idx):
                # diff features
                # code_diff = []
                batch_code_diff.append(' '.join(dataset[i]["code_diff"]))

                # positive test code

                pos_test_class = ' '.join(dataset[i]["pos_test_class"])
                # pos_test_code = ' '.join(dataset[i]["pos_test_methods"])
                # batch_pos_test.append(pos_test_class + ' ' + pos_test_code)
                batch_pos_test.append(pos_test_class)

                # negative
                neg_test_class = ' '.join(dataset[i]["neg_test_class"])
                # neg_test_code = ' '.join(dataset[i]["neg_test_methods"])
                # batch_neg_test.append(neg_test_class + ' ' + neg_test_code)
                batch_neg_test.append(neg_test_class)
                
                if mode == "test":
                    label = dataset[i]["label"]
                    labels.append(label)
                    ekstazi_label = dataset[i]["ekstazi_label"]
                    ekstazi_labels.append(ekstazi_label)
                    starts_label = dataset[i]["starts_label"]
                    starts_labels.append(starts_label)
            # end for

            # TODO: check the tokens
            batch_code_diff_input = tokenizer(batch_code_diff, padding=True, truncation=True, return_tensors="pt")
            batch_pos_test_input = tokenizer(batch_pos_test, padding=True, truncation=True, return_tensors="pt")
            batch_neg_test_input = tokenizer(batch_neg_test, padding=True, truncation=True, return_tensors="pt")
            
            if mode == "train" or mode == "valid":
                batches.append(
                    BertBatchData(
                        batch_code_diff_input.to(self.get_device()),
                        batch_pos_test_input.to(self.get_device()),
                        batch_neg_test_input.to(self.get_device())
                    )
                )
            elif mode == "test":
                batches.append(
                    BertTestBatchData(batch_code_diff_input,
                                      batch_pos_test_input,
                                      batch_neg_test_input,
                                      torch.tensor(labels, dtype=torch.int64, device=self.get_device()),
                                      torch.tensor(ekstazi_labels, dtype=torch.int64, device=self.get_device()),
                                      torch.tensor(starts_labels, dtype=torch.int64, device=self.get_device())
                                      )
                )
            curr_idx = end_idx
        return batches

    def run_gradient_step(self, batch_data, margin="large"):
        self.optimizer.zero_grad()
        pos_log_prob, neg_log_prob = self.forward(batch_data)
        if margin == "small":
            loss = hinge_loss(pos_log_prob, neg_log_prob, self.small_margin)
        else:
            loss = hinge_loss(pos_log_prob, neg_log_prob, self.margin)
        loss.backward()
        self.optimizer.step()
        return float(loss.cpu())

    def run_train(self):
        """Train the prediction model"""
        self.logger.info("===> start to train the model ... ")
        # load train and valid data
        valid_batches = self.create_pair_batches(mode="valid")
        if self.torch_device_name == "gpu":
            self.cuda()
        best_loss = float('inf')
        best_prec = 0
        patience_tally = 5

        # start training
        for epoch in range(self.max_epoch):
            if patience_tally > self.patience or best_prec > 0.9:
                print('Terminating')
                self.logger.info("===> terminating the training ...")
                break
            self.train()
            train_batches = self.create_pair_batches(mode="train")

            train_loss = 0

            total_iter = len(train_batches)
            for batch_iter, batch_data in enumerate(train_batches):
                step_loss = self.run_gradient_step(batch_data)
                train_loss += step_loss
                print(f"Training loss step {batch_iter+1}/{total_iter}: {step_loss}")



            # end if
            # do validation at each epoch
            self.eval()
            validation_loss = 0
            validation_predictions = []
            validation_labels = []
            with torch.no_grad():
                for batch_data in valid_batches:
                    # validation loss
                    pos_score, neg_score = self.forward(batch_data)
                    valid_loss = hinge_loss(pos_score, neg_score, 0.2)
                    validation_loss += float(valid_loss.cpu())
                    scores = torch.cat((pos_score.unsqueeze(1), neg_score.unsqueeze(1)), -1)  # BS, 2
                    # labels, preds = pairwise_rank_loss(scores)
                    labels, preds = threshold_valid_loss(scores)
                    validation_labels.extend(labels)
                    validation_predictions.extend(preds)

            # validation_precision = (len(validation_labels) - sum(validation_predictions)) / len(validation_labels)
            validation_precision, validation_recall, validation_f1 = compute_score(
                validation_predictions, validation_labels, verbose=False)

            if validation_f1 > best_prec:
                torch.save(self, self.model_save_dir / "best_model")
                saved = True
                best_prec = validation_f1
                patience_tally = 0
            else:
                saved = False
                patience_tally += 1
            print('Epoch: {}'.format(epoch + 1))
            print('Training loss: {}'.format(train_loss / len(train_batches)))
            print('Validation loss: {}'.format(validation_loss))
            print(
                f"Validation precision is {validation_precision}, recall is {validation_recall}, f1 is {validation_f1}.")
            if saved:
                print('Saved')
            print('-----------------------------------')

    def run_evaluation(self, select_rate=None, test_data_file=None, threshold=0.5, save=True, return_result=False,
                       subset=None):
        """Predicts labels for the (diff, test) pairs."""
        self.eval()
        self.mode = "test"
        if test_data_file:
            self.test_data_file = test_data_file

        test_data_list = read_data_from_file(self.test_data_file)

        sha_2_data = aggregate_eval_data(test_data_list)

        all_predictions = []
        all_labels = []
        all_ekstazi_labels = []
        all_starts_labels = []
        sha_2_time = []

        selected_failed_tests_num = 0
        missed_failed_tests_num = 0

        out_of_scope_test = 0
        in_scope_test = 0

        recall_per_sha = []
        precision_per_sha = []
        f1_per_sha = []
        eks_recall_per_sha = []
        starts_recall_per_sha = []
        starts_subset_recall_per_sha = []
        ekstazi_subset_recall_per_sha = []

        # to save for further process
        prediction_per_sha = []

        for s, s_data in sha_2_data.items():
            test_batches = self.create_pair_batches(mode="test", dataset=s_data)

            num_of_candidate_tests = s_data[0]["num_test_class"]
            num_of_changed_files = s_data[0]["num_changed_files"]

            self.logger.info(f"Number of changed files in {s} is {num_of_changed_files}."
                             f"Number of candidate test classes is {num_of_candidate_tests}")

            s_pred_scores = []
            s_starts_labels = []
            s_labels = []
            s_ekstazi_labels = []

            # Start to do prediction in this SHA
            with torch.no_grad():
                start_time = time.time()
                for b, batch_data in enumerate(test_batches):
                    print(f"Testing SHA: {s}")
                    sys.stdout.flush()
                    pos_score, _ = self.forward(batch_data)
                    s_pred_scores.extend([element.item() for element in pos_score.flatten()])
                    s_labels.extend([element.item() for element in batch_data.label.flatten()])
                    s_starts_labels.extend([element.item() for element in batch_data.starts_label.flatten()])
                    s_ekstazi_labels.extend([element.item() for element in batch_data.ekstazi_label.flatten()])
                # end for
                run_time = time.time() - start_time
                self.logger.info(f"Running time to do prediction is {run_time} seconds.")
                sha_2_time.append(run_time)
            # end with
            prediction_scores = np.zeros(int(num_of_candidate_tests))
            for i in range(0, len(s_pred_scores), num_of_candidate_tests):
                tmp = np.array(s_pred_scores[i: i + num_of_candidate_tests])
                prediction_scores = np.maximum(prediction_scores, tmp)
            # num_files = len(s_pred_scores) / int(num_of_candidate_tests)

            # prediction_scores /= num_of_changed_files
            preds = np.zeros(num_of_candidate_tests)
            if select_rate is None:
                preds[prediction_scores >= threshold] = 1
            else:
                select_size = int(select_rate * num_of_candidate_tests)
                if select_size > 0:
                    preds[prediction_scores.argsort()[-select_size:]] = 1
            labels = s_labels[:num_of_candidate_tests]
            ekstazi_labels = s_ekstazi_labels[:num_of_candidate_tests]
            starts_labels = s_starts_labels[:num_of_candidate_tests]
            preds = preds.tolist()
            # ad-hoc processing
            modified_test_class = [d["changed_class_name"] for d in s_data if "test" in d["changed_class_name"]]
            test_index = []
            for t in modified_test_class:
                test_index.extend(
                    [i for i, d in enumerate(s_data[:num_of_candidate_tests]) if d["pos_test_class"] == t])

            # To see how many failed are detected and how many are not
            for i, p in enumerate(preds):
                if p > 0 and labels[i] > 0:
                    selected_failed_tests_num += 1
                elif labels[i] > 0:
                    missed_failed_tests_num += 1
            # end for
            p, r, f = compute_score(predicted_labels=preds, gold_labels=labels)
            if r < 1:
                project_name = str(test_data_file).split('/')[-3]
                for k in proj_logs:
                    if k.split('_')[1] == project_name:
                        project = k
                t1, t2 = find_missing_tests(preds, labels, s_data, project)
                out_of_scope_test += t1
                in_scope_test += t2

            if select_rate is not None:
                # Add the results of models selection intersecting STARTS
                model_starts_preds = []
                starts_selected_labels = []
                for i in range(len(starts_labels)):
                    if starts_labels[i] == 1:
                        starts_selected_labels.append(labels[i])
                        model_starts_preds.append(prediction_scores[i])
                # end for
                select_size = int(select_rate * num_of_candidate_tests)
                model_starts_preds = np.array(model_starts_preds)
                starts_subset_preds = np.zeros(len(starts_selected_labels))
                if select_size > 0:
                    starts_subset_preds[model_starts_preds.argsort()[-select_size:]] = 1
                # end if
                if sum(starts_selected_labels) == 0:
                    pass
                else:
                    starts_subset_recall_per_sha.append(recall_score(starts_selected_labels, starts_subset_preds))

                # Add the results of models selection intersecting Ekstazi
                model_ekstazi_preds = []
                ekstazi_selected_labels = []
                for i in range(len(ekstazi_labels)):
                    if ekstazi_labels[i] == 1:
                        ekstazi_selected_labels.append(labels[i])
                        model_ekstazi_preds.append(prediction_scores[i])
                # end for
                select_size = int(select_rate * num_of_candidate_tests)
                model_ekstazi_preds = np.array(model_ekstazi_preds)
                ekstazi_subset_preds = np.zeros(len(ekstazi_selected_labels))
                if select_size > 0:
                    ekstazi_subset_preds[model_ekstazi_preds.argsort()[-select_size:]] = 1
                # end if
                if sum(ekstazi_selected_labels) == 0:
                    pass
                else:
                    ekstazi_subset_recall_per_sha.append(recall_score(ekstazi_selected_labels, ekstazi_subset_preds))

            recall_per_sha.append(recall_score(labels, preds))
            precision_per_sha.append(precision_score(labels, preds))
            eks_recall_per_sha.append((recall_score(labels, ekstazi_labels)))
            starts_recall_per_sha.append((recall_score(labels, starts_labels)))
            f1_per_sha.append(f1_score(labels, preds))

            all_predictions.extend(preds)
            all_ekstazi_labels.extend(ekstazi_labels)
            all_starts_labels.extend(starts_labels)
            all_labels.extend(labels)

            prediction_per_sha.append({
                "commit": s,
                "prediction_scores": prediction_scores.tolist(),
                "Ekstazi_labels": ekstazi_labels,
                "STARTS_labels": starts_labels,
                "labels": labels
            })

        if subset == "STARTS":
            return sum(starts_subset_recall_per_sha) / len(starts_subset_recall_per_sha)
        if subset == "Ekstazi":
            return sum(ekstazi_subset_recall_per_sha) / len(ekstazi_subset_recall_per_sha)

        compute_score(predicted_labels=all_predictions, gold_labels=all_labels)
        prec = sum(precision_per_sha) / len(precision_per_sha)
        rec = sum(recall_per_sha) / len(recall_per_sha)
        f1 = sum(f1_per_sha) / len(f1_per_sha)

        ek_prec = 0
        ek_rec = sum(eks_recall_per_sha) / len(eks_recall_per_sha)
        ek_f1 = 0

        sts_prec = 0
        sts_rec = sum(starts_recall_per_sha) / len(starts_recall_per_sha)
        sts_f1 = 0

        # average selected test class
        model_sel_num = sum(all_predictions)
        ek_sel_num = sum(all_ekstazi_labels)
        sts_sel_num = sum(all_starts_labels)
        total_num = len(all_predictions)
        if in_scope_test + out_of_scope_test == 0:
            pct = -1
        else:
            pct = out_of_scope_test / (in_scope_test + out_of_scope_test)
        result = {
            "precision": 100 * prec,
            "recall": 100 * rec,
            "f1": f1,
            "selected_pct": float(model_sel_num) / total_num,
            "ekstazi_precision": 100 * ek_prec,
            "ekstazi_recall": 100 * ek_rec,
            "ekstazi_f1": 100 * ek_f1,
            "ekstazi_selected_pct": float(ek_sel_num) / total_num,
            "starts_precision": 100 * sts_prec,
            "starts_recall": 100 * sts_rec,
            "starts_f1": 100 * sts_f1,
            "starts_selected_pct": float(sts_sel_num) / total_num,
            "total_missed_failed_tests": missed_failed_tests_num,
            "total_selected_failed_tests": selected_failed_tests_num,
            "pct_newly_add_missed_failed_tests": pct
        }

        if save:  # will change to per_file later if we use other approach to do prediction
            IOUtils.mk_dir(self.model_results_dir / "test-output")
            IOUtils.dump(self.model_results_dir / "test-output" / "per-file-result.json", result,
                         IOUtils.Format.jsonPretty)
            IOUtils.dump(self.model_results_dir / "running-time-per-sha.json", sha_2_time)
            print(result)
            IOUtils.dump(self.model_results_dir / f"per-sha-result.json", prediction_per_sha, IOUtils.Format.json)
        elif return_result:
            return result
        else:
            return rec
