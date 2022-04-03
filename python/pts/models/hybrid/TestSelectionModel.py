import random
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import time
import ipdb
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from seutil import IOUtils, LoggingUtils
from collections import Counter

from pts.Macros import Macros
from pts.models.hybrid.CodeEmbeddingStore import CodeEmbeddingStore
from pts.models.hybrid.dataloader import CodeClassDataLoader
from pts.models.hybrid.DiffTestModel import DiffTestModel
from pts.processor.data_utils.SubTokenizer import SubTokenizer
from pts.models.hybrid.utils import DiffTestBatchData, DiffPairBatchData, read_data_from_file, hinge_loss, compute_score


class TestSelectionModel(nn.Module):
    """Predict confidence score for a given code diff and test."""

    def __init__(self, config_file, model_data_dir: Path):
        super(TestSelectionModel, self).__init__()
        # load config file
        base_config_file = Macros.config_dir / config_file
        self.config = IOUtils.load(base_config_file, IOUtils.Format.jsonPretty)
        self.model_data_dir = model_data_dir
        self.train_data_file = self.model_data_dir / "train.json"
        self.valid_data_file = self.model_data_dir / "valid.json"
        self.test_data_file = self.model_data_dir / "test.json"
        # load hyper-param
        self.load_hyper_params()
        # create vocab
        self.create_vocab()
        # set up sub-modules
        self.diff_test_model = DiffTestModel(self.config, self.embed_size, self.embedding_store, self.hidden_size,
                                             self.cross_feature_size, self.encoder_layers, self.dropout, self.num_heads)
        # set up logging
        logging_file = self.model_data_dir / "model.log"
        LoggingUtils.setup(filename=str(logging_file))
        self.logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG)
        # feature_dim = self.config["cross_feature_dim"] + self.config["test_feature_dim"]
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
        # TODO self.vocab_size = self.config[""]
        self.torch_device_name = self.config["device_name"]

    def create_vocab(self):
        self.code_token_counter = Counter()
        code_lengths = []

        train_data = IOUtils.load_json_stream(self.train_data_file)
        valid_data = IOUtils.load(self.valid_data_file)

        print("===> creating vocabs ...")
        for dt in train_data:
            diff_tokens = dt["code_diff"]
            self.code_token_counter.update(diff_tokens)
            code_lengths.append(len(diff_tokens))

            pos_test_tokens = dt["pos_test_code"]
            self.code_token_counter.update(pos_test_tokens)
            code_lengths.append(len(pos_test_tokens))

            neg_test_tokens = dt["neg_test_code"]
            self.code_token_counter.update(neg_test_tokens)
            code_lengths.append(len(neg_test_tokens))
        # end for

        code_counts = np.asarray(sorted(self.code_token_counter.values()))
        code_threshold = int(np.percentile(code_counts, self.config["vocab_cut_off_pct"])) + 1
        self.max_code_length = int(np.percentile(np.asarray(sorted(code_lengths)),
                                                 self.config["length_cut_off_pct"]))
        self.embedding_store = CodeEmbeddingStore(code_threshold, self.config["embedding_size"],
                                                  self.code_token_counter,
                                                  self.config["dropout"])

    def get_device(self):
        """Returns the proper device."""
        if self.torch_device_name == 'gpu':
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def forward(self, batch_data, additional_features=None):
        pos_output, neg_output = self.diff_test_model.forward(batch_data, self.get_device())  # BS, output_size
        if additional_features:
            # TODO: additional features
            pass
        pos_logits = self.sigmoid(self.output_layer(pos_output))
        neg_logits = self.sigmoid(self.output_layer(neg_output))
        # convert feature vectors to probabilities
        if self.mode == "test":
            return pos_logits, neg_logits
        # Try: without log_softmax
        # scores = torch.cat((pos_logits, neg_logits), -1)  # BS, 2
        # log_prob = torch.nn.functional.log_softmax(scores, dim=-1)
        # pos_log_prob, neg_log_prob = log_prob[:, 0], log_prob[:, 1]
        return pos_logits[:,0], neg_logits[:,0]

    def create_pair_batches(self, mode="train", batch_size=32, shuffle=True, dataset=None):
        batches = []
        if mode == "train":
            dataset = read_data_from_file(self.train_data_file)
            if shuffle:
                random.shuffle(dataset)
        elif mode == "valid":
            dataset = read_data_from_file(self.valid_data_file)
            if shuffle:
                random.shuffle(dataset)
        elif mode == "test":
            if not dataset:
                dataset = read_data_from_file(self.test_data_file)

        curr_idx = 0
        while curr_idx < len(dataset):
            start_idx = curr_idx
            end_idx = min(start_idx + batch_size, len(dataset))

            code_diff_token_ids = []  # length: batch size
            code_diff_lengths = []
            pos_test_token_ids = []
            pos_test_lengths = []
            neg_test_token_ids = []
            neg_test_lengths = []

            labels = []

            for i in range(start_idx, end_idx):
                # code diff
                code_diff = dataset[i]["code_diff"]
                code_diff_ids = self.embedding_store.get_padded_code_ids(code_diff, self.max_code_length)
                code_diff_length = min(len(code_diff), self.max_code_length)
                code_diff_token_ids.append(code_diff_ids)
                code_diff_lengths.append(code_diff_length)
                # test code
                pos_test_code = dataset[i]["pos_test_code"]
                pos_test_code_ids = self.embedding_store.get_padded_code_ids(pos_test_code, self.max_code_length)
                pos_test_code_length = min(len(pos_test_code), self.max_code_length)
                pos_test_token_ids.append(pos_test_code_ids)
                pos_test_lengths.append(pos_test_code_length)

                neg_test_code = dataset[i]["neg_test_code"]
                neg_test_code_ids = self.embedding_store.get_padded_code_ids(neg_test_code, self.max_code_length)
                neg_test_code_length = min(len(neg_test_code), self.max_code_length)
                neg_test_token_ids.append(neg_test_code_ids)
                neg_test_lengths.append(neg_test_code_length)

                if mode == "test":
                    label = dataset[i]["label"]
                    labels.append(label)
            if mode == "train" or mode == "valid":
                batches.append(
                    DiffPairBatchData(torch.tensor(code_diff_token_ids, dtype=torch.int64, device=self.get_device()),
                                      torch.tensor(code_diff_lengths, dtype=torch.int64, device=self.get_device()),
                                      torch.tensor(pos_test_token_ids, dtype=torch.int64, device=self.get_device()),
                                      torch.tensor(pos_test_lengths, dtype=torch.int64, device=self.get_device()),
                                      torch.tensor(neg_test_token_ids, dtype=torch.int64, device=self.get_device()),
                                      torch.tensor(neg_test_lengths, dtype=torch.int64, device=self.get_device()))

                    )
            elif mode == "test":
                batches.append(
                    DiffTestBatchData(torch.tensor(code_diff_token_ids, dtype=torch.int64, device=self.get_device()),
                                      torch.tensor(code_diff_lengths, dtype=torch.int64, device=self.get_device()),
                                      torch.tensor(pos_test_token_ids, dtype=torch.int64, device=self.get_device()),
                                      torch.tensor(pos_test_lengths, dtype=torch.int64, device=self.get_device()),
                                      torch.tensor(neg_test_token_ids, dtype=torch.int64, device=self.get_device()),
                                      torch.tensor(neg_test_lengths, dtype=torch.int64, device=self.get_device()),
                                      torch.tensor(labels, dtype=torch.int64, device=self.get_device()))

                )
            curr_idx = end_idx
        return batches

    def run_gradient_step(self, batch_data):
        self.optimizer.zero_grad()
        pos_log_prob, neg_log_prob = self.forward(batch_data)
        loss = hinge_loss(pos_log_prob, neg_log_prob, 0.3)
        loss.backward()
        self.optimizer.step()
        return float(loss.cpu())

    def run_train(self):
        """Train the prediction model"""
        # load train and valid data
        valid_batches = self.create_pair_batches(mode="valid")
        if self.torch_device_name == "gpu":
            self.cuda()
        best_loss = float('inf')
        best_prec = 0
        patience_tally = 0

        # start training
        for epoch in range(self.max_epoch):
            if patience_tally > self.patience:
                print('Terminating')
                break
            self.train()
            train_batches = self.create_pair_batches(mode="train")

            train_loss = 0

            for batch_iter, batch_data in enumerate(train_batches):
                train_loss += self.run_gradient_step(batch_data)
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
                    validation_predictions.extend(scores.argmax(-1).tolist())
                    validation_labels.extend([0 for _ in range(scores.shape[0])])
            # ipdb.set_trace()
            validation_loss = validation_loss / len(valid_batches)
            validation_precision = (len(validation_labels) - sum(validation_predictions)) / len(validation_labels)
            # validation_precision, validation_recall, validation_f1 = compute_score(
            #     validation_predictions, validation_labels, verbose=False)
            
            if validation_precision > best_prec:
                torch.save(self, self.model_save_dir/"best_model")
                saved = True
                best_prec = validation_precision
                patience_tally = 0
            else:
                saved = False
                patience_tally += 1
            print('Epoch: {}'.format(epoch + 1))
            print('Training loss: {}'.format(train_loss / len(train_batches)))
            print('Validation loss: {}'.format(validation_loss))
            print(f"Validation precision is {validation_precision}, best precision is {best_prec}.")
            if saved:
                print('Saved')
            print('-----------------------------------')

    def run_evaluation(self, test_data_file=None):
        """Predicts labels for the (diff, test) pairs."""
        self.eval()
        self.mode = "test"
        if test_data_file:
            self.test_data_file = test_data_file

        test_data_list = read_data_from_file(self.test_data_file)

        sha_2_data = defaultdict(list)
        for test_data_sha in test_data_list:
            if "sha" not in test_data_sha:
                print(test_data_sha)
            sha_2_data[test_data_sha["sha"]].append(test_data_sha)

        all_predictions = []
        all_labels = []
        
        for s, s_data in sha_2_data.items():
            test_batches = self.create_pair_batches(mode="test", dataset=s_data)
            num_of_changed_files = len(set(["".join(t["code_diff"]) for t in s_data]))
            print(f"Number of changed files: {num_of_changed_files}")

            s_pred_scores = []
            s_labels = []

            with torch.no_grad():
                for b, batch_data in enumerate(test_batches):
                    print(f"Testing SHA: {s}")
                    sys.stdout.flush()
                    pos_score, _ = self.forward(batch_data)
                    s_pred_scores.extend([element.item() for element in pos_score.flatten()])
                    s_labels.extend([element.item() for element in batch_data.label.flatten()])
                # end for
            # end with
            num_of_candidate_tests = int(len(s_pred_scores) / num_of_changed_files)

            print(f"Num of tests is {num_of_candidate_tests}.")
            prediction_scores = np.zeros(int(num_of_candidate_tests))
            for i in range(0, len(s_pred_scores), num_of_candidate_tests):
                tmp = np.array(s_pred_scores[i: i+num_of_candidate_tests])
                prediction_scores += tmp
            preds = np.zeros(num_of_candidate_tests)
            preds[prediction_scores.argsort()[-4:]] = 1
            # ipdb.set_trace()
            labels = s_labels[:num_of_candidate_tests]
            preds.tolist()
            
            compute_score(predicted_labels=preds, gold_labels=labels)
            all_predictions.extend(preds)
            all_labels.extend(labels)

        compute_score(predicted_labels=all_predictions, gold_labels=all_labels)
        
    def create_batches(self, mode="train", batch_size=32, shuffle=True):
        batches = []
        if mode == "train":
            dataset = self.train_data
            if shuffle:
                random.shuffle(dataset)
        elif mode == "valid":
            dataset = self.valid_data
            if shuffle:
                random.shuffle(dataset)
        elif mode == "test":
            dataset = self.test_data

        curr_idx = 0
        while curr_idx < len(dataset):
            start_idx = curr_idx
            end_idx = min(start_idx + batch_size, len(dataset))

            code_diff_token_ids = []  # length: batch size
            code_diff_lengths = []
            test_token_ids = []
            test_lengths = []

            labels = []

            for i in range(start_idx, end_idx):
                # code diff
                code_diff = dataset[i]["code_diff"]
                code_diff_ids = self.embedding_store.get_padded_code_ids(code_diff, self.max_code_length)
                code_diff_length = min(len(code_diff), self.max_code_length)
                code_diff_token_ids.append(code_diff_ids)
                code_diff_lengths.append(code_diff_length)
                # test code
                test_code = dataset[i]["test_code"]
                test_code_ids = self.embedding_store.get_padded_code_ids(test_code, self.max_code_length)
                test_code_length = min(len(test_code), self.max_code_length)
                test_token_ids.append(test_code_ids)
                test_lengths.append(test_code_length)
                labels.append(dataset[i]["label"])

            batches.append(
                DiffTestBatchData(torch.tensor(code_diff_token_ids, dtype=torch.int64, device=self.get_device()),
                                  torch.tensor(code_diff_lengths, dtype=torch.int64, device=self.get_device()),
                                  torch.tensor(test_token_ids, dtype=torch.int64, device=self.get_device()),
                                  torch.tensor(test_lengths, dtype=torch.int64, device=self.get_device()),
                                  torch.tensor(labels, dtype=torch.int64, device=self.get_device()))
            )
        return batches
