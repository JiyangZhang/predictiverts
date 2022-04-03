import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from pts.models.CodeBertRTS.Pooling import Pooling
from pts.models.hybrid.FC_layer import FC_layer

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")


class CodeTransformer(nn.Module):

    def __init__(self, config, hidden_size=768, output_size=32, num_layers=1,
                 dropout=0.4, num_heads=16):
        super(CodeTransformer, self).__init__()

        self.config = config
        self.pooling = Pooling(hidden_size, pooling_mode="mean")
        self.fc = FC_layer(hidden_size*2, output_size, 1, dropout)

    def forward(self, batch_data, device):
        # Extract embeddings
        if device == torch.device('cuda'):
            CodeBert = AutoModel.from_pretrained("microsoft/codebert-base").to('cuda')
        else:
            CodeBert = AutoModel.from_pretrained("microsoft/codebert-base")
        with torch.no_grad():

            code_outputs = CodeBert(**batch_data.code_seqs)
            diff_code_embed = code_outputs.last_hidden_state

            diff_embedding = self.pooling.forward({"token_embeddings": diff_code_embed,
                                                   "cls_token_embeddings": code_outputs.pooler_output,
                                                   "attention_mask": batch_data.code_seqs["attention_mask"]})

            pos_test_outputs = CodeBert(**batch_data.pos_test_seqs)
            pos_test_embed = pos_test_outputs.last_hidden_state

            pos_test_embedding = self.pooling.forward({"token_embeddings": pos_test_embed,
                                                       "cls_token_embeddings": pos_test_outputs.pooler_output,
                                                       "attention_mask": batch_data.pos_test_seqs["attention_mask"]})

            neg_test_outputs = CodeBert(**batch_data.neg_test_seqs)
            neg_test_embed = neg_test_outputs.last_hidden_state
            neg_test_embedding = self.pooling.forward({"token_embeddings": neg_test_embed,
                                                       "cls_token_embeddings": neg_test_outputs.pooler_output,
                                                       "attention_mask": batch_data.neg_test_seqs["attention_mask"]})

        # FC layer

        pos_final_state = torch.cat((diff_embedding, pos_test_embedding), 1)
        pos_output = self.fc.forward(pos_final_state)
        neg_final_state = torch.cat((diff_embedding, neg_test_embedding), 1)
        neg_output = self.fc.forward(neg_final_state)

        return pos_output, neg_output
