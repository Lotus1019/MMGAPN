
import copy
# from pytorch_transformers import BertPreTrainedModel, RobertaModel
from optimal.transformers.modeling_bert import BertPreTrainedModel
from optimal.transformers.modeling_roberta import RobertaAdapterModel

from torch import nn
import torch.nn.functional as F

from optimal.transformers.modeling_roberta import (
    RobertaConfig,
    RobertaForSequenceClassification,
)
from data_kb_util import *

from NetModule import SlotSelfAttention,SlotAttentionLayer,PositionwiseFeedForward,Question_Answer_InteractiveAttention,MultiHeadAttention,MLP

from copy import deepcopy
from optimal.utils import transform_graph_geometric

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
}

ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-config.json",
}





def tailr_loss(lprobs, target, epsilon, min_weight, gamma, probs_model, ignore_index = None, reduce = True):
    lprobs = torch.nn.functional.softmax(lprobs, dim = -1).log()
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim = -1, index = target)
    smooth_loss = -lprobs.sum(dim = -1, keepdim = True)

    weight_theta_hat = probs_model.gather(dim = -1, index = target)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)

    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        raise NotImplementedError

    with torch.no_grad():
        weight_theta_hat = (weight_theta_hat.log() - (gamma + (1 - gamma) * weight_theta_hat).log()).exp()
        weight_theta_hat = torch.clamp(weight_theta_hat, min = min_weight, max = 1.0)

    tailr_loss = weight_theta_hat * nll_loss

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        tailr_loss = tailr_loss.sum()
    eps_i = epsilon / (lprobs.size(-1))
    loss = (1. - epsilon) * tailr_loss + eps_i * smooth_loss
    return loss, nll_loss


def mean_pooling(model_output, attention_mask):
    """
    Mean Pooling - Take attention mask into account for correct averaging.
    model_output: tensor of shape (b, l, d_k_original)
    attention_mask: tensor of shape (b, l)
    returns: tensor of shape (b, d_k_original)
    """

    token_embeddings = model_output #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def max_pooling(model_output, attention_mask):
    """
    Max Pooling - Take the max value over time for every dimension.
    model_output: tensor of shape (b, l, d_k_original)
    attention_mask: tensor of shape (b, l)
    returns: tensor of shape (b, d_k_original)
    """
    token_embeddings = model_output #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    return torch.max(token_embeddings, 1)[0]

def TailrLoss(pre, tgt, gamma, num_label):
    pre = F.softmax(pre, dim = -1)
    log_pre = (torch.log(gamma + (1 - gamma) * pre)) / (1 - gamma)

    tgt = np.eye(num_label)[tgt.cpu().numpy().astype(int)]
    tgt = torch.tensor(tgt, dtype = torch.float, device = "cpu")
    loss = -torch.sum(torch.mul(log_pre, tgt), dim = -1)
    return loss.mean()


class RobertaForSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, args,tokenizer_amr=None):
        super(RobertaForSequenceClassification, self).__init__(config)
        config.output_hidden_states = True
        self.num_labels = config.num_labels


        self.roberta_amr=RobertaAdapterModel(config)
        self.roberta_amr.resize_token_embeddings(len(tokenizer_amr))

        self.dense = nn.Linear(768 * 2, 768)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(768, 2)
    


        self.init_weights()
        self.k=4


        self.mlp_q1 = MLP(768, 768 * self.k, 768 // 2, 768 * self.k)
        self.mlp_q2 = MLP(768, 768 * self.k, 768 // 2, 768 * self.k)
        self.mlp_a1 = MLP(768, 768 * self.k, 768 // 2, 768 * self.k)
        self.mlp_a2 = MLP(768, 768 * self.k, 768 // 2, 768 * self.k)

        self.meta_q = nn.Linear(768 * 2, 768)
        self.meta_a = nn.Linear(768 * 2, 768)
     


        self.q_attn = MultiHeadAttention(heads = 6, d_model = 768)
        self.q_ffn = PositionwiseFeedForward(d_model = 768, d_ff = 768)

        self.q_slot_self_attn = SlotSelfAttention(
            SlotAttentionLayer(size = 768, self_attn = deepcopy(self.q_attn), feed_forward = deepcopy(self.q_ffn),
                               dropout = 0.1),
            N = 1)

        '''transformer架构 自注意力计算'''

        self.a_attn = MultiHeadAttention(heads = 6, d_model = 768)
        self.a_ffn = PositionwiseFeedForward(d_model = 768, d_ff = 768)

        self.a_slot_self_attn = SlotSelfAttention(
            SlotAttentionLayer(size = 768, self_attn = deepcopy(self.a_attn), feed_forward = deepcopy(self.a_ffn),
                               dropout = 0.1),
            N = 1)

        self.q_a_amr_interactiveattention=Question_Answer_InteractiveAttention()

    def metafortransform(self, q_feature1, q_feature2, a_feature1, a_feature2, batch_size, device):
        # q_feature1 q_feature2维度相同
        # meta-knowledge extraction
        q_feature = self.meta_q(torch.cat((q_feature1, q_feature2), dim = -1))
        a_feature = self.meta_a(torch.cat((a_feature1, a_feature2), dim = -1))

        '''Personalized transformation parameter matrix'''
        meta_q1 = self.mlp_q1(q_feature, device).reshape(batch_size, -1, 768, self.k)
        meta_q2 = self.mlp_q2(q_feature, device).reshape(batch_size, -1, self.k, 768)
        meta_a1 = self.mlp_a1(a_feature, device).reshape(batch_size, -1, 768, self.k)
        meta_a2 = self.mlp_a2(a_feature, device).reshape(batch_size, -1, self.k, 768)



        meta_bias_q1 = torch.mean(meta_q1, dim = 1)
        meta_bias_q2 = torch.mean(meta_q2, dim = 1)
        meta_bias_a1 = torch.mean(meta_a1, dim = 1)
        meta_bias_a2 = torch.mean(meta_a2, dim = 1)

        low_weight_q1 = []
        for i in range(meta_q1.shape[0]):
            weight = F.softmax(meta_q1[i] + meta_bias_q1[i], dim = 1)
            low_weight_q1.append(weight)
        low_weight_q1 = torch.stack(low_weight_q1, dim = 0).to(device)

        low_weight_q2 = []
        for i in range(meta_q2.shape[0]):
            weight = F.softmax(meta_q2[i] + meta_bias_q2[i], dim = 1)
            low_weight_q2.append(weight)
        low_weight_q2 = torch.stack(low_weight_q2, dim = 0).to(device)

        low_weight_a1 = []
        for i in range(meta_a1.shape[0]):
            weight = F.softmax(meta_a1[i] + meta_bias_a1[i], dim = 1)
            low_weight_a1.append(weight)
        low_weight_a1 = torch.stack(low_weight_a1, dim = 0).to(device)

        low_weight_a2 = []
        for i in range(meta_a2.shape[0]):
            weight = F.softmax(meta_a2[i] + meta_bias_a2[i], dim = 1)
            low_weight_a2.append(weight)
        low_weight_a2 = torch.stack(low_weight_a2, dim = 0).to(device)

        # The learned matrix as the weights of the transformed network
        q_embed_1 = torch.sum(torch.multiply(q_feature.unsqueeze(-1), low_weight_q1), dim = -2)
        q_embed = torch.sum(torch.multiply(q_embed_1.unsqueeze(-1), low_weight_q2), dim = -2)

        a_embed_1 = torch.sum(torch.multiply(a_feature.unsqueeze(-1), low_weight_a1), dim = -2)
        a_embed = torch.sum(torch.multiply(a_embed_1.unsqueeze(-1), low_weight_a2), dim = -2)

        return q_embed, a_embed

    def forward(self,question_graph_input_ids=None,answer_graph_input_ids=None,question_graph_attention_mask=None,answer_graph_attention_mask=None,graph_result=None,device=None):
        

        question_graph_edge_index=graph_result['question_graph_edge_index']
        answer_graph_edge_index=graph_result['answer_graph_edge_index']
        question_graph_edge_types=graph_result['question_graph_edge_types']
        answer_graph_edge_types=graph_result['answer_graph_edge_types']

        question_new_edge_index=[]
        question_new_edge_type=[]
        for graph_doc,type_doc in zip(question_graph_edge_index,question_graph_edge_types):
            question_new_edge_index.append(graph_doc)
            question_new_edge_type.append(type_doc)
        question_graph_batch=transform_graph_geometric(question_graph_input_ids,question_new_edge_index,question_new_edge_type,device=device)
        question_transformer_outputs=self.roberta_amr(question_graph_input_ids,attention_mask=question_graph_attention_mask,graph=question_graph_batch)


        answer_new_dege_index=[]
        answer_new_edge_type=[]
        for graph_doc,type_doc in zip(answer_graph_edge_index,answer_graph_edge_types):
            answer_new_dege_index.append(graph_doc)
            answer_new_edge_type.append(type_doc)
        answer_graph_batch=transform_graph_geometric(answer_graph_input_ids,answer_new_dege_index,answer_new_edge_type,device=device)
        answer_transformer_outputs=self.roberta_amr(answer_graph_input_ids,attention_mask=answer_graph_attention_mask,graph=answer_graph_batch)
        

        question_amr_sequence_output=question_transformer_outputs[0]
        answer_amr_sequence_output=answer_transformer_outputs[0]
        question_text_sequence_output=self.roberta_amr(question_graph_input_ids,attention_mask=question_graph_attention_mask)[0]
        answer_text_sequence_output=self.roberta_amr(answer_graph_input_ids,attention_mask=answer_graph_attention_mask)[0]

        question_meta_embedding,answer_meta_embedding=self.metafortransform(question_amr_sequence_output,question_text_sequence_output,answer_amr_sequence_output,answer_text_sequence_output,question_text_sequence_output.shape[0],device)


        question_amr_slot=self.q_slot_self_attn(self.dropout(question_meta_embedding))
        answer_amr_slot=self.a_slot_self_attn(self.dropout(answer_meta_embedding))

        # 问题和答案交互
        question_embedding,answer_embedding=self.q_a_amr_interactiveattention(question_amr_slot,answer_amr_slot)

        # 池化操作
        question_pooling=mean_pooling(question_embedding,question_graph_attention_mask)
        answer_pooling=mean_pooling(answer_embedding,answer_graph_attention_mask)

        # 问题向量和答案向量融合
        output=self.dense(torch.cat((question_pooling,answer_pooling),dim=1))
        output=torch.sigmoid(output)
        logits = self.out_proj(output)

        out = {}
        out['logits'] = logits
        out['pooling'] = output
        return out

