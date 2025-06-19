
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import glob
import logging
import os
import random
from typing import Tuple
from loss.LossManager import LossManager
from loss.tailr_loss import TailrLoss
from optimal.sample_weighting import weight_learner
from loss.SCAN_contrastive_loss import Contrastive_loss
from datetime import datetime
import torch.optim as optim
import Model_graph
import Model_Roberta_amr
from imap_qa import calc_map1, calc_mrr1, read_data, accuracy, calc_accuracy, all_calc_accuracy, calc_precision1
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import higher
from optimal.FCdebias2 import FC
from transformers import RobertaModel
from transformers import get_linear_schedule_with_warmup
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)
from utils_glue_graph import (compute_metrics, convert_examples_to_features,
                              output_modes, processors)
from optimal.adversarial import FGM, PGD
import itertools
from torch.autograd import Variable



logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)),
    ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}

EDGES_AMR = ["have-rel-role", "have-degree", "all-over", "distance-quantity", "date-entity", ":ARG0", ":ARG0-of",
             ":ARG1", ":ARG1-of", ":ARG2", ":ARG2-of", ":ARG3", ":ARG3-of", ":ARG4",
             ":ARG4-of", ":ARG5", ":ARG5-of", ":ARG6", ":ARG6-of", ":ARG7", ":accompanier", ":accompanier-of",
             ":age", ":age-of", ":beneficiary", ":beneficiary-of", ":century", ":concession", ":concession-of",
             ":condition", ":condition-of", ":conj-as-if", ":consist", ":consist-of", ":day", ":dayperiod",
             ":dayperiod-of", ":decade", ":degree", ":degree-of", ":destination", ":destination-of", ":direction",
             ":direction-of", ":domain", ":domain-of", ":duration", ":duration-of", ":era", ":example", ":example-of",
             ":extent", ":extent-of", ":frequency", ":frequency-of", ":instrument", ":instrument-of", ":li",
             ":location",
             ":location-of", ":manner", ":manner-of", ":medium", ":medium-of", ":mod", ":mod-of", ":mode", ":month",
             ":name", ":op1", ":op1-of", ":op10", ":op11", ":op12", ":op12_<lit>", ":op13", ":op14", ":op14_<lit>_:",
             ":op15", ":op16", ":op17", ":op18", ":op19", ":op19_<lit>_:", ":op1_<lit>", ":op2", ":op2-of", ":op20",
             ":op21", ":op22", ":op23", ":op24", ":op25", ":op25_<lit>_:", ":op26", ":op27", ":op27_<lit>_.", ":op28",
             ":op29", ":op3", ":op3-of", ":op30", ":op31", ":op32", ":op33", ":op34", ":op35", ":op36", ":op37",
             ":op38", ":op39", ":op4", ":op40", ":op41", ":op5", ":op6", ":op7", ":op8", ":op9", ":ord", ":ord-of",
             ":part", ":part-of", ":path", ":path-of", ":polarity", ":polarity-of", ":polite", ":poss", ":poss-of",
             ":prep-a", ":prep-about", ":prep-after", ":prep-against", ":prep-against-of", ":prep-along-to",
             ":prep-along-with", ":prep-amid", ":prep-among", ":prep-around", ":prep-as", ":prep-at", ":prep-back",
             ":prep-between", ":prep-by", ":prep-down", ":prep-for", ":prep-from", ":prep-in", ":prep-in-addition-to",
             ":prep-into", ":prep-of", ":prep-off", ":prep-on", ":prep-on-behalf", ":prep-on-behalf-of", ":prep-on-of",
             ":prep-on-side-of", ":prep-out-of", ":prep-over", ":prep-past", ":prep-per", ":prep-through", ":prep-to",
             ":prep-toward", ":prep-under", ":prep-up", ":prep-upon", ":prep-with", ":prep-without", ":purpose",
             ":purpose-of",
             ":quant", ":quant-of", ":quant101", ":quant102", ":quant104", ":quant113", ":quant114", ":quant115",
             ":quant118",
             ":quant119", ":quant128", ":quant141", ":quant143", ":quant146", ":quant148", ":quant164", ":quant165",
             ":quant166",
             ":quant179", ":quant184", ":quant189", ":quant194", ":quant197", ":quant208", ":quant214", ":quant217",
             ":quant228",
             ":quant246", ":quant248", ":quant274", ":quant281", ":quant305", ":quant306", ":quant308", ":quant309",
             ":quant312",
             ":quant317", ":quant324", ":quant329", ":quant346", ":quant359", ":quant384", ":quant396", ":quant398",
             ":quant408",
             ":quant411", ":quant423", ":quant426", ":quant427", ":quant429", ":quant469", ":quant506", ":quant562",
             ":quant597",
             ":quant64", ":quant66", ":quant673", ":quant675", ":quant677", ":quant74", ":quant754", ":quant773",
             ":quant785", ":quant787",
             ":quant79", ":quant797", ":quant801", ":quant804", ":quant86", ":quant870", ":quarter", ":range", ":scale",
             ":season",
             ":snt1", ":snt12", ":snt2", ":snt3", ":snt4", ":snt5", ":snt6", ":snt7", ":snt8", ":source", ":source-of",
             ":subevent",
             ":subevent-of", ":time", ":time-of", ":timezone", ":timezone-of", ":topic", ":topic-of", ":unit", ":value",
             ":weekday",
             ":weekday-of", ":year", ":year2"]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def prepare_optimizer(model, learning_rate, num_train_steps, warmup_ratio):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(num_train_steps * warmup_ratio), num_train_steps)
    return optimizer, scheduler


def collect_fn(batch):
    question_texts = []
    answer_texts = []
    graph_results = {}
    graph_results['question_graph_edge_index'] = []
    graph_results['answer_graph_edge_index'] = []
    graph_results['question_graph_edge_types'] = []
    graph_results['answer_graph_edge_types'] = []
    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_labels = []
    all_question_len_ids = []
    all_answer_len_ids = []
    all_input_ids_ner = []
    all_input_mask_ner = []
    all_question_len_ids_ner = []
    all_answer_len_ids_ner = []
    all_question_ner_len = []
    all_answer_ner_len = []
    all_question_graph_input_ids = []
    all_answer_graph_input_ids = []
    all_question_graph_attention_mask = []
    all_answer_graph_attention_mask = []
    all_question_parse_graph = []
    all_answer_parse_graph = []

    for sample in batch:
        tensor_sample, question_text, answer_text, graph_result, question_parse_graph, answer_parse_graph = sample
        input_ids, attention_mask, token_type_ids, labels, question_len_ids, answer_len_ids, question_graph_input_ids, answer_graph_input_ids, question_graph_attention_mask, answer_graph_attention_mask = tensor_sample
        all_input_ids.append(input_ids.tolist())
        all_attention_mask.append(attention_mask.tolist())
        all_token_type_ids.append(token_type_ids.tolist())
        all_labels.append(labels.tolist())
        all_question_parse_graph.append(question_parse_graph)
        all_answer_parse_graph.append(answer_parse_graph)
        all_question_len_ids.append(question_len_ids.tolist())
        all_answer_len_ids.append(answer_len_ids.tolist())
        all_question_graph_input_ids.append(question_graph_input_ids.tolist())
        all_answer_graph_input_ids.append(answer_graph_input_ids.tolist())
        all_question_graph_attention_mask.append(question_graph_attention_mask.tolist())
        all_answer_graph_attention_mask.append(answer_graph_attention_mask.tolist())

        question_texts.append(question_text)
        answer_texts.append(answer_text)
        graph_results['question_graph_edge_index'].append(graph_result['question_graph_edge_index'])
        graph_results['answer_graph_edge_index'].append(graph_result['answer_graph_edge_index'])
        graph_results['question_graph_edge_types'].append(graph_result['question_graph_edge_types'])
        graph_results['answer_graph_edge_types'].append(graph_result['answer_graph_edge_types'])

    all_input_ids = torch.tensor(all_input_ids)
    all_attention_mask = torch.tensor(all_attention_mask)
    all_token_type_ids = torch.tensor(all_token_type_ids)
    all_labels = torch.tensor(all_labels)
    all_question_len_ids = torch.tensor(all_question_len_ids)
    all_answer_len_ids = torch.tensor(all_answer_len_ids)
    all_question_graph_input_ids = torch.tensor(all_question_graph_input_ids)
    all_answer_graph_input_ids = torch.tensor(all_answer_graph_input_ids)
    all_question_graph_attention_mask = torch.tensor(all_question_graph_attention_mask)
    all_answer_graph_attention_mask = torch.tensor(all_answer_graph_attention_mask)

    return (
        (all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_question_len_ids, all_answer_len_ids, all_question_graph_input_ids,
         all_answer_graph_input_ids, all_question_graph_attention_mask, all_answer_graph_attention_mask),
        question_texts,
        answer_texts,
        graph_results,
        all_question_parse_graph,
        all_answer_parse_graph
    )


def train(args, train_dataset, model, tokenizer, meta_dev_dataset=None, tokenizer_amr=None):


    large_loss = 1000000000000
    name = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    name = "logs/" + str(name)

    test_name = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    if not os.path.isdir('test'):
        os.mkdir('test')
    test_name = 'test/' + str(test_name)

    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  num_workers=0, collate_fn=collect_fn)

    meta_dev_sampler = RandomSampler(meta_dev_dataset) if args.local_rank == -1 else DistributedSampler(
        meta_dev_dataset)
    meta_dev_dataloader = DataLoader(meta_dev_dataset, sampler=meta_dev_sampler, batch_size=args.meta_batch,
                                     num_workers=0, collate_fn=collect_fn)
    meta_dev_dataloader = itertools.cycle(meta_dev_dataloader)

    FCweight = FC([768, 768], dropout=0, k=4, device=args.device, beta=args.beta, topk=args.topk)
    model_amr = Model_Roberta_amr.RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,
                                                                                     args=args,
                                                                                     tokenizer_amr=tokenizer_amr)
    FCweight.to(args.device)
    model_amr.to(args.device)

    num_train_steps = int(len(train_dataset) / args.per_gpu_train_batch_size * args.num_train_epochs)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    meta_model = Model_graph.RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,
                                                                                           args=args)
    meta_model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)

    optimizer, scheduler = prepare_optimizer(model, args.learning_rate, num_train_steps, args.base_warmup)
    meta_optimizer, meta_scheduler = prepare_optimizer(meta_model, args.learning_rate, num_train_steps,
                                                       args.base_warmup)

    fc_param_optimizer = list(FCweight.parameters())
    fc_optimizer = optim.AdamW(fc_param_optimizer, lr=args.fc_lr)
    fc_scheduler = get_linear_schedule_with_warmup(fc_optimizer, int(num_train_steps * args.fc_warmup), num_train_steps)

    amr_param_optimizer = list(model_amr.parameters())
    amr_optimizer = optim.AdamW(amr_param_optimizer, lr=args.learning_rate)
    amr_scheduler = get_linear_schedule_with_warmup(amr_optimizer, int(num_train_steps * args.fc_warmup),
                                                    num_train_steps)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # 对抗训练
    if args.adv_option == "FGM":
        adversarial = FGM(model, emb_name="roberta.embeddings.word_embeddings.weight", epsilon=1.0)
        # adversarial_amr = FGM(model_amr, emb_name = "roberta.embeddings.word_embeddings.weight", epsilon = 1.0)
    elif args.adv_option == "PGD":
        adversarial = PGD(model, emb_name="roberta.embeddings.word_embeddings.weight", epsilon=1.0)
        # adversarial_amr = PGD(model_amr, emb_name = "roberta.embeddings.word_embeddings.weight", epsilon = 1.0)
    else:
        adversarial = ""
        # adversarial_amr = ""

    loss_manager = LossManager(device=args.device, loss_type=args.loss_type, cl_option=args.cl_option,
                               loss_cl_type=args.cl_method, gamma=args.tailr_gamma)
    loss_fct = TailrLoss(args.device, gamma=args.tailr_gamma, num_label=2)
    loss_fct_weight = TailrLoss(args.device, gamma=args.tailr_gamma, num_label=2, reduce=False)
    contrastive_loss = Contrastive_loss(args.tao)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    checkpoint = 0,
    MAP, MRR, P1 = 0.0, 0.0, 0.0
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    total_bce_loss, total_debias_bce_loss, total_con_loss = 0.0, 0.0, 0.0
    total_contrastive_loss = 0.0
    torch.cuda.empty_cache()
    model.zero_grad()
    model_amr.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    out = open(name + ".txt", "a+")
    out.write(str(args) + "\n")
    out.close()

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            model.train()
            _batch = batch[0]
            _batch = tuple(t.to(args.device) for t in _batch)
            # batch = _batch[0]
            inputs = {'input_ids': _batch[0],
                      'attention_mask': _batch[1],
                      'token_type_ids': _batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      # XLM and RoBERTa don't use segment_ids
                      'labels': _batch[3],
                      'question_len_ids': _batch[4],
                      'answer_len_ids': _batch[5],
                      'question_original_text': batch[1],
                      'answer_original_text': batch[2],
                      'device': args.device,

                      'max_seq_length': args.max_seq_length,
                      'question_ner_len': _batch[10],
                      'answer_ner_len': _batch[11],
                      'question_parse_graph': batch[4],
                      'answer_parse_graph': batch[5]
                      }
            inputs_cl = {
                'input_ids': _batch[0],
                'attention_mask': _batch[1],
                'token_type_ids': _batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                'question_graph_input_ids': _batch[12],
                'answer_graph_input_ids': _batch[13],
                'question_graph_attention_mask': _batch[14],
                'answer_graph_attention_mask': _batch[15],
                'graph_result': batch[3],
                'device': args.device
            }

            meta_model.load_state_dict(model.state_dict())
            meta_optimizer.load_state_dict(optimizer.state_dict())
            meta_optimizer.zero_grad()

            with higher.innerloop_ctx(meta_model, meta_optimizer) as (meta_m, meta_opt):
                meta_m.train()

                outputs_meta = meta_m(**inputs)
                logits_meta, cfeatures_meta = outputs_meta['logits'], outputs_meta['pooling']
                outputs_fc_meta = FCweight(cfeatures_meta.detach())
                debias_logits_meta, debias_fea_meta, bias_meta = outputs_fc_meta['debias_logits'], outputs_fc_meta[
                    'debias_fea'], outputs_fc_meta['bias']



                bce_loss_debias_meta = loss_fct(debias_logits_meta.view(-1, 2), _batch[3].view(-1))
                con_loss_meta = contrastive_loss(fea=cfeatures_meta, pos_fea=debias_fea_meta, neg_fea=bias_meta)
                cons_loss_meta = con_loss_meta  + bce_loss_debias_meta

                # 对比学习
                if args.cl_option:
                    outputs_etx_meta_1 = model_amr(**inputs_cl)
                    hidden_emb_etx_meta_1 = outputs_etx_meta_1['pooling']
                    contrast_loss_meta_1 = loss_manager.compute(input_x=logits_meta, target=_batch[3],
                                                                hidden_emb_x=cfeatures_meta,
                                                                hidden_emb_y=hidden_emb_etx_meta_1,
                                                                alpha=args.cl_loss_weight_amr)
                    del hidden_emb_etx_meta_1, outputs_etx_meta_1
                    outputs_etx_meta = meta_m(**inputs)
                    _, hidden_emb_etx_meta = outputs_etx_meta['logits'], outputs_etx_meta['pooling']
                    contrast_loss_meta = loss_manager.compute(input_x=logits_meta, target=_batch[3],
                                                              hidden_emb_x=cfeatures_meta,
                                                              hidden_emb_y=hidden_emb_etx_meta,
                                                              alpha=args.cl_loss_weight)
                else:
                    contrast_loss_meta = loss_manager.compute(logits_meta, _batch[3])

                loss_meta = cons_loss_meta + contrast_loss_meta + contrast_loss_meta_1

                meta_opt.step(loss_meta)

                batch_v = next(meta_dev_dataloader)
                _batch_v = batch_v[0]
                _batch_v = tuple(t.to(args.device) for t in _batch_v)
                inputs_v = {'input_ids': _batch_v[0],
                            'attention_mask': _batch_v[1],
                            'token_type_ids': _batch_v[2] if args.model_type in ['bert', 'xlnet'] else None,
                            # XLM and RoBERTa don't use segment_ids
                            'labels': _batch_v[3],
                            'question_len_ids': _batch_v[4],
                            'answer_len_ids': _batch_v[5],
                            'question_original_text': batch_v[1],
                            'answer_original_text': batch_v[2],
                            'device': args.device,
                            'max_seq_length': args.max_seq_length,
                            'question_ner_len': _batch_v[10],
                            'answer_ner_len': _batch_v[11],
                            'question_parse_graph': batch_v[4],
                            'answer_parse_graph': batch_v[5],
                            }
                input_v_amr = {
                    'input_ids': _batch_v[0],
                    'attention_mask': _batch_v[1],
                    'token_type_ids': _batch_v[2] if args.model_type in ['bert', 'xlnet'] else None,
                    'question_graph_input_ids': _batch_v[12],
                    'answer_graph_input_ids': _batch_v[13],
                    'question_graph_attention_mask': _batch_v[14],
                    'answer_graph_attention_mask': _batch_v[15],
                    'graph_result': batch_v[3],
                    'device': args.device
                }
                meta_m.eval()
                outputs_meta_v = meta_m(**inputs_v)
                cfeatures_meta_v = outputs_meta_v['pooling']
                outputs_fc_meta_v = FCweight(cfeatures_meta_v.detach())
                debias_logits_meta_v = outputs_fc_meta_v['debias_logits']
                loss_meta_v = loss_fct(debias_logits_meta_v.view(-1, 2), _batch_v[3].view(-1))

                output_meta_amr_v = model_amr(**input_v_amr)
                contrast_amr_loss = loss_fct(output_meta_amr_v['logits'], _batch_v[3])

                fc_optimizer.zero_grad()
                loss_meta_v.backward()
                fc_optimizer.step()
                fc_scheduler.step()

                amr_optimizer.zero_grad()
                contrast_amr_loss.backward()
                amr_optimizer.step()
                amr_scheduler.step()

            outputs = model(**inputs)
            logits, cfeatures = outputs['logits'], outputs['pooling']
            outputs_fc = FCweight(cfeatures.detach())
            debias_logits, debias_fea, bias = outputs_fc['debias_logits'], outputs_fc['debias_fea'], outputs_fc['bias']


            bce_loss_debias = loss_fct(debias_logits.view(-1, 2), _batch[3].view(-1))
            con_loss = contrastive_loss(fea=cfeatures, pos_fea=debias_fea, neg_fea=bias)
            cons_loss = con_loss  + bce_loss_debias
            if args.cl_option:
                outputs_etx = model(**inputs)
                _, hidden_emb_etx = outputs_etx['logits'], outputs_etx['pooling']
                contrast_loss = loss_manager.compute(input_x=logits, target=_batch[3],
                                                     hidden_emb_x=cfeatures,
                                                     hidden_emb_y=hidden_emb_etx,
                                                     alpha=args.cl_loss_weight)
                outputs_etx_1 = model_amr(**inputs_cl)
                hidden_emb_etx_1 = outputs_etx_1['pooling']
                contrast_loss_amr = loss_manager.compute(input_x=logits, target=_batch[3], hidden_emb_x=cfeatures,
                                                         hidden_emb_y=hidden_emb_etx_1, alpha=args.cl_loss_weight_amr)
            else:
                contrast_loss = loss_manager.compute(logits, _batch[3])

            loss = cons_loss + contrast_loss + contrast_loss_amr

            optimizer.zero_grad()
            loss.backward()



            attack_train(args, model, inputs, _batch[3], loss_fct, adversarial)

            tr_loss += loss.item()
            total_contrastive_loss += contrast_loss.item()


            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                epoch_iterator.set_postfix(loss=tr_loss / global_step,
                                           con_loss=total_contrastive_loss / global_step)

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    result, preds, acc_preds = evaluate(args, model, tokenizer)
                    filename = args.data_dir + "/" + args.task_name + "_valid.tsv"
                    t_f = read_data(filename)
                    map_val = calc_map1(t_f, preds, args.output_dir + "_" + str(global_step), test_name)
                    mrr_val = calc_mrr1(t_f, preds)
                    p1_val = calc_precision1(t_f, preds)
                    if map_val > MAP:
                        MAP = map_val
                        MRR = mrr_val
                        checkpoint = global_step
                        P1 = p1_val
                    out = open(name + ".txt", "a+")
                    out.write(
                        args.output_dir + " checkpoint-{}: MAP: {} ,MRR: {} ,P@1: {} \n".format(global_step, map_val,
                                                                                                mrr_val, p1_val))
                    out.close()

                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

                # 保存最佳参数
                if args.local_rank in [-1, 0] and args.save_steps > 0 and tr_loss < large_loss:
                    large_loss = tr_loss
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best_loss-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args_best_loss.bin'))
                    logger.info("Saving model checkpoint-best_loss to %s", output_dir)

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    out = open(name + ".txt", "a+")
    out.write(
        args.output_dir + " best_checkpoint-{}: MAP: {} ,MRR: {} ,P@1: {} \n".format(checkpoint, MAP,
                                                                                     MRR, P1))
    out.close()
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", checkpoint=None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    loss_manager = LossManager(device=args.device, loss_type=args.loss_type, cl_option=args.cl_option,
                               loss_cl_type=args.cl_method, gamma=args.tailr_gamma)
    loss_fct = TailrLoss(args.device, gamma=args.tailr_gamma, num_label=2)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=0, collate_fn=collect_fn)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            _batch = batch[0]
            _batch = tuple(t.to(args.device) for t in _batch)

            with torch.no_grad():
                inputs = {'input_ids': _batch[0],
                          'attention_mask': _batch[1],
                          'token_type_ids': _batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          # XLM and RoBERTa don't use segment_ids
                          'labels': _batch[3],
                          'question_len_ids': _batch[4],
                          'answer_len_ids': _batch[5],
                          'question_original_text': batch[1],
                          'answer_original_text': batch[2],
                          'device': args.device,
                          'max_seq_length': args.max_seq_length,
                          'question_ner_len': _batch[10],
                          'answer_ner_len': _batch[11],
                          'question_parse_graph': batch[4],
                          'answer_parse_graph': batch[5]
                          }

                output = model(**inputs)['logits']

                tmp_eval_loss = loss_fct(output, _batch[3])

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = output.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, output.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                eval_loss = eval_loss / nb_eval_steps

        if args.output_mode == "classification":
            # print(preds)
            f = open("out1.txt", "w+")
            for i in preds:
                f.write(str(i) + " \n")
            preds2 = preds[:, 1:]
            preds = np.argmax(preds, axis=1)

            ff = open("out2.txt", "w+")
            fff = open("out3.txt", "w+")
            ffff = open("out4.txt", "w+")
            for i in preds:
                ff.write(str(i) + " \n")
            for i in preds2:
                fff.write(str(i) + " \n")
            for i in out_label_ids:
                ffff.write(str(i) + "\n")

        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        print(args.task_name)

    return results, preds2, preds


def attack_train(args, model, inputs, labels, loss_fct, adversarial):
    """
            对抗训练
            """
    # FGM
    if args.adv_option == 'FGM':
        adversarial.attack()
        output = model(**inputs)['logits']
        loss_adv = loss_fct(output, labels)
        loss_adv.backward()
        adversarial.restore()
    # PGD
    if args.adv_option == 'PGD':
        adversarial.backup_grad()
        K = 3
        for t in range(K):
            adversarial.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
            if t != K - 1:
                model.zero_grad()
            else:
                adversarial.restore_grad()
            output = model(**inputs)[0]
            loss_adv = loss_fct.compute(output, labels)
            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        adversarial.restore()


class MyDataset(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    # tensors: Tuple[Tensor, ...]

    def __init__(self, all_question_original_text, all_answer_original_text,
                 graph_result, all_question_parse_graph, all_answer_parse_graph,
                 #  all_question_graph_edge_index,all_answer_graph_edge_index,all_question_graph_edge_types,all_answer_graph_edge_types,
                 *tensors) -> None:
        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.all_question_original_text = all_question_original_text
        self.all_answer_original_text = all_answer_original_text
        self.graph_result = graph_result
        self.all_question_parse_graph = all_question_parse_graph
        self.all_answer_parse_graph = all_answer_parse_graph



    def __getitem__(self, index):
        return (tuple(tensor[index] for tensor in self.tensors),
                self.all_question_original_text[0][index],
                self.all_answer_original_text[0][index],
                self.graph_result[index],
                self.all_question_parse_graph[index],
                self.all_answer_parse_graph[index],

                )

    def __len__(self):
        return self.tensors[0].size(0)


def load_meta_dev_cache_example(args, task, tokenizer):
    processor = processors[task]()
    output_mode = output_modes[task]

    cached_features_file = os.path.join(args.data_dir, 'cached_meta_{}_{}_{}_{}'.format(
        'dev',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_meta_dev_examples(args.data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ['roberta']),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                args=args
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_question_len_ids = torch.tensor([f.question_len_id for f in features], dtype=torch.long)
    all_answer_len_ids = torch.tensor([f.answer_len_id for f in features], dtype=torch.long)

    all_question_parse_graph = [f.question_parse_graph for f in features]
    all_answer_parse_graph = [f.answer_parse_graph for f in features]

    all_question_original_text = []
    all_question_original_text.append([f.question_original_text for f in features])

    all_answer_original_text = []
    all_answer_original_text.append([f.answer_original_text for f in features])


    # AMR图内容
    all_question_graph_input_ids = torch.tensor([f.question_graph_input_ids for f in features], dtype=torch.long)
    all_answer_graph_input_ids = torch.tensor([f.answer_graph_input_ids for f in features], dtype=torch.long)

    all_question_graph_attention_mask = torch.tensor([f.question_graph_attention_mask for f in features],
                                                     dtype=torch.long)
    all_answer_graph_attention_mask = torch.tensor([f.answer_graph_attention_mask for f in features], dtype=torch.long)

    graph_result = []
    for f in features:
        graph = {}
        graph['question_graph_edge_index'] = f.question_graph_edge_index.tolist()
        graph['answer_graph_edge_index'] = f.answer_graph_edge_index.tolist()
        graph['question_graph_edge_types'] = f.question_graph_edge_types.tolist()
        graph['answer_graph_edge_types'] = f.answer_graph_edge_types.tolist()
        graph_result.append(graph)

    dataset = MyDataset(all_question_original_text, all_answer_original_text,
                        graph_result,
                        all_question_parse_graph,
                        all_answer_parse_graph,
                        all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                        all_question_len_ids, all_answer_len_ids,
                        all_question_graph_input_ids,
                        all_answer_graph_input_ids,
                        all_question_graph_attention_mask,
                        all_answer_graph_attention_mask
                        )
    return dataset


def load_valid_cache_example(args, task, tokenizer):
    processor = processors[task]()
    output_mode = output_modes[task]

    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'valid',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(
            args.data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ['roberta']),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                args=args
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_question_len_ids = torch.tensor([f.question_len_id for f in features], dtype=torch.long)
    all_answer_len_ids = torch.tensor([f.answer_len_id for f in features], dtype=torch.long)

    '''先转为numpy.ndarraays,再转为tensor'''
    all_question_original_graph_128 = torch.tensor(np.array([f.question_original_graph_128 for f in features]))
    all_answer_original_graph_128 = torch.tensor(np.array([f.answer_original_graph_128 for f in features]))

    all_question_original_text = []
    all_question_original_text.append([f.question_original_text for f in features])

    all_answer_original_text = []
    all_answer_original_text.append([f.answer_original_text for f in features])


    dataset = MyDataset(all_question_original_text, all_answer_original_text,
                        all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                        all_question_len_ids, all_answer_len_ids,
                        all_question_original_graph_128, all_answer_original_graph_128
                        )
    return dataset


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(
            args.data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ['roberta']),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                args=args
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_question_len_ids = torch.tensor([f.question_len_id for f in features], dtype=torch.long)
    all_answer_len_ids = torch.tensor([f.answer_len_id for f in features], dtype=torch.long)

    all_question_parse_graph = [f.question_parse_graph for f in features]
    all_answer_parse_graph = [f.answer_parse_graph for f in features]

    all_question_original_text = []
    all_question_original_text.append([f.question_original_text for f in features])

    all_answer_original_text = []
    all_answer_original_text.append([f.answer_original_text for f in features])


    # AMR图内容
    all_question_graph_input_ids = torch.tensor([f.question_graph_input_ids for f in features], dtype=torch.long)
    all_answer_graph_input_ids = torch.tensor([f.answer_graph_input_ids for f in features], dtype=torch.long)

    all_question_graph_attention_mask = torch.tensor([f.question_graph_attention_mask for f in features],
                                                     dtype=torch.long)
    all_answer_graph_attention_mask = torch.tensor([f.answer_graph_attention_mask for f in features], dtype=torch.long)

    graph_result = []
    for f in features:
        graph = {}
        graph['question_graph_edge_index'] = f.question_graph_edge_index.tolist()
        graph['answer_graph_edge_index'] = f.answer_graph_edge_index.tolist()
        graph['question_graph_edge_types'] = f.question_graph_edge_types.tolist()
        graph['answer_graph_edge_types'] = f.answer_graph_edge_types.tolist()
        graph_result.append(graph)

    dataset = MyDataset(all_question_original_text, all_answer_original_text,
                        graph_result,
                        all_question_parse_graph,
                        all_answer_parse_graph,
                        all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                        all_question_len_ids, all_answer_len_ids,
                        all_question_graph_input_ids,
                        all_answer_graph_input_ids,
                        all_question_graph_attention_mask,
                        all_answer_graph_attention_mask
                        )
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # 设置一个最大损失
    best_loss = 100000000000

    ## Required parameters
    parser.add_argument("--data_dir", default="data/WIKI", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default="file/roberta", type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--task_name", default="wiki", type=str,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default="PATH/TO/sel/amr", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true', default=True,
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    # 测试集
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument('--loss_type', type=str, default='tailr', help="损失函数类型")
    # 是否使用对比学习
    parser.add_argument('--cl_option', type=bool, default=False, help="是否使用对比学习")
    parser.add_argument('--cl_method', type=str, default='Rdrop', help="对比学习的方法 Rdrop/InfoNCE")
    parser.add_argument('--cl_loss_weight', type=float, default=0.5, help="对比学习loss的比例")

    parser.add_argument('--tailr_gamma', type=float, default=1e-6, help="tailr损失函数的gamma参数")
    parser.add_argument("--adv_option", type=str, default="", help="对抗训练方法")

    parser.add_argument('--lrbl', type=float, default=1e-3, help='learning rate of balance')
    parser.add_argument('--num_f', type=int, default=5, help='number of fourier spaces')

    parser.add_argument('--base_warmup', default=0.1, type=float)
    parser.add_argument('--fc_warmup', default=0.1, type=float)
    parser.add_argument('--fc_lr', default=2e-5, type=float)
    parser.add_argument('--meta_batch', default=2, type=int)
    parser.add_argument('--cuda', default='cuda:0', type=str)
    parser.add_argument('--amr_checkpoint', type=str, default="file/spring/AMR3.parsing.pt",
                        help="AMR trained checkpoint;Required. Checkpoint to restore.")
    parser.add_argument('--amr_model', type=str, default="file/spring/pretrain_models/bart-large",
                        help="bart Model;AMR Model config to use to load the model class.")
    parser.add_argument('--beam-size', type=int, default=1, help="Beam Size")
    parser.add_argument('--penman-linearization', action='store_true', default=True,
                        help="Predict using PENMAN linearization instead of ours.")
    parser.add_argument('--use-pointer-tokens', action='store_true', default=True)
    parser.add_argument('--restore-name-ops', action='store_true')
    parser.add_argument('--describe', type=str, help="describe for model")
    parser.add_argument('--beta', default=0.4, type=float, help="beta for the loss")
    parser.add_argument('--topk', default=64, type=int, help="topk for the loss")
    parser.add_argument('--cl_loss_weight_amr', default=0.3, type=float, help="")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        '''设置单卡'''
        device = torch.device(args.cuda if torch.cuda.is_available() and not args.no_cuda else "cpu")
        '''gpu总数设置为1'''
        args.n_gpu = 1
        # args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device(args.cuda, args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        print(args.task_name)
        raise ValueError("Task not found: %s" % (args.task_name))

    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # 准备模型 分词器 配置文件
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # config = BertConfig.from_pretrained("bert-base-uncased", num_labels=num_labels, finetuning_task=args.task_name)
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name,
                                          output_hidden_states=True)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    tokenizer_amr = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case)

    new_tokens_vocab = {"additional_special_tokens": []}
    # sort by edge labels
    tokens_amr = sorted(EDGES_AMR, reverse=True)
    # add edges labels to model embeddings matrix
    for t in tokens_amr:
        new_tokens_vocab["additional_special_tokens"].append(t)
    num_added_toks = tokenizer_amr.add_special_tokens(new_tokens_vocab)
    print(num_added_toks, "tokens added.")

    model = Model_graph.RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,
                                                                                      config=config,
                                                                                      args=args)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        meta_dev_dataset = load_meta_dev_cache_example(args, args.task_name, tokenizer)
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, meta_dev_dataset,
                                     tokenizer_amr=tokenizer_amr)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    results = {}
    preds = []
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case)

        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in
                sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = Model_graph.RobertaForSequenceClassification.from_pretrained(checkpoint,
                                                                                              args=args)
            model.to(args.device)

            # 模型不参与训练
            model.eval()

            result, preds, acc_preds = evaluate(args, model, tokenizer, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
    # evaluation_type="valid" #else test
    # 要测试把valid改为test
    evaluation_type = "valid"  # else test
    if (evaluation_type == "valid"):
        filename = args.data_dir + "/" + args.task_name + "_valid.tsv"
    else:
        filename = args.data_dir + "/" + args.task_name + "_test.tsv"
    t_f = read_data(filename=filename)

    map_val = calc_map1(t_f, preds, '特定测试：', 'logs/test')
    mrr_val = calc_mrr1(t_f, preds)
    p1_val = calc_precision1(t_f, preds)
    acc_val = calc_accuracy(filename, acc_preds)
    print("MAP: " + str(map_val))
    print("MRR: " + str(mrr_val))
    print("P@1: " + str(p1_val))
    print("ACC:" + str(acc_val))
    acc, report, confusion = all_calc_accuracy(filename, acc_preds)
    # print("ACC:" + str(acc))
    print("Precision, Recall and F1-Score...")
    print(report)
    print("Confusion Matrix...")
    print(confusion)
    return results


if __name__ == '__main__':
    main()
