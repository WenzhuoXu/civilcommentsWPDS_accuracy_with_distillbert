import os
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import DistilBertForSequenceClassification, DistilBertModel
from transformers import BertTokenizerFast, DistilBertTokenizerFast


class DistilBertClassifier(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        return outputs


class DistilBertFeaturizer(DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.hidden_size

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        hidden_state = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        pooled_output = hidden_state[:, 0]
        return pooled_output


def initialize_model(model_name, num_labels, model_path=None):
    model = DistilBertClassifier.from_pretrained(model_name, num_labels=num_labels)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    return model


def initialize_eval_metric(eval_metric):
    import wilds.common.metrics.all_metrics as all_metrics
    name = eval_metric
    if name == 'acc':
        return all_metrics.Accuracy(all_metrics.multiclass_logits_to_pred)
    elif name == 'pearson':
        return all_metrics.PearsonCorrelation()
    else:
        raise NotImplementedError


def initialize_loss_function(config):
    name = config
    if name == 'xent':
        return nn.CrossEntropyLoss(reduction='mean')
    elif name == 'mse':
        return nn.MSELoss(reduction='mean')
    else:
        raise NotImplementedError


def initialize_bert_transform(model, max_token_length):
    def get_bert_tokenizer(model):
        if model == "bert-base-uncased":
            return BertTokenizerFast.from_pretrained(model)
        elif model == "distilbert-base-uncased":
            return DistilBertTokenizerFast.from_pretrained(model)
        else:
            raise ValueError(f"Model: {model} not recognized.")

    assert "bert" in model
    assert max_token_length is not None

    tokenizer = get_bert_tokenizer(model)

    def transform(text):
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_token_length,
            return_tensors="pt",
        )
        if model == "bert-base-uncased":
            x = torch.stack(
                (
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"],
                ),
                dim=2,
            )
        elif model == "distilbert-base-uncased":
            x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x

    return transform


def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def checkpoint_save(model, name, epoch):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint:', f)


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch


def evaluate_model(model, dataloader, logger, iteration, loss_fn, eval_metric, device, mode, checkpoint=None):
    # load checkpoint if provided
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
        avg_loss = 0
        avg_metric = 0

        for batch in dataloader:
            attention_mask, is_good = batch
            attention_mask = attention_mask.to(device)
            is_good = is_good.to(device)
            # detailed_description = detailed_description.to(device)

            logits = model(attention_mask)
            # compute loss
            loss = loss_fn(logits, is_good)
            avg_loss += loss.item()

            # compute metric
            metric = eval_metric.compute(logits, is_good.float())
            avg_metric += metric['acc_avg']

        avg_loss /= len(dataloader)
        avg_metric /= len(dataloader)

        if mode == 'val':
            logger.add_scalar('val_loss', avg_loss, iteration)
            logger.add_scalar('val_metric', avg_metric, iteration)
            print('-' * 72)
            print('Val loss: {:.4f}, Val metric: {:.4f}'.format(avg_loss, avg_metric))

        if mode == 'test':
            logger.add_scalar('test_loss', avg_loss, iteration)
            logger.add_scalar('test_metric', avg_metric, iteration)
            print('-' * 72)
            print('Test loss: {:.4f}, Test metric: {:.4f}'.format(avg_loss, avg_metric))

    model.train()
    return avg_loss, avg_metric


def to_ndarray(item: Any, dtype: np.dtype = None) -> np.ndarray:
    r"""
    Overview:
        Change `torch.Tensor`, sequence of scalars to ndarray, and keep other data types unchanged.
    Arguments:
        - item (:obj:`object`): the item to be changed
        - dtype (:obj:`type`): the type of wanted ndarray
    Returns:
        - item (:obj:`object`): the changed ndarray
    .. note:

        Now supports item type: :obj:`torch.Tensor`,  :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`
    """
    def transform(d):
        if dtype is None:
            return np.array(d)
        else:
            return np.array(d, dtype=dtype)

    if isinstance(item, dict):
        new_data = {}
        for k, v in item.items():
            new_data[k] = to_ndarray(v, dtype)
        return new_data
    elif isinstance(item, list) or isinstance(item, tuple):
        if len(item) == 0:
            return None
        elif hasattr(item, '_fields'):  # namedtuple
            return type(item)(*[to_ndarray(t, dtype) for t in item])
        else:
            new_data = []
            for t in item:
                new_data.append(to_ndarray(t, dtype))
            return new_data
    elif isinstance(item, torch.Tensor):
        if item.device != 'cpu':
            item = item.detach().cpu()
        if dtype is None:
            return item.numpy()
        else:
            return item.numpy().astype(dtype)
    elif isinstance(item, np.ndarray):
        if dtype is None:
            return item
        else:
            return item.astype(dtype)
    elif isinstance(item, bool) or isinstance(item, str):
        return item
    elif np.isscalar(item):
        return np.array(item)
    elif item is None:
        return None
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def get_data_distribution(dataset1, dataset2):
    """
        prob_1_m: prob dis of $beta$ porpotion of batch; [2, dim]
        prob_2_m: prob dis from $beta-alpha$ to $beta$ of batch [2, dim]

        In this case, the prob is multinomial
    """
    # print(dir(batch_0))
    # print(batch_0.metadata_array)
    metadata_array_1 = to_ndarray(dataset1.dataset.metadata_array)
    metadata_array_2 = to_ndarray(dataset2.metadata_array)
    
    # prob_1 --> prob dis of batch ; prob_2 --> prob dis of last $alpha$ porpotion of batch 
    prob_1, prob_2 = np.zeros(metadata_array_1.shape[1]), np.zeros(metadata_array_1.shape[1])
    batch_size_1 = metadata_array_1.shape[0]
    batch_size_2 = metadata_array_2.shape[0]
    for i in range(metadata_array_1.shape[1]):
        prob_1[i] = metadata_array_1[:, i].sum() / batch_size_1

    for j in range(metadata_array_2.shape[1]):
        prob_2[j] = metadata_array_2[:, j].sum() / batch_size_2

    # convert the prob into standard multinomial
    prob_1_m = []
    prob_2_m = []

    for i in range(len(prob_1)):
        p_1 = prob_1[i]
        p_2 = prob_2[i]
        prob_1_m.append([p_1, 1-p_1])
        prob_2_m.append([p_2, 1-p_2])

    prob_1_m = np.array(prob_1_m)
    prob_2_m = np.array(prob_2_m)

    return prob_1_m, prob_2_m


def epsilon_kl_divergence(y_recent, y_average, epsilon=0.02):
    # This function takes two probability distributions as input, and outputs its kl divergence. 
    # For a discrete distribution the divergence will be computed
    # exactly as is described in Runtian's paper.
    ind_recent = len(y_recent)
    ind_ave = len(y_average)

    if(ind_recent != ind_ave):
        print('The source and target data must have the same labels.') 

    div_label = np.zeros(ind_recent) # initialize divergence by labels
    for i in range(ind_recent):
        div_label[i] = np.sum(-1 * y_recent[i] * y_average[i] / \
        np.maximum(y_recent[i], epsilon)) + np.sum(y_average[i] * np.log(np.maximum(y_average[i], epsilon) / \
        np.maximum(y_recent[i], epsilon))) + 1

    # for i in range(ind_recent):
    #     div_label[i] = np.sum(y_recent[i] * y_average[i] / \
    #     np.maximum(y_recent[i], epsilon)) + np.sum(y_average[i] * np.log(y_average[i] / \
    #     np.maximum(y_average[i], epsilon))+ 1)  

    # The total divergence can be seen as a joint distribution of 
    # separate divergences. Assume the label probabilities are the same.

    return np.sum(div_label)