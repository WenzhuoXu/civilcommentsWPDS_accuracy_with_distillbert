import os
import warnings
from datetime import datetime

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


def evaluate_model(model, dataloader, logger, iteration, loss_fn, eval_metric, device, checkpoint=None, mode='val'):
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