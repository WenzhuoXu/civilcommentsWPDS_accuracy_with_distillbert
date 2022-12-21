import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.civilcomments_wpds import CivilCommentsWPDS
from civilcomments_utils import *
from w_distance import *
import numpy as np


def train(frac=0.5, batch_num = 1, test_frac=0.8, test_batch_num = 15, num_epochs=5000, batch_size=32, lr=1e-4, model_name='distilbert-base-uncased', loss_func='xent', eval_func='acc'):
    # set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # set up the model
    # model_name = 'distilbert-base-uncased'
    num_labels = 2
    model = initialize_model(model_name, num_labels).to(device)
    model.train()

    # set up the dataset
    seed = 2022
    # frac = 0.5 # fraction of the dataset to use
    transform = initialize_bert_transform(model_name, 512)
    # train_dataset = CivilCommentsWPDS(magic=seed).get_subset('train', frac, transform=transform)
    dataset = CivilCommentsWPDS(magic=seed).get_batch(t=batch_num, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
    # train_dataset = CivilCommentsWPDS(magic=seed).get_batch(t=batch_num, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # set up testing dataset
    # val_dataset = CivilCommentsWPDS(magic=seed).get_subset('val', frac, transform=transform)
    # val_dataset = CivilCommentsWPDS(magic=seed).get_batch(t=batch_num, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # set up the loss function
    loss_fn = initialize_loss_function(loss_func)

    # set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # set up the evaluation metric
    eval_metric = initialize_eval_metric(eval_func)

    # set up the summary writer
    logdir = os.path.join('./logs/train', str(batch_num))
    savedir = os.path.join('./checkpoints/', str(batch_num))
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    logger = SummaryWriter(logdir)

    # set up the training loop
    # num_epochs = 10
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_accuracy = 0
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            attention_mask, is_good = batch
            attention_mask = attention_mask.to(device)
            # detailed_description = detailed_description.to(device)
            is_good = is_good.to(device)

            # forward pass
            y_pred = model(attention_mask)
            loss = loss_fn(y_pred, is_good)

            # backward pass
            loss.backward()
            optimizer.step()

            # compute the accuracy
            accuracy = eval_metric.compute(y_pred, is_good.float())

            # log the loss and accuracy
            total_loss += loss.item()
            total_accuracy += accuracy['acc_avg']
            # print(f'Epoch {epoch}, batch {i}: loss={loss.item()}, accuracy={accuracy["acc_avg"]}')
        
        # print and log the loss and accuracy
        avg_loss = total_loss / len(train_dataloader)
        avg_accuracy = total_accuracy / len(train_dataloader)

        print(f'Epoch {epoch}: loss={avg_loss}, accuracy={avg_accuracy}')
        logger.add_scalar('loss', avg_loss, epoch)
        logger.add_scalar('accuracy', avg_accuracy, epoch)

        # test accuracy every 25 epochs
        if epoch % 2 == 0 and epoch != 0:
            evaluate_model(model, val_dataloader, logger, epoch, loss_fn, eval_metric, device, mode='val')

        # save the model every 100 epochs
        if epoch % 2 == 0 and epoch != 0:
            checkpoint_save(model, savedir, epoch)

    # save the model
    checkpoint_save(model, savedir, epoch)

    logtestdir = os.path.join('./logs_long/test', str(batch_num))
    logger_test = SummaryWriter(logtestdir)

    # test model on a new batch / a larger fraction of the dataset
    # test_dataset = CivilCommentsWPDS(magic=seed).get_subset('test', test_frac, transform=transform) # on a larger fraction of the dataset
    for j in range(test_batch_num):
        test_dataset = CivilCommentsWPDS(magic=seed).get_batch(t=j, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

        prob_train, prob_test = get_data_distribution(train_dataset, test_dataset)
        x = np.arange(0, prob_train.shape[0])
        # w_distance_metric = w_distance(x, prob_train, prob_test)
        kl_distance_metric_1 = epsilon_kl_divergence(prob_test, prob_train, epsilon=0.1)
        kl_distance_metric_2 = epsilon_kl_divergence(prob_test, prob_train, epsilon=0.2)
        kl_distance_metric_3 = epsilon_kl_divergence(prob_test, prob_train, epsilon=0.3)
        kl_distance_metric_4 = epsilon_kl_divergence(prob_test, prob_train, epsilon=0.4)

        print('KL distance between train and eval datasets:', kl_distance_metric_1, kl_distance_metric_2, kl_distance_metric_3, kl_distance_metric_4)

        logger_test.add_scalar('kl_distance_metric_1', kl_distance_metric_1, j)
        logger_test.add_scalar('kl_distance_metric_2', kl_distance_metric_2, j)
        logger_test.add_scalar('kl_distance_metric_3', kl_distance_metric_3, j)
        logger_test.add_scalar('kl_distance_metric_4', kl_distance_metric_4, j)
        # print('Wassertein distance between train and eval datasets:', w_distance_metric)

        test_loss, test_accuracy = evaluate_model(model, test_dataloader, logger_test, j, loss_fn, eval_metric, device, checkpoint=os.path.join(savedir, 'checkpoint-{:06d}.pth'.format(epoch)), mode='test')

    return test_loss, test_accuracy, kl_distance_metric_1

'''
def get_data_distribution(dataset_1, dataset_2):
    """
        prob_1_m: prob dis of $beta$ porpotion of batch; [2, dim]
        prob_2_m: prob dis from $beta-alpha$ to $beta$ of batch [2, dim]

        In this case, the prob is multinomial
    """
    # initialize the distribution
    prob_1_m = np.zeros(dataset_1.dataset.n_classes)
    prob_2_m = np.zeros(dataset_2.dataset.n_classes)

    y1_array = np.array(dataset_1.dataset.y_array)
    y2_array = np.array(dataset_2.dataset.y_array)

    # assign distribution to classes
    for i in range(dataset_1.dataset.n_classes):
        prob_1_m[i] = len(y1_array[y1_array == i]) / len(y1_array)
    
    for i in range(dataset_2.dataset.n_classes):
        prob_2_m[i] = len(y2_array[y2_array == i]) / len(y2_array)

    return prob_1_m, prob_2_m
'''

if __name__ == '__main__':
    loss, accuracy, distance = train()