import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.civilcomments_wpds import CivilCommentsWPDS
from civilcomments_utils import *


def main():
    # set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # set up the model
    model_name = 'distilbert-base-uncased'
    num_labels = 2
    model = initialize_model(model_name, num_labels).to(device)
    model.train()

    # set up the dataset
    seed = 2022
    frac = 0.5 # fraction of the dataset to use
    transform = initialize_bert_transform(model_name, 512)
    train_dataset = CivilCommentsWPDS(magic=seed).get_subset('train', frac, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # set up testing dataset
    val_dataset = CivilCommentsWPDS(magic=seed).get_subset('val', frac, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    # set up the loss function
    loss_fn = initialize_loss_function('xent')

    # set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # set up the evaluation metric
    eval_metric = initialize_eval_metric('acc')

    # set up the summary writer
    logdir = os.path.join('./logs/', get_cur_time())
    savedir = os.path.join('./checkpoints/', get_cur_time())
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    logger = SummaryWriter(logdir)

    # set up the training loop
    num_epochs = 10
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_accuracy = 0
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            attention_mask, is_good, detailed_description = batch
            attention_mask = attention_mask.to(device)
            detailed_description = detailed_description.to(device)
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
        if epoch % 25 == 0:
            evaluate_model(model, val_dataloader, logger, epoch, loss_fn, eval_metric, device)

        # save the model every 100 epochs
        if epoch % 100 == 0:
            checkpoint_save(model, savedir, epoch)

    # save the model
    checkpoint_save(model, savedir, epoch)

    # test model on a new batch / a larger fraction of the dataset
    batch_num = 0 # intended batch number if getting a batch
    test_dataset = CivilCommentsWPDS(magic=seed).get_subset('test', 1.0, transform=transform) # on a larger fraction of the dataset
    # test_dataset = CivilCommentsWPDS(magic=seed).get_batch(t=batch_num, transform=transform) # on a batch
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    evaluate_model(model, test_dataloader, logger, 0, loss_fn, eval_metric, device, checkpoint=os.path.join(savedir, 'checkpoint-{:06d}.pth'.format(epoch)))



if __name__ == '__main__':
    main()