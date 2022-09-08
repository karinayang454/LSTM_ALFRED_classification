import tqdm
import torch
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
import json
from torch.utils.data import TensorDataset, DataLoader
from model import LSTMModel
import matplotlib.pyplot as plt

from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
    encode_data
)


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #
    # Load data
    f = open('lang_to_sem_data.json')
    data = json.load(f) #keys = ['train', 'valid_seen']
    f.close()
    train_data=[]
    val_data=[]
    for i in data['train']:
        train_data.append(i)
    for i in data['valid_seen']:
        val_data.append(i)

    # build tokenizer table and encode the data
    vocab_to_index, index_to_vocab, len_cutoff = build_tokenizer_table(train_data)
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(train_data)
    x, y_a, y_t = encode_data(train_data, vocab_to_index, len_cutoff, actions_to_index, targets_to_index)
    val_x, val_y_a, val_y_t = encode_data(val_data, vocab_to_index, len_cutoff, actions_to_index, targets_to_index)

    # convert data from np to tensors
    train_dataset_a = TensorDataset(torch.from_numpy(x), torch.from_numpy(y_a))
    train_dataset_t = TensorDataset(torch.from_numpy(x), torch.from_numpy(y_t))
    val_dataset_a = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y_a))
    val_dataset_t = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y_t))

    # define train and val dataloaders
    train_loader_a = DataLoader(train_dataset_a, shuffle=True, batch_size = args.batch_size)
    val_loader_a = DataLoader(val_dataset_a, shuffle=False, batch_size = args.batch_size)
    train_loader_t = DataLoader(train_dataset_t, shuffle=True, batch_size = args.batch_size)
    val_loader_t = DataLoader(val_dataset_t, shuffle=False, batch_size = args.batch_size)

    return train_loader_a, val_loader_a, train_loader_t, val_loader_t,\
        len(vocab_to_index), len(actions_to_index), len(targets_to_index), len_cutoff



def setup_model(args, input_dim, output_dim):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model.
    # ===================================================== #
    # input_dim = size of dictionary of embeddings
    # ouput_dim = number of outputs
    # LSTM_layers = number of tokens per input
    embedding_dim = 100
    hidden_dim = 128
    model = LSTMModel(input_dim, embedding_dim, hidden_dim, output_dim)
    return model


def setup_optimizer(args, model):
    """
    return:
        - action_criterion: loss_fn
        - target_criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return criterion, optimizer

def train_epoch(args, model_a, model_t, loader_a, loader_t, optimizer_a, optimizer_t, action_criterion,
    target_criterion, device, training=True):
    epoch_action_loss = 0.0
    epoch_target_loss = 0.0
    num_itrs_a = 0

    # keep track of the model predictions for computing accuracy
    action_preds = []
    target_preds = []
    action_labels = []
    target_labels = []

    ################# Action Trainer ###################
    for (inputs, labels) in loader_a:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)
        num_itrs_a +=1

        # calculate the loss and train accuracy and perform backprop
        actions_out = model_a(inputs)

        # calculate the action loss
        assert(len(labels[0])==1)
        # print(actions_out.squeeze())
        # print(labels[:,0])
        action_loss = action_criterion(actions_out.squeeze(), labels[:,0].long())

        # step optimizer and compute gradients during training
        if training:
            optimizer_a.zero_grad()
            action_loss.backward()
            optimizer_a.step()
        # logging
        epoch_action_loss += action_loss.item()

        # take the prediction with the highest probability
        action_preds_ = actions_out.argmax(-1)

        # aggregate the batch predictions + labels
        action_preds.extend(action_preds_.cpu().numpy())
        action_labels.extend(labels[:,0].cpu().numpy())

    action_acc = accuracy_score(action_preds, action_labels)
    num_itrs_t = 0
    ################# Target Trainer ###################
    for (inputs, labels) in loader_t:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)
        num_itrs_t += 1

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        targets_out = model_t(inputs)

        # calculate target prediction loss
        assert(len(labels[0])==1)
        target_loss = target_criterion(targets_out.squeeze(), labels[:,0].long())

        # step optimizer and compute gradients during training
        if training:
            optimizer_t.zero_grad()
            target_loss.backward()
            optimizer_t.step()

        # logging
        epoch_target_loss += target_loss.item()

        # take the prediction with the highest probability
        target_preds_ = targets_out.argmax(-1)

        # aggregate the batch predictions + labels
        target_preds.extend(target_preds_.cpu().numpy())
        target_labels.extend(labels[:,0].cpu().numpy())
    target_acc = accuracy_score(target_preds, target_labels)

    return epoch_action_loss/num_itrs_a, epoch_target_loss/num_itrs_t, action_acc, target_acc

def validate(
    args, model_a, model_t, loader_a, loader_t, optimizer_a, optimizer_t, action_criterion, target_criterion, device
):
    # set model to eval mode
    model_a.eval()
    model_t.eval()

    # don't compute gradients
    with torch.no_grad():

        val_action_loss, val_target_loss, action_acc, target_acc = train_epoch(
            args,
            model_a,
            model_t,
            loader_a,
            loader_t,
            optimizer_a,
            optimizer_t,
            action_criterion,
            target_criterion,
            device,
            training=False,
        )

    return val_action_loss, val_target_loss, action_acc, target_acc

def train(args, model_a, model_t, loaders, optimizer_a, optimizer_t, action_criterion, target_criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model_a.train()
    model_t.train()
    train_action_losses = []
    train_target_losses = []
    train_action_accs = []
    train_target_accs = []    
    val_action_losses = []
    val_target_losses = []
    val_action_accs = []
    val_target_accs = []   
    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        (   train_action_loss, train_target_loss,
            train_action_acc, train_target_acc ) = train_epoch( args,
                                                                model_a,
                                                                model_t,
                                                                loaders["train_a"],
                                                                loaders["train_t"],
                                                                optimizer_a,
                                                                optimizer_t,
                                                                action_criterion,
                                                                target_criterion,
                                                                device)

        # some logging
        print(
            f"train action loss : {train_action_loss} | train target loss: {train_target_loss}"
        )
        train_action_losses.append(train_action_loss)
        train_target_losses.append(train_target_loss)
        print(
            f"train action acc : {train_action_acc} | train target acc: {train_target_acc}"
        )
        train_action_accs.append(train_action_acc)
        train_target_accs.append(train_target_acc) 

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
                args,
                model_a,
                model_t,
                loaders["val_a"],
                loaders["val_t"],
                optimizer_a,
                optimizer_t,
                action_criterion,
                target_criterion,
                device,
            )

            print(
                f"val action loss : {val_action_loss} | val target loss: {val_target_loss}"
            )
            print(
                f"val action acc : {val_action_acc} | val target acc: {val_target_acc}"
            )
            val_action_losses.append(val_action_loss)
            val_target_losses.append(val_target_loss)
            val_action_accs.append(val_action_acc)
            val_target_accs.append(val_target_acc) 
    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy,
    # 3) validation loss, 4) validation accuracy
    # ===================================================== #
    fig = plt.figure(figsize=(10,10))

    # training loss
    plt.subplot(2, 2, 1)
    x = list(range(1, len(train_action_losses) + 1))
    plt.plot(x, train_action_losses, label = "train_loss_action")
    plt.plot(x, train_target_losses, label = "train_loss_target")
    plt.gca().set_title("Training Losses per Epoch")
    plt.legend()

    # training accuracy
    plt.subplot(2, 2, 2)
    plt.plot(x, train_action_accs, label = "train_acc_action")
    plt.plot(x, train_target_accs, label = "train_acc_target")
    plt.gca().set_title("Training Accuracies per Epoch")
    plt.legend()

    # validation loss
    plt.subplot(2, 2, 3)
    x = list(range(1, len(val_action_losses) + 1))
    plt.plot(x, val_action_losses, label = "val_loss_action")
    plt.plot(x, val_target_losses, label = "val_loss_target")
    plt.gca().set_title("Validation Losses per Epoch")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(x, val_action_accs, label = "val_acc_action")
    plt.plot(x, val_target_accs, label = "val_acc_target")
    plt.gca().set_title("Validation Accuracies per Epoch")
    plt.legend()

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig("outputs.png")

def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader_a, val_loader_a, train_loader_t, val_loader_t, \
        dict_len, actions_len, targets_len, num_tokens    = setup_dataloader(args)
    loaders = {"train_a": train_loader_a, "val_a": val_loader_a,
               "train_t": train_loader_t, "val_t": val_loader_t}

    # build model
    model_a = setup_model(args, dict_len, actions_len)
    model_t = setup_model(args, dict_len, targets_len)
    print(model_a)
    print(model_t)

    # get optimizer and loss functions
    action_criterion, optimizer_a = setup_optimizer(args, model_a)
    target_criterion, optimizer_t = setup_optimizer(args, model_t)

    if args.eval:
        val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
            args,
            model_a,
            model_t,
            loaders["val_a"],
            loaders["val_t"],
            optimizer_a,
            optimizer_t,
            action_criterion,
            target_criterion,
            device,
        )
    else:
        train(
            args, model_a, model_t, loaders, optimizer_a, optimizer_t, action_criterion, target_criterion, device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument(
        "--force_cpu", action="store_true", help="debug mode")
    parser.add_argument(
        "--eval", action="store_true", help="run eval")
    parser.add_argument(
        "--num_epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", type=int, default=5, help="number of epochs between every eval loop"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()

    main(args)
