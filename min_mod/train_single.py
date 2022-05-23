import argparse
import torch
import numpy as np
from models import MLP
import torch.nn as nn
from utils import load_dataset, load_data, gather_sequence, stats
from preprocess import Dataset
from torch.utils.tensorboard import SummaryWriter


def init_model():
    '''
    Initialize the model
    :return: initialized model and optimizer
    '''
    model = MLP(feature_dim=args.feat_dim, device=args.device, selective_attention=selective_attention).to(device)
    if args.pretrained_model != None:
        model.load_state_dict(torch.load(args.save_directory + "/" + args.pretrained_model))
    model.train()
    optimizer = torch.optim.Adamax(model.parameters(), weight_decay=0.0001, lr=0.01)
    return model, optimizer

def run_batch(batch_states, batch_features, batch_labels, batch_targets, loss_fn, opt, model):
    '''
    Run one batch update
    :param batch_states: batch states
    :param batch_features: batch features
    :param batch_labels: batch labels
    :param batch_targets: batch targets
    :param loss_fn: loss function
    :param opt: optimizer
    :param model: model to be trained
    :return: cross-entropy loss for the batch
    '''
    opt.zero_grad()
    if perfect_attention:
        outputs, target_pred, states = model(batch_states, batch_features, batch_targets.reshape(-1, 2))
    else:
        outputs, target_pred, states = model(batch_states, batch_features)
    loss = loss_fn(outputs, batch_labels)
    loss.backward()
    opt.step()
    return loss.item()


def eval_dev_loss(model, loss_fn, dev_states, dev_features, dev_locations, dev_labels):
    '''
    Evaluate model on dev set.
    :param model: model to be evaluated
    :param loss_fn: loss function
    :param dev_states: grid states in dev set
    :param dev_features: input features in dev set
    :param dev_locations: target locations in dev set
    :param dev_labels: labels in dev set
    :return: cross-entropy loss for the dev set
    '''
    model.eval()
    if perfect_attention:
        outputs, target_pred, states = model(dev_states, dev_features, dev_locations.reshape(-1, 2))
    else:
        outputs, target_pred, states = model(dev_states, dev_features)
    loss = loss_fn(outputs, dev_labels)
    model.train()
    dev_res = loss.item()
    return dev_res

def train(run, model, scheduler, writer, opt):
    '''
    Trains the model for the given number of epochs.
    :param run: current run
    :param model: model to be trained
    :param scheduler: the model's learning rate scheduler
    :param writer: tensorboard logger
    :param opt: the model's optimizer
    :return:
    '''
    smallest_loss = 100000
    loss_fn = nn.CrossEntropyLoss()
    for e in range(args.epochs):
        rand_perm = torch.randperm(len(train_labels))
        train_losses = []
        for i in range(0, rand_perm.shape[0] - rand_perm.shape[0] % args.batch_size, args.batch_size):
            batch_states = train_states[rand_perm[i:i + args.batch_size]]
            batch_features = train_features[rand_perm[i:i + args.batch_size]]
            batch_targets = train_locations[rand_perm[i:i + args.batch_size]]
            batch_labels = train_labels[rand_perm[i:i + args.batch_size]]
            loss = run_batch(batch_states, batch_features, batch_labels, batch_targets, loss_fn, opt, model)
            train_losses.append(loss)
            scheduler.step()
        dev_loss = eval_dev_loss(model, loss_fn, dev_states, dev_features, dev_locations, dev_labels)
        stats(train_losses, dev_loss, e, writer)
        if dev_loss < smallest_loss:
            print("Saving model")
            torch.save(model.state_dict(),
                       args.save_directory + "/" + args.train_split + "_" + str(run))
            smallest_loss = dev_loss

def init_runs():
    '''
    Initializes training runs.
    :return:
    '''
    for run in range(args.runs):
        writer = SummaryWriter(log_dir="%s/run_%s" % (args.log_dir, str(run)))
        torch.manual_seed(run)
        model, opt = init_model()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, epochs=args.epochs,
                                                        steps_per_epoch=train_features.shape[0] // args.batch_size,
                                                        cycle_momentum=False, max_lr=0.01)
        train(run, model, scheduler, writer, opt)
        writer.flush()
        writer.close()
        torch.save(model.state_dict(),
                   args.save_directory + "/" + args.train_split + "_perfect_attention_" + args.perfect_attention + "_last_" + str(
                       run))

def init_data():
    '''
    Initializes args, train, dev and split H data.
    :return: initialized args, train, dev and split H data
    '''
    perfect_attention = args.perfect_attention.lower() == "true"
    selective_attention = args.selective_attention.lower() == "true"
    device = torch.device(args.device)
    relu = torch.nn.ReLU()
    train_features, train_states, train_labels, train_locations, _ = load_data(
        args.data_folder + "/" + args.train_split + ".pkl", device)
    dev_features, dev_states, dev_labels, dev_locations, _ = load_data(
        args.data_folder + "/" + args.dev_split + ".pkl", device)
    return perfect_attention, device, selective_attention, relu, train_features, \
           dev_features, train_states, train_labels, train_locations, dev_states, dev_labels, dev_locations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="../data_processed")
    parser.add_argument("--train_split", type=str, default="train_subset_0.01")
    parser.add_argument("--dev_split", type=str, default="dev")
    parser.add_argument("--save_directory", type=str, default="../trained_models")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lang_dim", type=int, default=64)
    parser.add_argument("--feat_dim", type=int, default=16)
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--perfect_attention", type=str, default="True")
    parser.add_argument("--selective_attention", type=str, default="True")
    parser.add_argument("--log_dir", type=str, default="../runs")
    args = parser.parse_args()
    perfect_attention, device, selective_attention, relu, train_features, \
    dev_features, train_states, train_labels, train_locations, dev_states, dev_labels, dev_locations = init_data()
    init_runs()
