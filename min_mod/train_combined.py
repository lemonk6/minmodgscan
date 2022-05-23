import argparse
import torch
import numpy as np
import cma
from models import MLP
import torch.nn as nn
from utils import load_dataset, load_data, gather_sequence, preprocess_adverb_dataset, stats
from preprocess import Dataset
from torch.utils.tensorboard import SummaryWriter

def update(att_accs, losses, att, es):
    '''
    Runs one CMA-ES update
    :param att_accs: attention match accuracies
    :param losses: cross-entropy training loss
    :param att: the selective attention matrix to be optimized
    :param es: the evolutionary strategy
    :return: updated selective attention matrix
    '''
    if not args.indirect_feedback: # if auxiliary feedback, use attention match accuracy for optimization
        rew = -att_accs
    else: # otherwise use the cross-entropy loss of agent's controller network
        rew = losses
    es.tell(att, rew)
    att = es.ask()
    return att


def init_es(es_nr):
    '''
    Initializes the CMA-ES.
    :param es_nr: population size
    :return: initial selective attention matrix and the evolutionary strategy
    '''
    es = cma.CMAEvolutionStrategy(
        np.zeros(args.lang_dim * args.feat_dim), 0.1, {"popsize": es_nr}
    )
    att = es.ask()
    return att, es


def init_models(att, run):
    '''
    Initializes as many models as there are members of the CMA-ES population.
    :param att: selective attention matrices for the models
    :param run: current run
    :return: list of initialized models with optimizers and schedulers
    '''
    list_of_models, list_of_opts, list_of_sched = [], [], []
    for m in range(args.popsize):
        torch.manual_seed(run)
        model = MLP(feature_dim=args.feat_dim, device=args.device,
                    selective_attention=args.selective_attention, grid_size=args.grid_size,
                    action_attention=action_attention, sub_pos=sub_pos).to(device)
        model.train()
        list_of_models.append(model)
        list_of_models[m].att.weight.data = torch.from_numpy(att[m].reshape(args.feat_dim, args.lang_dim)).float().to(
            device)
        optimizer = torch.optim.Adamax(model.parameters(), weight_decay=args.weight_decay, lr=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=epochs,
                                                        steps_per_epoch=train_features.shape[0] // args.batch_size,
                                                        cycle_momentum=False, max_lr=0.01)
        list_of_opts.append(optimizer)
        list_of_sched.append(scheduler)
    return list_of_models, list_of_opts, list_of_sched


def run_batch(list_of_models, list_of_opts, batch_states, batch_features, batch_targets, batch_labels, loss_fn,
              att):
    '''
    Runs one batch update for all models/members of the CMA-ES population.
    :param list_of_models: models to the trained
    :param list_of_opts: the models' optimizers
    :param batch_states: batch grid states
    :param batch_features: batch input features
    :param batch_targets: batch targets
    :param batch_labels: batch labels
    :param loss_fn: loss function
    :param att: selective attention matrices
    :return: losses and attention match accuracies for the batch
    '''
    losses = np.zeros(args.popsize)
    att_accs = np.zeros(args.popsize)
    for m in range(len(list_of_models)):
        if args.selective_attention:
            list_of_models[m].att.weight.data = torch.from_numpy(
                att[m].reshape(args.feat_dim, args.lang_dim)).float().to(device)
        list_of_opts[m].zero_grad()
        if not args.perfect_attention:
            outputs, target_pred, states = list_of_models[m](batch_states, batch_features)
        else:
            outputs, target_pred, states = list_of_models[m](batch_states, batch_features, batch_targets.reshape(-1, 2))
        correct = (target_pred == batch_targets.reshape(-1, 2)).float().mean()
        loss = loss_fn(outputs, batch_labels)
        loss.backward()
        list_of_opts[m].step()
        losses[m] = np.mean(loss.item())
        att_accs[m] = correct
    return losses, att_accs


def dev_eval(list_of_models, loss_fn):
    '''
    Evaluates models on the dev set.
    :param list_of_models: models to be evaluated
    :param loss_fn: loss function
    :return: loss on the dev set
    '''
    dev_res = np.zeros(args.popsize)
    for m in range(len(list_of_models)):
        list_of_models[m].eval()
        if not args.perfect_attention:
            outputs, target_pred, states = list_of_models[m](dev_states, dev_features)
        else:
            outputs, target_pred, states = list_of_models[m](dev_states, dev_features, dev_locations.reshape(-1, 2))
        loss = loss_fn(outputs, dev_labels)
        dev_res[m] = loss.item()
    return dev_res

def save_model(min_rew, dev_losses, list_of_models, run):
    '''
    Saves the best model/member of the population
    :param min_rew: best performance so far
    :param dev_losses: cross-entropy losses on the dev set
    :param list_of_models: all models/members of the population
    :param run: current run
    :return: new best performance
    '''
    min_arg = np.argmin(dev_losses)
    new_min_rew = min_rew
    if dev_losses[min_arg] < min_rew:
        print("Saving model")
        torch.save(list_of_models[min_arg].state_dict(),
                   args.save_directory + "/" + args.train_split + "_" + str(run))
        new_min_rew = dev_losses[min_arg]
    return new_min_rew

def train(att, run, es, writer):
    '''
    Runs training for the given number of epochs.
    :param att: selective attention matrices
    :param run: current run
    :param es: evolutionary strategy
    :param writer: tensorboard logger
    :return:
    '''
    list_of_models, list_of_opts, list_of_sched = init_models(att, run)
    min_rew = 100000
    loss_fn = nn.CrossEntropyLoss()
    for e in range(epochs):
        rand_perm = torch.randperm(len(train_labels))
        epoch_acc, epoch_loss = [], []
        for i in range(0, rand_perm.shape[0] - rand_perm.shape[0] % args.batch_size, args.batch_size):
            batch_states = train_states[rand_perm[i:i + args.batch_size]]
            batch_features = train_features[rand_perm[i:i + args.batch_size]]
            batch_targets = train_locations[rand_perm[i:i + args.batch_size]]
            batch_labels = train_labels[rand_perm[i:i + args.batch_size]]
            losses, att_accs = run_batch(list_of_models, list_of_opts, batch_states, batch_features,
                                         batch_targets, batch_labels, loss_fn, att)
            epoch_acc.append(att_accs)
            epoch_loss.append(losses)
            for sched in list_of_sched:
                sched.step()
            if args.selective_attention and (not args.indirect_feedback or (
                    i // args.batch_size + e * updates_per_epoch) >= 500):
                att = update(att_accs, losses, att, es)
        dev_loss = dev_eval(list_of_models, loss_fn)
        stats(np.array(epoch_loss).mean(axis=0), dev_loss.mean(), e, writer)
        min_rew = save_model(min_rew, dev_loss, list_of_models, run)
        for m in list_of_models:
            m.train()
    last_min = np.argmin(dev_loss)
    torch.save(list_of_models[last_min].state_dict(),
               args.save_directory + "/" + args.train_split + "_last_" + str(run))


def init_runs():
    '''
    Initialize 10 training runs.
    :return:
    '''
    for run in range(args.runs):
        torch.manual_seed(run)
        writer = SummaryWriter(log_dir="%s/run_%s" % (args.log_dir, str(run)))
        if args.selective_attention:
            att, es = init_es(args.popsize)
        else:
            att = [np.random.rand(args.lang_dim, args.feat_dim)]
        train(att, run, es, writer)
        writer.flush()
        writer.close()

def init_data():
    '''
    Initialize args, train, dev, and split H data
    :return: initialized args, train, dev, and split H data
    '''
    action_attention = args.action_attention.lower() == "true"
    sub_pos = args.sub_pos.lower() == "true"
    device = torch.device(args.device)
    relu = torch.nn.ReLU()
    train_features, train_states, train_labels, train_locations, _ = load_data(
        args.data_folder + "/" + args.train_split + ".pkl", device)
    updates_per_epoch = train_features.shape[0] // args.batch_size
    if args.indirect_feedback:
        epochs = args.epochs + -(500 // -updates_per_epoch)
    else:
        epochs = args.epochs
    dev_features, dev_states, dev_labels, dev_locations, _ = load_data(args.data_folder + "/" + args.dev_split + ".pkl",
                                                                       device)
    return action_attention, sub_pos, device, relu, train_features, train_states, train_labels, train_locations, \
           epochs, dev_features, dev_states, dev_labels, dev_locations, updates_per_epoch

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
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--lang_dim", type=int, default=64)
    parser.add_argument("--feat_dim", type=int, default=16)
    parser.add_argument("--indirect_feedback", type=bool, default=False)
    parser.add_argument("--selective_attention", type=bool, default=True)
    parser.add_argument("--perfect_attention", type=bool, default=False)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--log_dir", type=str, default="../runs")
    parser.add_argument("--grid_size", type=int, default=6)
    parser.add_argument("--action_attention", type=str, default="true")
    parser.add_argument("--sub_pos", type=str, default="False")
    args = parser.parse_args()
    action_attention, sub_pos, device, relu, train_features, train_states, train_labels, train_locations, \
    epochs, dev_features, dev_states, dev_labels, dev_locations, updates_per_epoch = init_data()
    init_runs()
