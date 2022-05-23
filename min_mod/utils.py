import pickle
from min_mod.preprocess import Dataset
import numpy as np
from min_mod.GroundedScan.dataset import GroundedScan
import torch

def gather_sequence(id, idx, ids, input_states, input_features, target_locations, labels):
    '''
    Gathers inputs belonging to the same action sequence.
    :param id: ID of the action sequence
    :param idx: step index for the whole dataset
    :param ids: all action sequence indeces of the dataset
    :param input_states: all input states
    :param input_features: all input features
    :param target_locations: all target locations
    :param labels: all labels
    :return: input states, features, target locations, labels for the action sequence; updated step index
    '''
    buffer_states, buffer_features, buffer_target_locs, buffer_labels = [], [], [], []
    while idx < labels.shape[0] and ids[idx] == id:
        buffer_states.append(input_states[idx])
        buffer_features.append(input_features[idx])
        buffer_target_locs.append(target_locations[idx])
        buffer_labels.append(labels[idx])
        idx += 1
    return buffer_states, buffer_features, buffer_target_locs, buffer_labels, idx

def preprocess_adverb_dataset(path_name, device):
    '''
    Retrieves the states, input features, target locations, and labels of the split H data.
    :param path_name: path to the preprocessed split H data
    :param device: spu or cuda
    :return: loaded action sequences for the split H dataset
    '''
    input_features, input_states, labels, target_locations, ids = load_data(path_name, device)
    idx = 0
    sequences = []
    for id in torch.unique(ids):
        buffer_states, buffer_features, buffer_target_locs, buffer_labels, idx = gather_sequence(id, idx, ids,
                                                                                                 input_states,
                                                                                                 input_features,
                                                                                                 target_locations,
                                                                                                 labels)
        sequences.append((torch.stack(buffer_states), torch.stack(buffer_features), torch.stack(buffer_target_locs),
                          torch.stack(buffer_labels)))
    return sequences

def stats(epoch_loss, dev_loss, e, writer):
    '''
    Prints the train and dev loss
    :param epoch_loss: train loss
    :param dev_loss: dev loss
    :param e: epoch
    :param writer: tensorboard logger
    :return:
    '''
    mean_train_loss = np.array(epoch_loss).mean(axis=0)
    mean_dev_loss = dev_loss
    print("Epoch ", e)
    print("Avg train loss: ", mean_train_loss)
    print("Dev loss: ", mean_dev_loss)
    writer.add_scalar("Loss/train", mean_train_loss, e)
    writer.add_scalar("Loss/dev", mean_dev_loss, e)

def load_dataset(file_name):
    '''
    Loads the specified dataset in pickle form.
    :param file_name: dataset to be loaded
    :return: loaded dataset
    '''
    open_file = open(file_name, "rb")
    examples = pickle.load(open_file)
    open_file.close()
    return examples

def load_data(path_name, device):
    '''
    Loads the input features, grid states, labels, target locations and sequence IDs of a specified dataset.
    :param path_name: dataset to load
    :param device: cpu or cuda
    :return: input features, grid states, labels, target locations and sequence IDs of the dataset
    '''
    examples = load_dataset(path_name)
    states = torch.from_numpy(np.array(examples.grid_states)).float().to(device)
    features = torch.from_numpy(np.array(examples.input_features)).float().to(device)  
    labels = torch.from_numpy(np.array(examples.labels)).to(device)  
    locations = torch.from_numpy(np.array(examples.target_locations)).to(device)
    ids = torch.from_numpy(np.array(examples.ids)).to(device)
    return features, states, labels, locations, ids
