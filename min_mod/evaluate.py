import argparse
import pickle
import torch
import numpy as np
from models import MLP
from utils import load_dataset, Dataset
import os


def gather_sequence(id, i):
    '''
    Gathers inputs belonging to the same action sequence.
    :param id: ID of the action sequence to be gathered.
    :param i: step index for the total specified dataset
    :return: grid states, input features, target locations, buffer labels for the action sequence; updated step index
    '''
    buffer_states, buffer_features, buffer_target_locs, buffer_labels = [], [], [], []
    while i < labels.shape[0] and ids[i] == id:
        buffer_states.append(input_states[i])
        buffer_features.append(input_features[i])
        buffer_target_locs.append(target_locations[i])
        buffer_labels.append(labels[i])
        i += 1
    return buffer_states, buffer_features, buffer_target_locs, buffer_labels, i


def safe_div(x, y):
    '''
    Safe division to avoid errors for division by zero.
    :param x: dividend
    :param y: divisor
    :return: quotient if divisor is not zero, zero otherwise
    '''
    if y == 0: return 0
    return x / y


def evaluate():
    '''
    Runs through all sequences in the specified dataset and records exact match accuracy, attention accuracy, and exact
    match accuracy if the target object is correctly identified.
    '''
    correct = correct_if_correct_attention = correct_attention = incorrect = idx = 0
    correct_att = []
    for id in np.unique(ids):
        buffer_states, buffer_features, buffer_target_locs, buffer_labels, idx = gather_sequence(id, idx)
        if args.perfect_attention:
            outputs, target_pred, _ = model(torch.tensor(buffer_states).float().to(device),
                                            torch.tensor(buffer_features).float().to(device),
                                            torch.tensor(buffer_target_locs).reshape(-1, 2).to(device))
        else:
            outputs, target_pred, _ = model(torch.tensor(buffer_states).float().to(device),
                                            torch.tensor(buffer_features).float().to(device))
        correct_att.append((target_pred == torch.tensor(buffer_target_locs).to(device).reshape(-1, 2)).float().mean())
        if torch.all(torch.argmax(outputs, dim=1) == torch.tensor(buffer_labels).to(device)):
            correct += 1
        else:
            incorrect += 1
        if torch.all(target_pred == torch.tensor(buffer_target_locs).to(device).reshape(-1, 2)):
            correct_attention += 1
            if torch.all(torch.argmax(outputs, dim=1) == torch.tensor(buffer_labels).to(device)):
                correct_if_correct_attention += 1

    attention_accuracy = torch.mean(torch.tensor(correct_att))
    sequence_accuracy = safe_div(correct, correct + incorrect)
    seq_accuracy_correct_attention_only = safe_div(correct_if_correct_attention, correct_attention)
    return attention_accuracy, sequence_accuracy, seq_accuracy_correct_attention_only


def inits():
    '''
    Loads the specified model and dataset.
    '''
    selective_attention = args.selective_attention.lower() == "true"
    sub_pos = args.sub_pos.lower() == "true"
    device = torch.device(args.device)
    directory = os.fsencode(args.model_dir_path)
    action_attention = args.action_attention.lower() == "true"
    test_examples = load_dataset(args.data_folder + "/" + args.test_split + ".pkl")
    seq_results, att_results, seq_correct_att_results = [], [], []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        model = MLP(device=device, selective_attention=selective_attention, action_attention=action_attention,
                    sub_pos=sub_pos).to(device)
        model.load_state_dict(torch.load(args.model_dir_path + "/" + filename, map_location=device))
        model.eval()
        labels = np.array(test_examples.labels)
        input_features = np.array(test_examples.input_features)
        input_states = np.array(test_examples.grid_states)
        target_locations = np.array(test_examples.target_locations)
        ids = np.array(test_examples.ids)
    return labels, input_features, input_states, target_locations, ids, model, seq_results, att_results, \
               seq_correct_att_results, device


def print_res():
    '''
    Prints evaluation results to console.
    '''
    print("Attention: ", np.array(att_results).mean(), np.array(att_results).std())
    print("Exact sequence match: ", np.array(seq_results).mean(), np.array(seq_results).std())
    print("Exact sequence match for sequences with correct attention: ",
          np.array(seq_correct_att_results).mean(), np.array(seq_correct_att_results).std())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="../data_processed")
    parser.add_argument("--test_split", type=str, default="dev")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--perfect_attention", type=bool, default=False)
    parser.add_argument("--selective_attention", type=str, default="True")
    parser.add_argument("--sub_pos", type=str, default="False")
    parser.add_argument("--action_attention", type=str, default="true")
    parser.add_argument("--model_dir_path", type=str,
                        default="../trained_models/direct_attention/direct_attention_1.0/best_dev")
    args = parser.parse_args()
    labels, input_features, input_states, target_locations, ids, model, seq_results, att_results, \
    seq_correct_att_results, device = inits()
    attention_accuracy, sequence_accuracy, seq_correct_att = evaluate()
    att_results.append(attention_accuracy.item())
    seq_results.append(sequence_accuracy)
    seq_correct_att_results.append(seq_correct_att)
    print_res()
