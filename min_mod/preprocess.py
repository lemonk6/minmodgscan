import torch
from min_mod.models import ESN
from min_mod.GroundedScan.dataset import GroundedScan
import pickle
import numpy as np
import argparse
import random

class Dataset:
    '''
    Stores steps with their labels, input features, grid states, target locations, sequence ID and input commands.
    '''
    def __init__(self, labels, input_features, input_states, target_locations, ids, commands):
        self.labels = labels
        self.input_features = input_features
        self.grid_states = input_states
        self.target_locations = target_locations
        self.ids = ids
        self.commands = commands


def get_lang_input(command):
    '''
    Turns input command into embedding using the Echo State Network.
    :param command: language command
    :return: encoded command
    '''
    hid_state_lang = language_net.init_hidden()
    tokens = []
    for word in command.split(","):
        tokens += word.split(" ")
    token_sequence = [token_to_id[token] for token in tokens]
    token_sequence = torch.tensor([token_sequence], dtype=torch.int64, requires_grad=False).to(args.device)
    for t in range(len(tokens)):
        token_input = token_sequence[:, t]
        token_input = token_input.data.clone().detach()
        token_input = torch.nn.functional.one_hot(token_input, len(all_tokens))
        hid_state_lang, linear_lang = language_net(token_input.float().to(language_net.device), hid_state_lang)
    return linear_lang


def get_act_input(last_output, obs, target_feats, last_orientation):
    '''
    Extracts agent location, agent orientation, and target location
    :param last_output: agent's previous action
    :param obs: grid state
    :param target_feats: feature vector of the target object
    :param last_orientation: the agent's previous orientation
    :return: agent location, agent orientation, and target location
    '''
    agent_info = obs[:, :, 11:]
    agent_loc = np.where(agent_info[:, :, 0] == 1)
    agent_vector = (agent_info[agent_loc[0], agent_loc[1], 1:]).reshape(1, -1)
    target_loc = np.where(np.all(obs[:, :, :11] == target_feats, axis=2))
    return agent_loc, agent_vector, target_loc


def single_step(command, labels, input_states, obs, last_output, target_feats, lang_input, ids, input_features, i,
                target_locations, last_steps, last_orients, last_orientation):
    '''
    Collects input features for one action step.
    :param command: correct action output
    :param labels: list that collects all labels
    :param input_states: list that collects all grid states
    :param obs: grid state
    :param last_output: agent's previous output
    :param target_feats: features of the target object
    :param lang_input: language embedding
    :param ids: list that collects action sequence ids
    :param input_features: list that collects input features
    :param i: action sequence index
    :param target_locations: list that collects target locations
    :param last_steps: agent's past steps
    :param last_orients: agent's past orientations
    :param last_orientation: agent's previous orientation
    :return: agent's most recent output action and orientation
    '''
    command_one_hot = torch.zeros(1, 6)
    mapped_action = actions_mapping[command]
    labels.append(mapped_action)
    command_one_hot[0, mapped_action] = 1
    input_states.append(obs)
    agent_loc, agent_vector, target_loc = get_act_input(last_output, obs, target_feats, last_orientation)
    combined_input = np.concatenate(
        (lang_input, last_steps.reshape(1, -1), last_orients.reshape(1, -1),
         np.array([agent_loc[0] / 5, agent_loc[1] / 5]).T, agent_vector), axis=1)
    input_features.append(combined_input)
    target_locations.append(target_loc)
    ids.append(i)
    last_output = command_one_hot
    return last_output, torch.from_numpy(agent_vector)


def run_filters(examples):
    '''
    Filters out specified shape-color and/or verb-adverb combinations.
    :param examples: sequences in the dataset
    :return: filtered sequences
    '''
    filter_color_shape = len(shape_color_combos) > 1 and "not" not in shape_color_combos
    filter_color_shape_neg = len(shape_color_combos) > 1 and "not" in shape_color_combos
    filter_verb_adverb = len(verb_adverb_combos) > 1 and "not" not in verb_adverb_combos
    filter_verb_adverb_neg = len(verb_adverb_combos) > 1 and "not" in verb_adverb_combos
    if filter_color_shape or filter_verb_adverb or filter_color_shape_neg or filter_verb_adverb_neg:
        filtered_examples = []
        included = set()
        for i, example in enumerate(examples):
            target = " ".join(example["referred_target"].split()[-2:])
            verb_adverb = " ".join([example["verb_in_command"], example["manner"]])
            if (filter_color_shape and target in shape_color_combos) or (
                    filter_verb_adverb and verb_adverb in verb_adverb_combos) or (
                    filter_color_shape_neg and target not in shape_color_combos) or (
                    filter_verb_adverb_neg and verb_adverb not in verb_adverb_combos):
                filtered_examples.append(example)
                included.add(verb_adverb)
    else:
        return examples
    return filtered_examples


def select_subset(examples):
    '''
    Creates a subset of a given dataset.
    :param examples: dataset for which to create a subset
    :return: indices of the sequences in the subset
    '''
    if make_subset:
        num_examples = int(args.subset_size * len(examples))
        k_random_indices = random.sample(range(0, len(examples)), k=num_examples)
        for k_idx in range(args.k):
            k_random_indices.append(k_idx)
        k_random_indices.append(0) # include the one "cautiously" example
    else:
        k_random_indices = range(len(examples))
    return k_random_indices


def process_situations(examples_all):
    '''
    Preprocesses the situations in the given dataset.
    :param examples_all: all situations in the dataset in raw form
    :return: grid states, input features, labels, target locations, sequence ids and language commands for all steps in
    the dataset
    '''
    input_states, input_features, labels, target_locations, ids, commands = [], [], [], [], [], []
    examples = run_filters(examples_all)
    k_random_indices = select_subset(examples)
    for i in sorted(k_random_indices):
        example = examples[i]
        sit = dataset.initialize_rl_example(example, simplified_world_representation=True)
        obs = np.array(sit[1])
        this_steps = torch.zeros((20, 6))
        this_orientation = torch.zeros((20, 4))
        command = example['command']
        lang_input = get_lang_input(command)
        commands.append(command)
        last_output = torch.zeros((1, 6))
        last_orientation = torch.zeros((1, 4))
        target_feats = obs[int(example['situation']['target_object']['position']['row']), int(
            example['situation']['target_object']['position']['column'])][:11]
        for action in example['target_commands'].split(","):
            last_output, agent_vector = single_step(action, labels, input_states, obs, last_output,
                                                                   target_feats, lang_input, ids, input_features, i,
                                                                   target_locations, this_steps[-20:].ravel(),
                                                                   this_orientation[-20:].ravel(),
                                                                   last_orientation)
            last_orientation = agent_vector
            this_steps = torch.cat((this_steps, last_output))
            this_orientation = torch.cat((this_orientation, agent_vector))
            obs, _, _ = dataset.take_step(action, simple_situation_representation=True)
    return input_states, input_features, labels, target_locations, ids, commands


def init_tokens():
    '''
    Initialize word-to-index mappings for command words and output actions.
    :return: mappings for command words and output actions
    '''
    all_tokens = ['zigzagging', 'small', 'while', 'a', 'push', 'to', 'yellow', 'hesitantly', 'cautiously', 'square',
                  'red',
                  'green', 'cylinder', 'blue', 'walk', 'circle', 'pull', 'big', 'spinning']
    token_to_id = {token: idx for idx, token in enumerate(all_tokens)}
    actions_mapping = {
        "turn right": 0,
        "turn left": 1,
        "walk": 2,
        "push": 3,
        "pull": 4,
        "stay": 5
    }
    return all_tokens, token_to_id, actions_mapping


def init_language_ESN(device):
    '''
    Initializes the Echo State Network used to create command embeddings.
    :param device: cpu or cuda
    :return: initialized ESN
    '''
    language_net = ESN(n_in=19, n_res=400, n_out=19,
                       ro_hidden_layers=1,
                       leaking_rate=0.1,
                       spec_radius=0.99,
                       lin_size=64,
                       density=0.01,
                       device=device,
                       is_feedback=True,
                       fb_scaling=0.0,
                       batch_size=1)
    language_net.to(device)
    try:
        language_net.load_state_dict(torch.load("../trained_models/language_ESN"))
    except:
        torch.save(language_net.state_dict(), "../trained_models/language_ESN")
    for param in language_net.parameters():
        param.requires_grad = False
    language_net.eval()
    return language_net


def generate_dataset():
    '''
    Generates a preprocessed dataset from a given raw dataset.
    :return:
    '''
    if args.dataset_name is not None:
        file_name = args.save_directory + "/%s.pkl" % args.dataset_name
    elif make_subset:
        file_name = args.save_directory + "/%s_subset_%s.pkl" % (args.compositional_split, str(args.subset_size))
    else:
        file_name = args.save_directory + "/%s.pkl" % args.compositional_split
    open_file = open(file_name, "wb")
    situations = list(dataset.get_raw_examples(split=args.compositional_split))
    input_states, input_features, labels, target_locations, ids, commands = process_situations(situations)
    ds = Dataset(labels, input_features, input_states, target_locations, ids, commands)
    pickle.dump(ds, open_file)
    open_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../data_raw/dataset.txt", required=False)
    parser.add_argument("--compositional_split", type=str, default="train", required=False)
    parser.add_argument("--make_subset", type=str, default="True", required=False)
    parser.add_argument("--subset_size", type=float, default=0.1, required=False)
    parser.add_argument("--shape_color_filter", type=str, default="", required=False)
    parser.add_argument("--adverb_verb_filter", type=str, default="", required=False)
    parser.add_argument("--save_directory", type=str, default="../data_processed", required=False)
    parser.add_argument("--k", type=int, default=1, required=False)
    parser.add_argument("--dataset_name", type=str, default=None, required=False)
    parser.add_argument("--device", type=str, default="cpu", required=False)
    args = parser.parse_args()
    make_subset = args.make_subset.lower() == "true"
    dataset_path = args.dataset_path
    dataset = GroundedScan.load_dataset_from_file(dataset_path, save_directory=args.save_directory, k=args.k)
    shape_color_combos = [x.strip().lower() for x in args.shape_color_filter.split(",")]
    verb_adverb_combos = [x.strip().lower() for x in args.adverb_verb_filter.split(",")]

    all_tokens, token_to_id, actions_mapping = init_tokens()
    language_net = init_language_ESN(torch.device(args.device))
    generate_dataset()
