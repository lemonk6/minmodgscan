# Using this Project

## Getting started

Install all necessary python packages with a package manager

``` pip install -r requirements.txt ```

## Data preparation

#### Generate gSCAN data
To retrieve the grounded SCAN dataset, go to Laura Ruis' [GitHub repository](https://github.com/LauraRuis/groundedSCAN#using-the-repository) and follow the steps outlined in the ReadMe under "Generating data". 
Once you have generated the ```dataset.txt``` file for the compositional splits, place it in the ```data_raw``` folder of this project.

#### Preprocess gSCAN data
To preprocess the data, run the ``preprocess.py`` script in ```min_mod```. The script takes several optional arguments:
* `--dataset_path` specifies where the raw `dataset.txt`is located. By default it is set to `../data_raw/dataset.txt.
* `--save_directory` specifies where to save the preprocessed dataset. By default this is set to `../data_processed`.
* `--dataset_name` specifies under what name to save the preprocessed dataset. By default, datasets will be saved under the name of the compositional split and the size of its subset (if specified).
* `--compositional_split` specifies which compositional split to preprocess. The options are `train, test, dev, visual, situational_1, situational_2, contextual, adverb_1, adverb_2,` and `visual_easier`. The default is `train`.
* `--make_subset` specifies whether to create a subset of the given compositional split. Defaults to `True`.
* `--subset_size` specifies the size of the desired subset. For example, to create 5% subset of the training set, use `--compositional_split train --make_subset True --subset_size 0.05`
* `--shape_color_filter` specifies whether to filter out certain target shape-color combinations. For example, to exclude red squares, yellow squares, green cylinders, and blue circles, you could use `--shape_color_filter="not, red square, yellow square, green cylinder, blue circle"`. 
To include _only_ red squares, yellow squares, green cylinders, and blue circles, you could use `--shape_color_filter="red square, yellow square, green cylinder, blue circle"`.
* `--adverb_color_filter` specifies whether to filter out sequences with certain adverb-verb combinations. For example, to exclude pulling while spinning, pushing while zigzagging, and walking hesitantly, you could use `--adverb_verb_filter="not, pull while spinning, push while zigzagging, walk hesitantly"`. 
To include _only_ pulling while spinning, pushing while zigzagging, and walking hesitantly, you could use `--adverb_verb_filter="pull while spinning, push while zigzagging, walk hesitantly"`.
* `--k` specifies the number of examples to include where the adverb "cautiously" appears. By default this is set to `1.
* `--device` specifies which device PyTorch should use. By default is set to `cpu`.

## Training
Once you have your preprocessed data in ```data_processed```, you can start training using either `train_single.py` or `train_combined.py`.
The first trains a single model while the second trains a population of models with evolved selective attention matrices.
Both 

The training scripts have the following optional arguments:
* `--data_folder` specifies where the preprocessed data is located. By default set to `./data_processed`.
* `--train_split` specifies which split to train on, e.g. `train`or `train_subset_0.02`.
* ``--dev_split`` specifies what to use as the dev set. By default set to `dev`.
* `--save_directory` specifies where to save the trained models. By default set to `../trained_models`.
* `--device` specifies which device PyTorch should use. By default is set to `cpu` but can be set to `cuda` if available.
* `--epochs` specifies the number of epochs to train for. The default is `100`.
* `--runs` specifies the number of runs. The default is `10`.
* ```--batch_size``` specifies the batch size to use. The default is ``4096``.
* ```--perfect_attention``` specifies whether to assume perfect selective attention or use trained selective attention. The default is `True` for `train_single.py`and `False` for `train_combined.py`.
* `--selective_attention` specifies whether to use selective attention. By default set to `True`.
* ``--log_dir`` specifies where to save TensorBoard logs. By default set to `../runs`.

``train_single.py`` has the additional option
* ``--pretrained_model`` where you can specify the path to a model you would like to continue training.

``train_combined.py`` has the additional options
* ```--popsize``` specifies the size of the CMA-ES population for evolving the selective attention matrix. By default is set to `8`.
* ```--indirect_feedback``` specifies whether to disable auxiliary feedback for the optimization of the selective attention matrix. By default is set to `False` (i.e., auxiliary feedback is enabled).
* ```--weight_decay``` specifies the weight decay to use for training. By default is set to `0.0001`.
* ```--action_attention``` specifies whether to use the action attention mechanism. By default is set to `True`.
* ```--sub_pos``` specifies whether to use relative target distances instead of absolute coordinates. By default set to `False`.

The ``trained_models`` folder contains the unablated models from the experiments in the paper, including the Echo State Network used to produce the command embeddings.  Ablated models and the other variations that were tested are available upon request.

## Evaluating
To evaluate a trained model, run `evaluate.py`. This script takes the following optional arguments:
* ````--model_dir_path```` specifies the folder that the models which are to be evaluated are in. By default set to ``../trained_models/direct_attention/direct_attention_1.0``
* ```--data_folder``` specifies the location of the preprocessed data. By default set to `../data_processed`.
* ```--test_split``` specifies which split to test the model on. This could be `test`, `dev`, `visual`, `situational_1`, `situational_2`, `contextual`, `adverb_1`, `adverb_2, `visual_easier`, or whatever name you saved your preprocessed datasets under.
* ```--device``` specifies which device PyTorch should use. By default is set to `cpu`.
* ```--perfect_attention``` specifies whether to assume perfect selective attention for evaluation. By default set to `False`. If you trained a model with perfect attention, remember to enable this for evaluation as well, otherwise the model will use its untrained random attention matrix and probably not do great.
* ```--selective_attention``` specifies whether to use selective attention during evaluation. By default set to `True`. If you trained a model without selective attention, remember to disable this for evaluation as well.
* ```--sub_pos``` specifies whether to use relative target distances instead of absolute coordinates. By default set to `False`. If you trained a model with relative target distances, remember to enable this for evaluation as well.
* ````--action_attention```` specifies whether to use the action attention mechanism during evaluation.

# Acknowledgments
This code builds in parts on other repositories, namely:

* Laura Ruis' [gSCAN](https://github.com/LauraRuis/groundedSCAN) repository (MIT License), for data generation and loading
* Vadim Alperovich's [TextGenESN](https://github.com/VirtualRoyalty/TextGenESN) repository (MIT License), for the Echo State Network implementation