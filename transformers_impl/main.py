import argparse
import json
import os
import sys
import time
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    set_seed,
)

from transformers_impl.trainer import *
from transformers_impl.trainer_extended import FedTrainerExtended
from transformers_impl.utils import *

# command-line arguments
parser = argparse.ArgumentParser("hate speech classification using federated learning")
parser.add_argument("--data", type=str, default="data", help="location of the data corpus")
parser.add_argument(
    "--dataset_type",
    type=str,
    default="federated_offence",
    choices=["comb", "vidgen_binary", "vidgen_multiclass", "federated_offence"],
    help="which dataset to run the experiment with",
)
parser.add_argument(
    "--batch_size", type=int, default=64, help="batch size for fine-tuning the model",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="random seed for deterministic behaviour and reproducibility",
)
parser.add_argument(
    "--model_type",
    type=str,
    default="distilbert",
    choices=["fnet", "distilbert", "bert", "distilroberta", "roberta", "fused_model"],
    help="specify which model to use (fnet, distilbert, bert, distilroberta, roberta,fused_model)",
)
parser.add_argument("--rounds", type=int, default=1000, help="number of training rounds")
parser.add_argument("--C", type=float, default=0.1, help="client fraction")
parser.add_argument("--K", type=int, default=100, help="number of clients for iid partition")
parser.add_argument(
    "--E",
    "--epochs",
    type=int,
    default=1,
    help="number of training epochs on local dataset for each round",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="fedprox",
    choices=["fedavg", "fedprox", "fedopt"],
    help="specify which algorithm to use during local updates aggregation (fedopt, fedprox, fedavg)",
)
parser.add_argument("--mu", type=float, default=0.01, help="proximal term constant")
parser.add_argument("--client_lr", type=float, default=2e-5, help="learning rate for client")
parser.add_argument("--server_lr", type=float, default=0.0, help="learning rate for server")
parser.add_argument(
    "--class_weights",
    action="store_true",
    default=False,
    help="determine if experiments should use class weights in loss function or not",
)
parser.add_argument("--es_patience", type=int, default=10, help="early stopping patience level")
parser.add_argument("--save", type=str, default="exp", help="experiment name")
parser.add_argument("--datasets", type=str, required=True, help="comma seperated names of the datasets")
parser.add_argument("--cuda_device", type=int, default=0, required=False, help="cuda device number")
args = parser.parse_args()

# proximal term is 0.0 in case of fedavg
if args.algorithm == "fedavg":
    args.mu = 0.0

# we don't need server optimizer in case of fedprox and fedavg + centralized training
if args.algorithm != "fedopt" or args.K == 1:
    args.server_lr = None
args.algorithm = None if args.K == 1 else args.algorithm

args.save = "{}-{}".format(args.save, time.strftime("%Y%m%d-%H%M%S"))

if not os.path.exists(args.save):
    os.mkdir(args.save)
print("Experiment dir: {}".format(args.save))

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

warnings.filterwarnings("ignore")


def main():
    if not torch.cuda.is_available():
        logging.info("No GPU device available")
        sys.exit(1)

    # log all the arguments
    logging.info(args)

    # reproducibility set seed
    set_seed(args.seed)
    datasets = args.datasets.split(',')
    print(f'datasets = {datasets}')
    if len(datasets) != int(args.K):
        logging.error("datasets size does not match with no of clients")
        sys.exit(1)

    federated_experiments = False
    raw_data = []
    if 'olid' in datasets or 'davidson' in datasets or 'hasoc' in datasets or 'hatexplain' in datasets:
        print('federated experiments activated')
        federated_experiments = True
        for d_set in datasets:
            train = pd.read_csv(f'../FederatedOffense/ft_{d_set}.csv',
                                sep='\t')  # using finetune set for client training
            train, valid = train_test_split(train, test_size=0.1, random_state=777)
            test = pd.read_csv(f'../FederatedOffense/data/{d_set}/{d_set}_test.csv', sep='\t')
            raw_data.append(
                {
                    "train": train,
                    "valid": valid,
                    "test": test
                }
            )
    else:
        raw_data = {
            "train": pd.read_csv(os.path.join(args.data, "train.csv")),
            "valid": pd.read_csv(os.path.join(args.data, "valid.csv")),
            "test": pd.read_csv(os.path.join(args.data, "test.csv")),
        }

    # ignore abusive category due to insufficient examples
    if not federated_experiments:
        for split in raw_data:
            raw_data[split] = raw_data[split][raw_data[split]["category"] != "abusive"]
            raw_data[split].reset_index(inplace=True, drop=True)

    # load the tokenizer from huggingface hub
    model_ckpt = get_model_ckpt(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    categories = get_categories(args.dataset_type)
    new_categories = [cat for cat in categories if cat != "abusive"]
    category2id = {cat: idx for idx, cat in enumerate(new_categories)}
    id2category = {idx: cat for idx, cat in enumerate(new_categories)}
    args.class_names = new_categories

    print(category2id)
    print(id2category)

    if federated_experiments:
        dataset = []
        for data_element in raw_data:
            dataset.append({
                "train": create_data_iter(data_element["train"], category2id, tokenizer, input_col='text',
                                          target_col='labels'),
                "valid": create_data_iter(data_element["valid"], category2id, tokenizer, input_col='text',
                                          target_col='labels'),
                "test": create_data_iter(data_element["test"], category2id, tokenizer, input_col='text',
                                         target_col='labels'),
            })
    else:
        dataset = {
            "train": create_data_iter(raw_data["train"], category2id, tokenizer),
            "valid": create_data_iter(raw_data["valid"], category2id, tokenizer),
            "test": create_data_iter(raw_data["test"], category2id, tokenizer),
        }

    # categorical classes for hate speech data
    if federated_experiments:
        classes = list(
            np.unique([elem["labels"].item() for elem in dataset[0]["train"]]))  # getting labels from one dataset
        class_array = np.array([elem["labels"].item() for elem in dataset[0]["train"]])
    else:
        classes = list(np.unique([elem["labels"].item() for elem in dataset["train"]]))
        class_array = np.array([elem["labels"].item() for elem in dataset["train"]])
    num_classes = len(classes)
    args.classes = classes
    args.num_classes = num_classes

    if federated_experiments:
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=classes, y=classes
        )
    else:
        if args.class_weights:
            class_weights = class_weight.compute_class_weight(
                class_weight="balanced", classes=classes, y=class_array
            )
        else:
            class_weights = class_weight.compute_class_weight(
                class_weight="balanced", classes=classes, y=classes
            )
    args.class_weights_array = class_weights
    logging.info(f"# of classes: {num_classes}")
    if federated_experiments:
        logging.info(f"class weights in the first dataset: {class_weights}")
    else:
        logging.info(f"class weights: {class_weights}")
    # load the model from huggingface hub
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt, num_labels=num_classes, label2id=category2id, id2label=id2category,
    )
    # setting devide number
    torch.cuda.set_device(args.cuda_device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        model.cuda()

    if federated_experiments:
        iid_data_dict = iid_partition_for_federated_offence(dataset, args.K)  # pass number of clients
    else:
        # dict mapping clients to the data samples in iid fashion
        iid_data_dict = iid_partition(dataset["train"], args.K)

    # log the config for each run
    config_dict = dict(
        rounds=args.rounds,
        C=args.C,
        K=args.K,
        E=args.E,
        model=args.model_type,
        algorithm=args.algorithm,
        mu=args.mu,
        client_lr=args.client_lr,
        server_lr=args.server_lr,
        batch_size=args.batch_size,
        class_weights=args.class_weights,
    )
    with open(os.path.join(args.save, "config.json"), "w") as fp:
        json.dump(config_dict, fp=fp, indent=2)

    fl_trainer = FedTrainerExtended(
        args,
        tokenizer,
        model,
        local_data_idxs=iid_data_dict,
        dataset=dataset
    )
    model = fl_trainer.train()


if __name__ == "__main__":
    main()
