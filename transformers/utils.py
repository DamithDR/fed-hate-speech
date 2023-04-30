import numpy as np


def iid_partition(data, clients):
    r""" Creates iid partitions of data over clients.
    """
    num_items_per_client = int(len(data) / clients)
    client_dict = {}
    data_idxs = list(range(len(data)))

    for i in range(clients):
        client_dict[i] = set(np.random.choice(data_idxs, num_items_per_client, replace=False))
        data_idxs = list(set(data_idxs) - client_dict[i])

    return client_dict


def iid_partition_for_federated_offence(data, clients):
    r""" Creates iid partitions of data over clients.
        """
    client_dict = {}

    for i in range(clients):
        data_idxs = list(range(len(data[i]['train'])))
        client_dict[i] = set(data_idxs)

    return client_dict


def get_model_ckpt(model_type):
    if model_type == "distilbert":
        return "distilbert-base-uncased"
    elif model_type == "bert":
        return "bert-base-uncased"
    elif model_type == "distilroberta":
        return "distilroberta-base"
    elif model_type == "roberta":
        return "roberta-base"
    elif model_type == "fnet":
        return "google/fnet-base"
    elif model_type == "fused_model":
        return "../../FederatedOffence/model_davidson_model_olid/"
    else:
        raise NotImplementedError
