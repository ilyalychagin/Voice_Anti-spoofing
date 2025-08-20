import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(data):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    flacs = []
    labels = []    

    for el in data:
        flacs.append(el['data_object'])
        labels.append(el['label'])

    flacs = pad_sequence(flacs, batch_first=True)    
    labels = torch.Tensor(labels).long()

    result_batch = {
        "data_object": flacs,
        "labels": labels
    }
    return result_batch
