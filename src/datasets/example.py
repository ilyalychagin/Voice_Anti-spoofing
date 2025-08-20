import numpy as np
import torch

from pathlib import Path
from typing import Callable, Optional

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json

class VoiceSpoofDataset(BaseDataset):
    """
    Dataset for voice spoofing detection tasks.
    
    Contains audio files with corresponding labels indicating whether
    the voice is bonafide or spoofed.
    """

    def __init__(
        self,
        path2dir: str = None,
        path2dscr: str = None,
        transform: Optional[Callable] = None,
        part: "str" = "train",
        *args, **kwargs
    ):        
        """
        Args:
            transform (Optional[Callable]): transformation to apply to audio data
            path2dir (str): path to directory containing audio files
            path2dscr (str): path to descriptor file with metadata
            part (str): partition name (train, val, test)
        """
        self.transform = transform
        self.path2dir = path2dir
        self.path2dscr = path2dscr

        index = self.create_or_load_index(part) 

        super().__init__(index, *args, **kwargs)

    def create_or_load_index(self, part):
        """
        Load existing index file or create new one if it doesn't exist.
        
        Args:
            part (str): partition name
            
        Returns:
            index (list[dict]): list of dictionaries with file metadata
        """
        index_path = ROOT_PATH / "data" / part / "index.json"
        if not index_path.exists():
            self.create_index(part)
            
        return read_json(index_path)

    def create_index(self, part):
        """
        Create index from descriptor file by parsing metadata and labels.
        
        Args:
            part (str): partition name
        """
        path2dir = ROOT_PATH / self.path2dir
        path2dscr = ROOT_PATH / self.path2dscr

        file = []
        with path2dscr.open("rt") as handle:
            file = handle.readlines()
    
        index = [{"path": (path2dir / (line.split(' ')[1] + '.flac')).as_posix(),
                  "label": line.split(' ')[4] == "spoof\n"} 
                 for line in file]

        torch.manual_seed(0)
        indexes = torch.randperm(len(index))
        train_indexes = indexes[:int(len(index))]

        train_index = [index[i] for i in train_indexes]

        index_path = ROOT_PATH / "data" / part / "index.json"
        write_json(train_index, str(index_path))
