from typing import Union
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal # for Python3.6/3.7 users
import os
import torch
import numpy as np
from .pydssp_numpy import (
    get_hbond_map as get_hbond_map_numpy,
    assign as assign_numpy
)
from .pydssp_torch import (
    get_hbond_map as get_hbond_map_torch,
    assign as assign_torch
)

from .util import parse_pdb

C3_ALPHABET = np.array(['-', 'H', 'E', 'h']) # adding `h` as left-handed helix https://en.wikipedia.org/wiki/Protein_secondary_structure
# equivalent onehot encode would be [loop, helix, strand, left_handed_helix] = [0,1,2,3]


def get_hbond_map(
    coord: Union[torch.Tensor, np.ndarray],
    return_e: bool=False
    ) -> Union[torch.Tensor, np.ndarray]:
    assert type(coord) in [torch.Tensor, np.ndarray], 'Input type must be torch.Tensor or np.ndarray'
    if type(coord) == torch.Tensor:
        return get_hbond_map_torch(coord, return_e=return_e)
    elif type(coord) == np.ndarray:
        return get_hbond_map_numpy(coord, return_e=return_e)


def assign(
    coord: Union[torch.Tensor, np.ndarray],
    out_type: Literal['onehot', 'index', 'c3'] = 'c3'
    ) -> np.ndarray:
    assert type(coord) in [torch.Tensor, np.ndarray], "Input type must be torch.Tensor or np.ndarray"
    assert out_type.lower() in ['onehot', 'index', 'c3'], "Output type must be 'onehot', 'index', or 'c3'"
    out_type = out_type.lower()
    # main calcuration
    if type(coord) == torch.Tensor:
        onehot = assign_torch(coord)
    elif type(coord) == np.ndarray:
        onehot = assign_numpy(coord)
    # output one-hot
    if out_type == 'onehot':
        return onehot
    # output index
    index = torch.argmax(onehot.to(torch.float), dim=-1) if type(onehot) == torch.Tensor else np.argmax(onehot, axis=-1)
    if out_type == 'index':
        return index
    # output c3
    c3 = C3_ALPHABET[index.cpu().numpy()] if type(index) == torch.Tensor else C3_ALPHABET[index]
    return c3


def assign_pdb(
    pdb,
    out_type: Literal['onehot', 'index', 'c3'] = 'c3'
    ) -> np.ndarray:
    assert type(pdb) == str, "Input type must be str"
    assert os.path.exists(pdb), "Input file does not exist"
    #assert type(coord) in [torch.Tensor, np.ndarray], "Input type must be torch.Tensor or np.ndarray"
    out_type = out_type.lower() # avoid typos
    assert out_type in ['onehot', 'index', 'c3'], "Output type must be 'onehot', 'index', or 'c3'"

    parse_pdb_out = parse_pdb(pdb)
    pdb_xyz = parse_pdb_out['xyz']
    coord = pdb_xyz[:,:4] # backbone atoms - N, CA, C, CB
    # main calcuration
    if type(coord) == torch.Tensor:
        onehot = assign_torch(coord)
    elif type(coord) == np.ndarray:
        onehot = assign_numpy(coord)
    # output one-hot
    if out_type == 'onehot':
        return onehot
    # output index
    index = torch.argmax(onehot.to(torch.float), dim=-1) if type(onehot) == torch.Tensor else np.argmax(onehot, axis=-1)
    if out_type == 'index':
        return index
    # output c3
    c3 = C3_ALPHABET[index.cpu().numpy()] if type(index) == torch.Tensor else C3_ALPHABET[index]
    return c3
