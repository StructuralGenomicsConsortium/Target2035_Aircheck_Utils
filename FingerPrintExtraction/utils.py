import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdchem import Mol


def is_mol(obj):
    return isinstance(obj, Mol)


def to_mol(smi):
    if isinstance(smi, Mol):
        return smi
    if isinstance(smi, str):
        return MolFromSmiles(smi)


def catch_boost_argument_error(e) -> bool:
    """
    This ugly code is to try and catch the Boost.Python.ArgumentError that rdkit throws when you pass an argument of
    the wrong type into the function
    Parameters
    ----------
    e : an Exception
        the Exception raised by the code

    Returns
    -------
    bool
        True if it is the Boost.Python.ArgumentError, False if it is not
    """
    if str(e).startswith("Python argument types"):
        return True
    else:
        return False


def to_list(obj):
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, str):
        return [obj]
    elif not hasattr(obj, "__iter__"):
        return [obj]
    else:
        return list(obj)
