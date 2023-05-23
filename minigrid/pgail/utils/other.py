import random
import numpy
import torch
import collections


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d

def clip_grad_value(parameters, clip_value: float) -> None:
    r"""Clips gradient of an iterable of parameters at specified value.

    Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clip_value (float or int): maximum allowed value of the gradients.
            The gradients are clipped in the range
            :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
    """
    #if isinstance(parameters, torch.Tensor):
    #    parameters = [parameters]
    clip_value = float(clip_value)
    for p in parameters:
        if p.grad is None:
            continue
        #if not torch.isfinite(p.grad).all():
            #print("Clipping infinite grad")
        p.grad = p.grad.nan_to_num(nan = 0.0, posinf = 1.0, neginf = -1.0)
        if not torch.isfinite(p.grad).all():
            print("Clipping infinite grad Failed???!!!")