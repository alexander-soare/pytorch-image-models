""" PyTorch FX Based Feature Extraction Helpers

An extension/alternative to timm.models.features making use of PyTorch FX. Here, the idea is to:
    1. Symbolically trace a model producing a graph based intermediate representation (PyTorch FX functionality with
        some custom tweaks)
    2. Identify desired feature extraction nodes and reconfigure them as output nodes while deleting all unecessary
        nodes. (custom - inspired by https://github.com/pytorch/vision/pull/3597)
    3. Write the resulting graph into a GraphModule (PyTorch FX functionality)

Copyright 2021 Alexander Soare
"""
from typing import Callable, Dict
import math
from collections import OrderedDict
from pprint import pprint
from inspect import ismethod

import torch
from torch import nn
from torch import fx
import torch.nn.functional as F

from .fx_helpers import fx_and, fx_float_to_int
from .features import _get_feature_info

# # Layers we went to treat as leaf modules for FeatureGraphNet
from .layers import Conv2dSame, ScaledStdConv2dSame, BatchNormAct2d, BlurPool2d, CondConv2d, StdConv2dSame
from .layers import GatherExcite, DropPath
from .layers.non_local_attn import BilinearAttnTransform
from .layers.pool2d_same import MaxPool2dSame, AvgPool2dSame
# from .xcit import PositionalEncodingFourier  # TODO do I want to make xcit responsible for this?


# These modules will not be traced through. From a code design perspective, we put anything under .layers here manually
_leaf_modules = {
    Conv2dSame, ScaledStdConv2dSame, BatchNormAct2d, BlurPool2d, CondConv2d, StdConv2dSame, GatherExcite, DropPath,
    BilinearAttnTransform, MaxPool2dSame, AvgPool2dSame,# PositionalEncodingFourier,
}

try:
    from .layers import InplaceAbn
    _leaf_modules.add(InplaceAbn)
except ImportError:
    pass


def register_leaf_module(module: nn.Module):
    """
    Any module not under .layers (at the time of writing, that would be modules defined in timm.models) should get this
    decorator.
    """
    _leaf_modules.add(module)
    return module


class TimmTracer(fx.Tracer):
    """
    Temporary bridge from torch.fx.Tracer to include any general workarounds required to make FX work for us
    """
    def __init__(self, autowrap_modules=(math, ), autowrap_functions=(), enable_cpatching=False):
        super().__init__(autowrap_modules=autowrap_modules, enable_cpatching=enable_cpatching)
        # FIXME: This is a workaround for a PyTorch feature request https://github.com/pytorch/pytorch/issues/62021
        self._autowrap_function_ids.update(set([id(f) for f in autowrap_functions]))

    def create_node(self, kind, target, args, kwargs, name=None, type_expr=None):
        # FIXME: This is a workaround for a PyTorch issue https://github.com/pytorch/pytorch/issues/61970
        if target == F.pad:
            kwargs['value'] = float(kwargs['value'])
        return super().create_node(kind, target, args, kwargs, name=name, type_expr=type_expr)


class LeafNodeTracer(TimmTracer):
    """
    Account for desired leaf nodes.
    """
    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, tuple(_leaf_modules)):
            return True
        return super().is_leaf_module(m, module_qualified_name)


# taken from https://github.com/pytorch/examples/blob/master/fx/module_tracer.py
# with modifications for 
class NodePathTracer(LeafNodeTracer):
    """
    NodePathTracer is an FX tracer that, for each operation, also records the qualified name of the Node from which the
    operation originated. A qualified name here is a `.` seperated path walking the hierarchy from top level module
    down to leaf operation or leaf module.
    """
    # TODO should these really be class attributes?
    # The current qualified name of the Module being traced. The top-level
    # module is signified by empty string. This is updated when entering
    # call_module and restored when exiting call_module
    current_module_qualified_name : str = ''
    # A map from FX Node to the qualname of the Module from which it
    # originated. This is recorded by `create_proxy` when recording an
    # operation
    node_to_originating_module = OrderedDict()

    def call_module(self, m: torch.nn.Module, forward: Callable, args, kwargs):
        """
        Override of Tracer.call_module (see
        https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer.call_module).
        This override:
        1) Stores away the qualified name of the caller for restoration later
        2) Installs the qualified name of the caller in `current_module_qualified_name`
           for retrieval by `create_proxy`
        3) Delegates into the normal Tracer.call_module method
        4) Restores the caller's qualified name into current_module_qualified_name
        """
        old_qualname = self.current_module_qualified_name
        try:
            module_qualified_name = self.path_of_module(m)
            self.current_module_qualified_name = module_qualified_name
            if not self.is_leaf_module(m, module_qualified_name):
                out = forward(*args, **kwargs)
                return out
            return self.create_proxy('call_module', module_qualified_name, args, kwargs)
        finally:
            self.current_module_qualified_name = old_qualname

    def create_proxy(self, kind: str, target: fx.node.Target, args, kwargs, name=None, type_expr=None):
        """
        Override of `Tracer.create_proxy`. This override intercepts the recording
        of every operation and stores away the current traced module's qualified
        name in `node_to_originating_module`
        """
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr)
        self.node_to_originating_module[proxy.node] = self._get_node_qualified_name(
            self.current_module_qualified_name, proxy.node)
        return proxy

    def _get_node_qualified_name(self, module_qualified_name: str, node: fx.node.Node):
        if node.op == 'call_module':
            # Node terminates in a leaf module so the module_qualified_name is a complete description of the node
            return module_qualified_name
        else:
            # Node terminates in non- leaf module so the node name needs to be appended
            return module_qualified_name + '.' + str(node)


def print_graph_node_qualified_names(model: nn.Module):
    """
    Dev utility to prints nodes in order of execution. Useful for choosing `return_nodes` for a FeatureGraphNet design.
    This is useful for two reasons:
        1. Not all submodules are traced through. Some are treated as leaf modules. See `LeafNodeTracer`
        2. Leaf ops that occur more than once in the graph get a `_{counter}` postfix.
        
    WARNING: Changes to the operations in the original module might not change the module's overall behaviour, but they
    may result in changes to the postfixes for the names repeated ops, thereby breaking feature extraction.
    """
    tracer = NodePathTracer(autowrap_functions=(fx_and, fx_float_to_int))
    tracer.trace(model)
    pprint(list(tracer.node_to_originating_module.values()))


def get_intermediate_nodes(model: nn.Module, return_nodes: Dict[str, str]) -> nn.Module:
    """
    Creates a new FX-based module that returns intermediate nodes from a given model. This is achieved by re-writing
    the computation graph of the model via FX to return the nodes layers. All unused nodes are removed, together
    with their corresponding parameters.
    Args:
        model (nn.Module): model on which we will extract the features
        return_nodes (Dict[name, new_name]): a dict containing the names (or partial names - see note below) of the
            nodes for which the activations will be returned as the key of the dict. The value of the dict is the name
            of the returned activation (which the user can specify).
            A note on node specification: A node is specified as a `.` seperated path walking the hierarchy from top
            level module down to leaf operation or leaf module. For instance `blocks.5.3.bn1`. Nevertheless, the keys
            in this dict need not be fully specified. One could provide `blocks.5` as a key, and the last node with
            that prefix will be selected.
            While designing a feature extractor one can use the `print_graph_node_qualified_names` utility as a guide
            to which nodes are available.

    TODO what is the node qualname when we reuse the same module more than once?
    TODO check the assumption that we got the nodes in order.

    Acknowledgement: Starter code from https://github.com/pytorch/vision/pull/3597
    """
    # TODO have duplicate nodes but full coverage for module names
    return_nodes = {str(k): str(v) for k, v in return_nodes.items()}

    # Instantiate our NodePathTracer and use that to trace the model
    tracer = NodePathTracer()
    graph = tracer.trace(model)

    name = model.__class__.__name__ if isinstance(model, nn.Module) else model.__name__
    m = fx.GraphModule(tracer.root, graph, name)
    
    available_nodes = [f'{v}.{k}' for k, v in tracer.node_to_originating_module.items()]
    # FIXME We don't know if we should expect this to happen
    assert len(set(available_nodes)) == len(available_nodes), \
        "There are duplicate nodes! Please raise an issue https://github.com/rwightman/pytorch-image-models/issues"
    # Check that all outputs in return_nodes are present in the model
    for query in return_nodes.keys():
        if not any([m.startswith(query) for m in available_nodes]):
            raise ValueError(f"return_node: {query} is not present in model")

    # Remove existing output nodes
    # TODO deal with multiple outputs
    orig_output_node = None
    for n in reversed(m.graph.nodes):
        if n.op == "output":
            orig_output_node = n
            break
    assert orig_output_node
    # And remove it
    m.graph.erase_node(orig_output_node)

    # Find nodes corresponding to return_nodes and make them into output_nodes
    nodes = [n for n in m.graph.nodes]
    output_nodes = OrderedDict()
    for n in reversed(nodes):
        module_qualname = tracer.node_to_originating_module.get(n)
        for query in return_nodes:
            if module_qualname.startswith(query):
                output_nodes[return_nodes[query]] = n
                return_nodes.pop(query)
                break
    output_nodes = OrderedDict(reversed(list(output_nodes.items())))

    # And add them in the end of the graph
    with m.graph.inserting_after(nodes[-1]):
        m.graph.output(output_nodes)

    m.graph.eliminate_dead_code()
    m.recompile()

    # Remove unused modules / parameters
    m = fx.GraphModule(m, m.graph, name)
    return m


class FeatureGraphNet(nn.Module):
    """
    Take the provided model and transform it into a graph module. This class wraps the resulting graph module while
    also keeping the original model's non-parameter properties for reference. The original model is discarded.

    WARNING: Changes to the operations in the original module might not change the module's overall behaviour, but they
    may result in changes to the postfixes for the names repeated ops, thereby breaking feature extraction.
    """
    def __init__(self, model, out_indices, out_map=None):
        super().__init__()
        self.feature_info = _get_feature_info(model, out_indices)
        # NOTE the feature_info key is innapropriately named 'module' because prior to FX only modules could be
        # provided. Recall that here, we may also provide nodes referring to individual ops
        if out_map is not None:
            assert len(out_map) == len(out_indices)
        return_nodes = {info['module']: out_map[i] if out_map is not None else info['module']
                        for i, info in enumerate(self.feature_info) if i in out_indices}
        self.graph_module = get_intermediate_nodes(model, return_nodes)
        # Keep non-parameter model properties for reference
        for attr_str in model.__dir__():
            attr = getattr(model, attr_str)
            if (not attr_str.startswith('_') and attr_str not in self.__dir__() and not ismethod(attr)
                    and not isinstance(attr, (nn.Module, nn.Parameter))):
                setattr(self, attr_str, attr)

    def forward(self, x):
        return self.graph_module(x)