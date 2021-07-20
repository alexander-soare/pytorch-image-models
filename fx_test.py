from typing import Sequence

import torch
from torch import fx
from timm.models.fx_helpers import fx_and, fx_float_to_int
from timm.models.fx_features import NodePathTracer

import timm

MODEL = 'efficientnetv2_l'
model = timm.create_model(MODEL)

tracer = NodePathTracer(autowrap_functions=(fx_and, fx_float_to_int))
graph = tracer.trace(model)
graph_module = fx.GraphModule(model, graph)
# print(graph_module.code)
print("TRACING SUCCESSFUL")


input_size = timm.get_model_default_value(MODEL, 'input_size') or (3, 224, 224)
inp = torch.zeros(1, *input_size)
out = model(inp)
if isinstance(out, Sequence):
    out = out[0]
out = out.mean()
out.backward()

print("FORWARD/BACKWARD SUCCESSFUL")


jit_graph_module = torch.jit.script(graph_module)
# out = model(inp)
# if isinstance(out, Sequence):
#     out = out[0]
# out = out.mean()
# out.backward()

print("JIT FORWARD/BACKWARD SUCCESSFUL")