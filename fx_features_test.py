import torch
import timm

model = timm.create_model('resnet50', features_only=True, feature_cfg={'feature_cls': 'fx'}, out_indices=(0,))
print(model.graph_module)
# model.feature_info = FeatureInfo(model.feature_info, out_indices=(0, 1, 2))
# return_nodes = {info['module']: info['module'] for info in model.feature_info.info}
# print_graph_node_qualified_names(model)
# model = FeatureGraphNet(model)
torch.jit.script(model)
# # print(model.feature_info.info)
# # # # tracer = NodePathTracer()
# # # # graph = tracer.trace(model)
# # # sys.stdout = open('fx_graph.txt', 'w')
# # # model.graph.print_tabular()
# # # # for node in model.graph.nodes:
# # # #     module_qualname = tracer.node_to_originating_module.get(node)
# # # #     print('Node', node, 'is from module', module_qualname)
# # # # print(model)
inp = torch.zeros(2, 3, 224, 224)
with torch.no_grad():
    out = model(inp)
for k, o in out.items():
    print(k, o.shape)