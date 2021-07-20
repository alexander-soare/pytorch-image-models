import sys
import gc
from multiprocessing import Manager, Pool, Process
from typing import Sequence
import argparse

import pandas as pd
import torch
from torch import fx
from timm.models.fx_helpers import fx_and, fx_float_to_int
from timm.models.fx_features import NodePathTracer
import timm
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='store_true', default=False)
parser.add_argument('-s', '--start-from', default=0, type=int)
parser.add_argument('-g', '--gpu', action='store_true', default=False)
args = parser.parse_args()

deprecated = ['vit_base_resnet50_384']
EXCLUDE_JIT_FILTERS = ['*iabn*', 'tresnet*', 'dla*', 'hrnet*', 'ghostnet*', 'vit_large_*', 'vit_huge_*']

pd.set_option('display.max_colwidth', 100)
pd.set_option("display.max_rows", 1000)

manager = Manager()
results = manager.dict()


def check_model(model_str):
    if model_str in deprecated:
        return
    model = timm.create_model(model_str)
    tracer = NodePathTracer(autowrap_functions=(fx_and, fx_float_to_int))

    if args.verbose:
        print(model_str)

    # TRACE
    try:
        graph = tracer.trace(model)
    except (TypeError, fx.proxy.TraceError) as e:
        torch.fx.proxy.TraceError()
        results[model_str] = {
            'at': 'trace',
            'status': 'fail',
            'error_class': e.__class__.__name__,
            'error_msg': (' '.join(str(e).split(' ')[:12]) + '...').strip()}
        if args.verbose:
            print(results[model_str])
        return

    # FORWARD
    input_size = timm.get_model_default_value(model_str, 'input_size') or (3, 224, 224)
    device = torch.device('cuda') if (input_size[-1] < 448 and args.gpu) else torch.device('cpu')
    inp = torch.zeros(1, *input_size).to(device)
    graph_module = fx.GraphModule(model, graph).to(device)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    for _ in range(2):
        try:
            out = graph_module(inp)
        except Exception as e:
            results[model_str] = {
                'at': 'forward',
                'status': 'fail',
                'error_class': e.__class__.__name__,
                'error_msg': (' '.join(str(e).split(' ')[:12]) + '...').strip()}
            if results[model_str]['error_msg'].startswith('Expected more than 1 value per channel'):
                inp = torch.zeros(2, *input_size).to(device)
                continue
            if args.verbose:
                print(results[model_str])
            return

    if isinstance(out, Sequence):
        out = out[0]
    out = out.mean()

    # BACKWARD
    try:
        out.backward()
    except Exception as e:
        results[model_str] = {
            'at': 'backward',
            'status': 'fail',
            'error_class': e.__class__.__name__,
            'error_msg': (' '.join(str(e).split(' ')[:12]) + '...').strip()}
        if args.verbose:
            print(results[model_str])
        return

    if model_str in timm.list_models(filter=EXCLUDE_JIT_FILTERS):
        results[model_str] = {'status': 'pass'}
        return

    # JIT SCRIPT
    try:
        jit_graph_module = torch.jit.script(graph_module).to(device)
        del graph_module
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        results[model_str] = {
            'at': 'jit.script',
            'status': 'fail',
            'error_class': e.__class__.__name__,
            'error_msg': (' '.join(str(e).split(' ')[:12]) + '...').strip()}
        if args.verbose:
            print(results[model_str])
        return

    # JIT FORWARD
    try:
        out = jit_graph_module(inp)
    except Exception as e:
        results[model_str] = {
            'at': 'jit forward',
            'status': 'fail',
            'error_class': e.__class__.__name__,
            'error_msg': (' '.join(str(e).split(' ')[:12]) + '...').strip()}
        if args.verbose:
            print(results[model_str])
        return

    if isinstance(out, Sequence):
        out = out[0]
    out = out.mean()

    # JIT BACKWARD
    try:
        out.backward()
    except Exception as e:
        results[model_str] = {
            'at': 'jit backward',
            'status': 'fail',
            'error_class': e.__class__.__name__,
            'error_msg': (' '.join(str(e).split(' ')[:12]) + '...').strip()}
        if args.verbose:
            print(results[model_str])
        return

    results[model_str] = {'status': 'pass'}


model_strs = timm.list_models()[args.start_from:]

# with Pool(4) as pool:
#     list(tqdm(pool.imap_unordered(check_model, model_strs, chunksize=4), total=len(model_strs)))

for model_str in tqdm(model_strs):
    process = Process(target=check_model, args=(model_str,))
    process.start()
    process.join()
    torch.cuda.empty_cache()


sys.stdout = open('fx_coverage_results_2.txt', 'w')

print()
print("Models that passed:")
print(passed := [k for k, v in results.items() if v['status'] == 'pass'])

print()
print("Models that failed:")
print([k for k, v in results.items() if v['status'] == 'fail'])

print()
print("Fail reasons")
fail_reasons = [
    [k, f"({v['at']}) {v['error_class']}: {v['error_msg']}"] for k, v in results.items() if v['status'] == 'fail']
df = pd.DataFrame(fail_reasons, columns=['Model', 'Reason'])
print(df)

print()
print("Fail reasons summarised")
print(df.groupby("Reason").count())

print()
print(f"Proportion of models that passed: {100 * len(passed)/len(results):.2f} %")

sys.stdout.close()