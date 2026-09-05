# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Inspect ONNX operators, tensor shapes, parameters, and activation-node consumers."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict

import onnx


def tensor_shape(value) -> list[int | str]:
    """Return an ONNX value shape with symbolic dimensions preserved."""
    return [d.dim_value or d.dim_param or "?" for d in value.type.tensor_type.shape.dim]


def inspect(path: str) -> dict:
    """Return graph structure and activation audit details for an ONNX model."""
    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    consumers = defaultdict(list)
    for node in graph.node:
        for name in node.input:
            consumers[name].append(node.name or node.op_type)
    activations = []
    for node in graph.node:
        if node.op_type in {"Sigmoid", "Softmax"}:
            activations.append(
                {
                    "op_type": node.op_type,
                    "name": node.name,
                    "inputs": list(node.input),
                    "outputs": list(node.output),
                    "consumers": [consumer for output in node.output for consumer in consumers[output]],
                }
            )
    initializers = {x.name for x in graph.initializer}
    return {
        "inputs": {x.name: tensor_shape(x) for x in graph.input if x.name not in initializers},
        "outputs": {x.name: tensor_shape(x) for x in graph.output},
        "operators": dict(sorted(Counter(x.op_type for x in graph.node).items())),
        "activations": activations,
        "initializer_parameters": sum(math.prod(x.dims) for x in graph.initializer),
    }


def main() -> int:
    """Run the ONNX report and optionally enforce an activation whitelist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="ONNX model path")
    parser.add_argument("--check", action="store_true", help="fail on Softmax or a non-whitelisted Sigmoid")
    parser.add_argument("--allow-sigmoid", action="append", default=[], metavar="REGEX")
    args = parser.parse_args()
    report = inspect(args.model)
    print(json.dumps(report, indent=2))
    if not args.check:
        return 0
    blocked = [x for x in report["activations"] if x["op_type"] == "Softmax"]
    blocked += [
        x
        for x in report["activations"]
        if x["op_type"] == "Sigmoid" and not any(re.search(pattern, x["name"]) for pattern in args.allow_sigmoid)
    ]
    if blocked:
        print(f"Activation audit failed: {len(blocked)} non-whitelisted node(s).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
