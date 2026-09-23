"""Trainer-specific network → ONNX graph (docs/design/0009 Decision 4).

Every exporter returns an `Exported`: the graph (all weights as ordinary initializers -- sharding
into external files happens later, in `packaging`, the same way for every trainer), a reference
forward pass to check the graph against, and descriptive metadata. Graphs take a `[batch, inputs]`
tensor named `observation` and return `[batch, outputs]` named `outputs`, in the export's dtype, so a
client or an evaluation job can batch many decisions into one call.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import onnx
from evolve.neat import NeatGenome
from evolve.networks import describe, parameter_count
from evolve.neuro import WeightVector
from onnx import TensorProto, helper, numpy_helper

from modelpack.champions import Champion, QTable, load_champion

# Opset 17 / IR 8: old enough that every ONNX Runtime from the last few years (Python and Web) runs it,
# new enough for everything these exporters emit.
OPSET = 17
IR_VERSION = 8

INPUT = "observation"
OUTPUT = "outputs"

# float64 exists because the evolved networks are float64 all the way through (`math.tanh` on Python
# floats): rounding their weights to float32 flips real decisions near ties (docs/design/0009), while
# ONNX Runtime's CPU/WASM kernels in float64 agree with the Python on every decision measured.
DTYPES: dict[str, tuple[type, int]] = {
    "float32": (np.float32, TensorProto.FLOAT),
    "float64": (np.float64, TensorProto.DOUBLE),
}


@dataclass(frozen=True)
class Exported:
    model: onnx.ModelProto
    reference: Callable[[Sequence[float]], list[float]]
    reference_name: str
    trainer: str
    source_format: str
    description: str
    parameters: int
    num_inputs: int
    num_outputs: int
    dtype: str


def _finish(nodes, initializers, num_inputs: int, num_outputs: int, name: str, dtype: str) -> onnx.ModelProto:
    elem = DTYPES[dtype][1]
    graph = helper.make_graph(
        nodes,
        name,
        inputs=[helper.make_tensor_value_info(INPUT, elem, ["batch", num_inputs])],
        outputs=[helper.make_tensor_value_info(OUTPUT, elem, ["batch", num_outputs])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="modelpack")
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    return model


def export_weight_vector(network: WeightVector, dtype: str = "float32") -> Exported:
    """A fixed MLP: one `Gemm → Tanh` per layer (every layer, output included, is tanh-squashed --
    `evolve.neuro._forward`). The flat weight layout is, per layer, `out*in` weights (row = output
    unit) then `out` biases."""
    np_dtype = DTYPES[dtype][0]
    sizes = network.layer_sizes
    weights = np.asarray(network.weights, dtype=np.float64)
    nodes, initializers = [], []
    current, offset = INPUT, 0
    for i in range(len(sizes) - 1):
        fan_in, fan_out = sizes[i], sizes[i + 1]
        w = weights[offset : offset + fan_in * fan_out].reshape(fan_out, fan_in)
        b = weights[offset + fan_in * fan_out : offset + fan_in * fan_out + fan_out]
        offset += fan_in * fan_out + fan_out
        initializers += [
            numpy_helper.from_array(w.astype(np_dtype), f"layer{i}.weight"),
            numpy_helper.from_array(b.astype(np_dtype), f"layer{i}.bias"),
        ]
        last = i == len(sizes) - 2
        pre, out = (f"{OUTPUT}_pre", OUTPUT) if last else (f"layer{i}.pre", f"layer{i}.out")
        nodes += [
            helper.make_node("Gemm", [current, f"layer{i}.weight", f"layer{i}.bias"], [pre], transB=1),
            helper.make_node("Tanh", [pre], [out]),
        ]
        current = out
    return Exported(
        model=_finish(nodes, initializers, sizes[0], sizes[-1], "weight_vector", dtype),
        reference=network.forward,
        reference_name="evolve.neuro.WeightVector.forward (float64)",
        trainer="evolve.neuro",
        source_format="weight_vector.json",
        description=f"MLP {describe(network)}, tanh",
        parameters=parameter_count(network),
        num_inputs=sizes[0],
        num_outputs=sizes[-1],
        dtype=dtype,
    )


def neat_layers(genome: NeatGenome) -> list[list[tuple[int, list[tuple[int, float]]]]]:
    """The genome's live nodes grouped by depth: a node's depth is 1 + the deepest node feeding it
    (inputs and bias are depth 0). Every node in one group depends only on earlier groups, so a group is
    one dense matrix multiply. Returns, per group, `(slot, [(source_slot, weight), ...])` using
    `NeatGenome._plan`'s value slots -- the same compiled, dead-end-pruned order `forward()` uses."""
    _, steps, _ = genome._plan
    depth = dict.fromkeys(range(genome.num_inputs + 1), 0)
    groups: dict[int, list[tuple[int, list[tuple[int, float]]]]] = {}
    for slot, incoming in steps:
        d = 1 + max((depth[source] for source, _ in incoming), default=0)
        depth[slot] = d
        groups.setdefault(d, []).append((slot, incoming))
    return [groups[d] for d in sorted(groups)]


def export_neat(genome: NeatGenome, dtype: str = "float32") -> Exported:
    """An evolved DAG as a few dense layers instead of hundreds of scalar ops (docs/design/0009).

    A running activation matrix `V` starts as the observation; each depth group is
    `tanh(V @ W + b)` with zeros wherever no gene connects a column of `V` to that node, and its
    result is concatenated onto `V`, so a node can read from *any* earlier depth (NEAT's skip
    connections). The bias node isn't a column: its genes become the Gemm's bias vector. Outputs are
    gathered from `V` at the end."""
    np_dtype = DTYPES[dtype][0]
    num_inputs = genome.num_inputs
    bias_slot = num_inputs
    _, _, output_slots = genome._plan
    column = {slot: slot for slot in range(num_inputs)}  # value slot -> column of V
    width = num_inputs
    nodes, initializers = [], []
    current = INPUT
    for g, group in enumerate(neat_layers(genome)):
        w = np.zeros((width, len(group)), dtype=np.float64)
        b = np.zeros(len(group), dtype=np.float64)
        for j, (_slot, incoming) in enumerate(group):
            for source, weight in incoming:
                if source == bias_slot:
                    b[j] += weight
                else:
                    w[column[source], j] += weight
        initializers += [
            numpy_helper.from_array(w.astype(np_dtype), f"depth{g}.weight"),
            numpy_helper.from_array(b.astype(np_dtype), f"depth{g}.bias"),
        ]
        nodes += [
            helper.make_node("Gemm", [current, f"depth{g}.weight", f"depth{g}.bias"], [f"depth{g}.pre"]),
            helper.make_node("Tanh", [f"depth{g}.pre"], [f"depth{g}.out"]),
            helper.make_node("Concat", [current, f"depth{g}.out"], [f"depth{g}.values"], axis=1),
        ]
        for j, (slot, _) in enumerate(group):
            column[slot] = width + j
        width += len(group)
        current = f"depth{g}.values"
    indices = np.asarray([column[s] for s in output_slots], dtype=np.int64)
    initializers.append(numpy_helper.from_array(indices, "output_columns"))
    nodes.append(helper.make_node("Gather", [current, "output_columns"], [OUTPUT], axis=1))
    return Exported(
        model=_finish(nodes, initializers, num_inputs, genome.num_outputs, "neat", dtype),
        reference=genome.forward,
        reference_name="evolve.neat.NeatGenome.forward (float64)",
        trainer="evolve.neat",
        source_format="neat.json",
        description=f"evolved graph {describe(genome)}, tanh (neat)",
        parameters=parameter_count(genome),
        num_inputs=num_inputs,
        num_outputs=genome.num_outputs,
        dtype=dtype,
    )


def export_qtable(table: QTable, dtype: str = "float32") -> Exported:
    """A tabular policy (docs/design/0010 Phase 1) as a lookup: `row = (observation != 0) . [1, 2, 4, ...]`, then
    `Gather` that row of the table -- the action values an interface's argmax decodes, exactly as the agent's greedy
    policy does. Only binary-discretized tables export: a binned one would need its bin edges in the graph, and no
    game on a leaderboard uses one."""
    if table.discretizer["kind"] != "binary":
        raise ValueError(f"only a binary-discretized table exports, not {table.discretizer['kind']!r}")
    np_dtype, elem = DTYPES[dtype]
    bits, actions = table.num_inputs, table.num_actions
    powers = np.asarray([[float(1 << i)] for i in range(bits)], dtype=np_dtype)
    values = np.asarray(table.values, dtype=np.float64).reshape(table.states, actions).astype(np_dtype)
    initializers = [
        numpy_helper.from_array(np.zeros((), dtype=np_dtype), "zero"),
        numpy_helper.from_array(powers, "powers"),
        numpy_helper.from_array(np.asarray([-1], dtype=np.int64), "flat"),
        numpy_helper.from_array(values, "table"),
    ]
    nodes = [
        helper.make_node("Equal", [INPUT, "zero"], ["is_zero"]),
        helper.make_node("Not", ["is_zero"], ["is_set"]),
        helper.make_node("Cast", ["is_set"], ["bits"], to=elem),
        helper.make_node("MatMul", ["bits", "powers"], ["row_2d"]),
        helper.make_node("Reshape", ["row_2d", "flat"], ["row_float"]),
        helper.make_node("Cast", ["row_float"], ["row"], to=TensorProto.INT64),
        helper.make_node("Gather", ["table", "row"], [OUTPUT], axis=0),
    ]
    return Exported(
        model=_finish(nodes, initializers, bits, actions, "qtable", dtype),
        reference=table.forward,
        reference_name=f"modelpack.champions.QTable.forward ({table.algorithm}, float64)",
        trainer="rl.tabular",
        source_format="qtable.json",
        description=f"{table.algorithm} table, {table.states} states x {actions} actions",
        parameters=len(table.values),
        num_inputs=bits,
        num_outputs=actions,
        dtype=dtype,
    )


def export_network(network: Champion, dtype: str = "float32") -> Exported:
    if isinstance(network, NeatGenome):
        return export_neat(network, dtype)
    if isinstance(network, WeightVector):
        return export_weight_vector(network, dtype)
    if isinstance(network, QTable):
        return export_qtable(network, dtype)
    raise TypeError(f"no exporter for {type(network).__name__}")


def export_network_json(text: str, dtype: str = "float32") -> Exported:
    """Export a champion artifact as stored by the training jobs (`modelpack.champions.load_champion`)."""
    json.loads(text)  # fail early, with a JSON error rather than an exporter one, on a corrupt artifact
    return export_network(load_champion(text), dtype)
