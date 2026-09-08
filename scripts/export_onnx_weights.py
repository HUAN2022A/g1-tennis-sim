"""ONNX 冻结策略权重导出：解析 ONNX graph（initializer + node 拓扑），
逐层导出权重/偏置/激活类型/附加参数到 flax 可加载的 npz。

实际结构以 graph 为准（不按文档写死）。已识别的线性链模式：
    Gemm(transB=1) [+ Sigmoid | Relu | Tanh | Sigmoid+Mul(x, sigmoid(x))=SiLU]
以及游离 initializer（如 std_param）经 Identity 暴露为附加 graph 输出。

用法（onnx 包未入依赖，用 --with 临时注入）：
    uv run --with onnx python scripts/export_onnx_weights.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from onnx import numpy_helper

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = REPO_ROOT / "storage/ckpts/pns_wo_priv264.onnx"
DEFAULT_OUT = REPO_ROOT / "storage/ckpts/rl/pns_wo_priv264_flax.npz"

# 文档宣称的架构（用于对比打印，不参与导出逻辑）
DOC_CLAIM = {
    "arch": "136 -> 1024 x3 -> 29",
    "activation": "Sigmoid x3（输出层无激活）",
    "params": 2_269_213,  # 136*1024+1024 + (1024*1024+1024)*2 + 29*1024+29 ≈ 2.28M
}

# 单算子激活（ONNX 图中的直连后继）
SIMPLE_ACTS = {"Sigmoid": "sigmoid", "Relu": "relu", "Tanh": "tanh"}


def _attr_dict(node):
    from onnx import helper

    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def _canonical_layer_id(node_name: str, weight_name: str) -> str:
    """'/mlp/mlp.0/Gemm' -> 'mlp.0'；退化时用权重名去后缀。"""
    if node_name:
        stem = node_name.rsplit("/", 1)[0].lstrip("/")
        if stem:
            return stem
    return weight_name.rsplit(".", 1)[0]


def extract_architecture(onnx_path: str | Path) -> dict:
    """沿数据流从 graph 输入走到输出，提取 Dense 层与附加参数。

    Returns:
        dict: {layers, extras, graph_input, graph_outputs, total_params}
    """
    import onnx

    graph = onnx.load(str(onnx_path)).graph
    inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}

    producer = {}          # tensor name -> 产出它的 node
    consumers = defaultdict(list)  # tensor name -> 消费它的 nodes
    for node in graph.node:
        for out in node.output:
            producer[out] = node
        for inp in node.input:
            if inp in inits or inp in producer or any(inp == g.name for g in graph.input):
                consumers[inp].append(node)

    graph_outputs = [o.name for o in graph.output]
    layers, used_inits = [], set()

    def gemm_layer(node) -> dict:
        attrs = _attr_dict(node)
        if not (node.op_type == "Gemm" and attrs.get("alpha", 1.0) == 1.0
                and attrs.get("beta", 1.0) == 1.0
                and attrs.get("transB", 0) == 1 and len(node.input) == 3):
            raise NotImplementedError(
                f"不支持的 Gemm 形式: name={node.name} attrs={attrs}")
        w_name, b_name = node.input[1], node.input[2]
        weight, bias = inits[w_name], inits[b_name]
        if weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise NotImplementedError(f"Gemm 权重形状异常: {w_name}{weight.shape}")
        used_inits.update((w_name, b_name))
        return {
            "id": _canonical_layer_id(node.name, w_name),
            "type": "gemm",
            "in_features": int(weight.shape[1]),
            "out_features": int(weight.shape[0]),
            "weight_key": w_name,
            "bias_key": b_name,
            "trans_b": True,
            "activation": "linear",  # 由调用方按后继节点填充
        }

    def classify_activation(tensor: str) -> tuple[str, str]:
        """检查 tensor 的后继，返回 (activation, 下一层输入 tensor)。"""
        succs = consumers.get(tensor, [])
        if not succs:
            return "linear", tensor
        if len(succs) == 1 and succs[0].op_type == "Gemm":
            return "linear", tensor
        # SiLU / Swish 模式: x 同时进入 Sigmoid 与 Mul, Mul(x, Sigmoid(x))
        muls = [n for n in succs if n.op_type == "Mul"]
        sigs = [n for n in succs if n.op_type == "Sigmoid"]
        if len(sigs) == 1 and len(muls) == 1 and tensor in muls[0].input \
                and muls[0] in consumers.get(sigs[0].output[0], []):
            return "silu", muls[0].output[0]
        if len(succs) == 1 and succs[0].op_type in SIMPLE_ACTS:
            return SIMPLE_ACTS[succs[0].op_type], succs[0].output[0]
        raise NotImplementedError(
            f"无法识别的激活模式: {tensor!r} 的后继 {[n.op_type for n in succs]}")

    cur = graph.input[0].name
    while True:
        succs = [n for n in consumers.get(cur, []) if n.op_type != "Identity"]
        if not succs:
            break  # 到达 graph 输出，主链结束
        if len(succs) > 1:
            raise NotImplementedError(
                f"主链在 {cur!r} 处分叉: {[n.op_type for n in succs]}")
        node = succs[0]
        if node.op_type == "Gemm":
            layer = gemm_layer(node)
            layer["activation"], cur = classify_activation(node.output[0])
            layers.append(layer)
        else:
            raise NotImplementedError(
                f"主链上不支持的节点: {node.op_type} ({node.name})")

    # 未被主链 Gemm 消费的 initializer -> 附加参数（记录其 Identity 输出名，若有）
    identity_out = {n.input[0]: n.output[0] for n in graph.node if n.op_type == "Identity"}
    extras = [
        {
            "initializer": name,
            "shape": list(arr.shape),
            "graph_output": identity_out.get(name),
        }
        for name, arr in inits.items() if name not in used_inits
    ]

    total_params = int(sum(
        l["in_features"] * l["out_features"] + l["out_features"] for l in layers))
    total_params += int(sum(int(np.prod(e["shape"])) for e in extras))
    return {
        "layers": layers,
        "extras": extras,
        "inits": inits,
        "graph_input": {"name": graph.input[0].name,
                        "dim": int(graph.input[0].type.tensor_type.shape.dim[1].dim_value)},
        "graph_outputs": graph_outputs,
        "total_params": total_params,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    arch = extract_architecture(args.onnx)

    out_dim = arch["layers"][-1]["out_features"]
    metadata = {
        "format_version": 1,
        "source_onnx": str(Path(args.onnx).resolve()),
        "input": arch["graph_input"],
        "output": {"name": "continuous_actions", "dim": out_dim},
        "graph_outputs": arch["graph_outputs"],
        "layers": arch["layers"],
        "extras": arch["extras"],
        "total_params": arch["total_params"],
    }

    arrays = {"metadata": np.array(json.dumps(metadata, ensure_ascii=False, indent=2))}
    for layer in arch["layers"]:
        arrays[layer["weight_key"]] = arch["inits"][layer["weight_key"]]
        arrays[layer["bias_key"]] = arch["inits"][layer["bias_key"]]
    for extra in arch["extras"]:
        arrays[extra["initializer"]] = arch["inits"][extra["initializer"]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)

    # ---------------- 打印实际架构与文档对比 ----------------
    print(f"ONNX graph: {args.onnx}")
    print(f"  输入 {arch['graph_input']['name']}({arch['graph_input']['dim']}) "
          f"-> graph 输出 {arch['graph_outputs']}")
    chain = f"{arch['graph_input']['dim']}" + "".join(
        f" -> {l['out_features']}" for l in arch["layers"])
    act_chain = " -> ".join(
        f"{l['id']}:{l['activation']}" for l in arch["layers"])
    print(f"\n实际架构: {chain}")
    print(f"实际激活: {act_chain}")
    print(f"\n{'层':<12}{'形状':<16}{'参数量':>12}  激活")
    for layer in arch["layers"]:
        n = layer["in_features"] * layer["out_features"] + layer["out_features"]
        print(f"{layer['id']:<12}{layer['in_features']:>5}x{layer['out_features']:<8}"
              f"{n:>12,}  {layer['activation']}")
    for extra in arch["extras"]:
        print(f"{extra['initializer']:<12}{tuple(extra['shape'])!s:<16}"
              f"{int(np.prod(extra['shape'])):>12,}  (附加参数, graph 输出 "
              f"{extra['graph_output']})")
    print(f"合计参数量: {arch['total_params']:,}")

    print(f"\n文档宣称: {DOC_CLAIM['arch']} | {DOC_CLAIM['activation']} "
          f"| ~{DOC_CLAIM['params']:,} 参数")
    doc_dims = [int(d) for d in DOC_CLAIM["arch"].replace("->", " ").replace("x3", "").split()]
    act_dims = [arch["graph_input"]["dim"]] + [l["out_features"] for l in arch["layers"]]
    if doc_dims != act_dims:
        print(f"  [差异] 层宽 {doc_dims} vs 实际 {act_dims}")
    doc_acts = [a for a in act_chain.split(" -> ")]
    silu_like = all(a.endswith(":silu") for a in doc_acts[:-1]) and doc_acts[-1].endswith(":linear")
    print(f"  [差异] 激活: 文档 Sigmoid x3 vs 实际 "
          f"{'SiLU(x)=x*sigmoid(x) x3' if silu_like else doc_acts}")
    if arch["extras"]:
        print(f"  [差异] 文档未提及附加参数: "
              f"{[e['initializer'] + str(e['shape']) for e in arch['extras']]}")
    delta = arch["total_params"] - DOC_CLAIM["params"]
    print(f"  [差异] 参数量: 文档 ~{DOC_CLAIM['params']:,} vs 实际 {arch['total_params']:,} "
          f"(+{delta:,}, 与文件 {args.onnx.stat().st_size / 1e6:.2f}MB ≈ "
          f"{arch['total_params'] * 4 / 1e6:.2f}M f32 参数一致)")

    print(f"\n已写出: {args.out}")
    print(f"flax 加载: from rl.frozen_policy import load_policy; "
          f"model = load_policy('{args.out.relative_to(REPO_ROOT)}')")


if __name__ == "__main__":
    main()
