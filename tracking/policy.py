import numpy as np
import onnxruntime as rt
from pathlib import Path
from dataclasses import dataclass


def _parse_device_id(device: str) -> int:
    """从设备字符串解析设备 ID

    Args:
        device: 设备字符串，如 "cuda:0" 或 "cpu"

    Returns:
        设备 ID，对于 "cuda:1" 返回 1，其他情况返回 0
    """
    if "cuda" in device and ":" in device:
        return int(device.split(":")[-1])
    return 0


def _get_onnx_providers(device: str):
    """根据设备类型获取 ONNX Runtime 执行提供者列表

    Args:
        device: 设备字符串，如 "cuda:0" 或 "cpu"

    Returns:
        执行提供者列表，按优先级排序
    """
    available = rt.get_available_providers()
    if "cuda" in device:
        device_id = _parse_device_id(device)
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {"device_id": device_id}))
        providers.append("CPUExecutionProvider")
        return providers
    return ["CPUExecutionProvider"]


# 对于小型策略 MLP（输入维度约几百），并行化反而有害：
# 线程池在层之间的同步开销主导了实际的矩阵乘法成本，
# 而且当循环休眠到控制频率时，工作线程会被挂起。
# 使用单个 intra-op 线程 + 自旋可以消除唤醒抖动
# （在 Jetson Orin 上，50Hz 循环的延迟从 ~2ms 降至 ~0.4ms）。
# 如果更大的模型需要更多并行化，可通过环境变量 HGPT_ORT_INTRA_THREADS 覆盖。
import os as _os

_DEFAULT_INTRA = int(_os.environ.get("HGPT_ORT_INTRA_THREADS", "1"))
_DEFAULT_INTER = int(_os.environ.get("HGPT_ORT_INTER_THREADS", "1"))


def _make_session_options(
    intra_op: int = _DEFAULT_INTRA,
    inter_op: int = _DEFAULT_INTER,
) -> rt.SessionOptions:
    """创建 ONNX Runtime 会话选项

    优化策略：
    1. 启用所有图优化
    2. 顺序执行模式（避免并行开销）
    3. 单线程 + 自旋锁（减少 50Hz 控制循环的唤醒抖动）
    4. 禁用动态块调度器（避免核心被让出）

    Args:
        intra_op: 算子内并行线程数
        inter_op: 算子间并行线程数

    Returns:
        配置好的会话选项
    """
    sess_opts = rt.SessionOptions()
    sess_opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.intra_op_num_threads = intra_op
    sess_opts.inter_op_num_threads = inter_op
    # 保持工作线程在调用之间自旋，使得周期性的 50Hz 循环
    # 不需要在每一步都支付 futex_wake 的成本
    try:
        sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
        sess_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
        # 抢占动态块调度器，避免其积极地让出核心
        sess_opts.add_session_config_entry("session.dynamic_block_base", "0")
    except Exception:
        pass
    return sess_opts


def _make_tensorrt_session(
    onnx_path: str,
    strict: bool = False,
    device_id: int = 0,
) -> rt.InferenceSession:
    """构建 ONNX Runtime TensorRT 会话

    TensorRT 可以显著加速推理，特别是在 NVIDIA GPU 上。
    首次运行时会构建并缓存 TensorRT 引擎，后续运行直接加载缓存。

    Args:
        onnx_path: ONNX 模型文件路径
        strict: 是否严格模式（True 时禁止回退到 CUDA/CPU）
        device_id: GPU 设备 ID

    Returns:
        配置好的推理会话
    """
    sess_opts = _make_session_options()

    # TensorRT 引擎缓存目录
    trt_cache_dir = Path("storage/logs/trt_cache") / Path(onnx_path).stem
    trt_cache_dir.mkdir(parents=True, exist_ok=True)
    trt_provider = ("TensorrtExecutionProvider", {
        "device_id": device_id,
        "trt_fp16_enable": False,  # 禁用 FP16（保持精度）
        "trt_max_workspace_size": 4 * 1024 ** 3,  # 最大工作空间 4GB
        "trt_engine_cache_enable": True,  # 启用引擎缓存
        "trt_engine_cache_path": str(trt_cache_dir),
        "trt_builder_optimization_level": 3,  # 最高优化级别
    })

    available = rt.get_available_providers()
    if strict:
        if "TensorrtExecutionProvider" not in available:
            raise RuntimeError(
                "strict_trt=True 但 TensorrtExecutionProvider 不可用。"
                f"available={available}"
            )
        providers = [trt_provider]
    else:
        # 非严格模式：按优先级尝试 TensorRT -> CUDA -> CPU
        providers = []
        if "TensorrtExecutionProvider" in available:
            providers.append(trt_provider)
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {"device_id": device_id}))
        providers.append("CPUExecutionProvider")

    session = rt.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
    actual = session.get_providers()
    print(f"[TensorRT] 已加载 {Path(onnx_path).name}  providers={actual}")
    if strict and (len(actual) == 0 or actual[0] != "TensorrtExecutionProvider"):
        raise RuntimeError(
            "strict_trt=True 但 TensorRT 不是活动的提供者顺序："
            f"{actual}"
        )
    return session


def _warmup_onnx(session: rt.InferenceSession, n: int = 20):
    """预热 ONNX 推理会话

    首次推理通常较慢（需要分配内存、编译内核等），
    通过预热可以消除首次推理的延迟尖峰。

    Args:
        session: ONNX 推理会话
        n: 预热迭代次数
    """
    inputs = session.get_inputs()
    feeds = {}
    for inp in inputs:
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        feeds[inp.name] = np.random.randn(*shape).astype(np.float32)
    for _ in range(n):
        session.run(None, feeds)
    print(f"[ONNX] 预热完成（{n} 次迭代）")


@dataclass
class Args:
    """策略配置参数"""
    policy_type: str = "mlp"  # 策略类型："mlp" | "transformer"
    device: str = "cuda:0"    # 推理设备
    onnx_track: str = ""      # ONNX 模型文件路径


class _ONNXFastInfer:
    """为 ONNX 推理添加基于 IOBinding 的快速路径的混入类

    ``infer(obs)`` 保持原始行为（每次调用返回新的 ndarray）。
    当调用者可以每步复用相同的输入缓冲区时（部署循环中的常见情况），
    调用 :py:meth:`bind_input_buffer` 一次，然后使用 :py:meth:`infer_bound`，
    它可以跳过 ORT 的每次调用输入复制 + 输出分配，并写入稳定的输出数组。

    这是实时控制的关键优化，可以将推理延迟从 ~2ms 降至 ~0.4ms。
    """

    onnx_model: rt.InferenceSession  # 由子类填充

    def _post_init(self) -> None:
        """初始化 IOBinding 相关的元数据"""
        # 缓存 I/O 元数据
        self._inp_name = self.onnx_model.get_inputs()[0].name  # "obs"
        self._out_name = "continuous_actions"
        self._iobinding = None
        self._obs_buf: np.ndarray | None = None
        self._out_buf: np.ndarray | None = None

    def bind_input_buffer(self, obs_shape: tuple[int, ...]) -> np.ndarray:
        """分配持久的（CPU）输入缓冲区 + IO 绑定

        返回输入缓冲区。调用者应在每次 :py:meth:`infer_bound` 调用之前
        将观测数据就地写入该缓冲区。

        Args:
            obs_shape: 观测向量的形状，如 (1, 136)

        Returns:
            输入缓冲区，调用者需要每步填充此缓冲区
        """
        out_shape = self.onnx_model.get_outputs()[0].shape
        out_shape = tuple(d if isinstance(d, int) and d > 0 else s
                          for d, s in zip(out_shape, obs_shape))
        # 回退：如果形状是动态的，使用已知的动作维度 29
        if any(not isinstance(d, int) or d <= 0
               for d in self.onnx_model.get_outputs()[0].shape):
            out_shape = (obs_shape[0], 29)

        self._obs_buf = np.zeros(obs_shape, dtype=np.float32)
        self._out_buf = np.zeros(out_shape, dtype=np.float32)

        self._iobinding = self.onnx_model.io_binding()
        self._iobinding.bind_input(
            name=self._inp_name,
            device_type="cpu",
            device_id=0,
            element_type=np.float32,
            shape=obs_shape,
            buffer_ptr=self._obs_buf.ctypes.data,
        )
        self._iobinding.bind_output(
            name=self._out_name,
            device_type="cpu",
            device_id=0,
            element_type=np.float32,
            shape=out_shape,
            buffer_ptr=self._out_buf.ctypes.data,
        )
        return self._obs_buf

    def infer_bound(self) -> np.ndarray:
        """使用预绑定的 IOBinding 缓冲区运行推理

        调用此方法时，``self._obs_buf`` 必须已经包含观测数据。
        返回输出缓冲区（在下一次推理之前不要修改）。

        Returns:
            输出缓冲区，包含网络输出的动作
        """
        self.onnx_model.run_with_iobinding(self._iobinding)
        return self._out_buf


class MLP_Policy_ONNX(_ONNXFastInfer):
    """MLP 策略的 ONNX 推理封装

    支持 CPU 和 CUDA 推理，自动选择最优的执行提供者。
    """

    def __init__(self, config):
        providers = _get_onnx_providers(getattr(config, "device", "cuda"))
        sess_opts = _make_session_options()
        self.onnx_model = rt.InferenceSession(
            config.onnx_track,
            sess_options=sess_opts,
            providers=providers,
        )
        print(
            f"[ONNX] 已加载 {config.onnx_track}  "
            f"providers={self.onnx_model.get_providers()}  "
            f"intra={sess_opts.intra_op_num_threads} "
            f"inter={sess_opts.inter_op_num_threads}"
        )
        _warmup_onnx(self.onnx_model)
        self._post_init()

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """标准推理接口

        Args:
            obs: 观测向量，形状 (batch_size, obs_dim)

        Returns:
            动作向量，形状 (batch_size, action_dim)
        """
        nn_action = self.onnx_model.run(["continuous_actions"], {"obs": obs})[0]
        return nn_action


class MLP_Policy_TensorRT(_ONNXFastInfer):
    """MLP 策略的 TensorRT 推理封装

    使用 NVIDIA TensorRT 加速推理，首次运行时会构建并缓存优化引擎。
    适合在 NVIDIA GPU 上部署，可以显著降低推理延迟。
    """

    def __init__(self, config, strict_trt: bool = False):
        """
        Args:
            config: 配置对象（包含 onnx_track, device 等）
            strict_trt: 是否严格模式（True 时强制使用 TensorRT，不允许回退）
        """
        self.onnx_model = _make_tensorrt_session(
            config.onnx_track,
            strict=strict_trt,
            device_id=_parse_device_id(getattr(config, "device", "cuda")),
        )
        _warmup_onnx(self.onnx_model)
        self._post_init()

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """标准推理接口"""
        nn_action = self.onnx_model.run(["continuous_actions"], {"obs": obs})[0]
        return nn_action


# Transformer ONNX 与 MLP 共享 "obs" -> "continuous_actions" 的接口约定；
# 只有观测形状不同（B, K, D）vs (B, D)，因此我们复用 MLP 的管道。
class Transformer_Policy_ONNX(MLP_Policy_ONNX):
    """Transformer 策略的 ONNX 推理封装

    与 MLP 的区别仅在于输入观测的形状：
    - MLP: (batch, obs_dim)
    - Transformer: (batch, seq_len, obs_dim)
    """
    pass


class Transformer_Policy_TensorRT(MLP_Policy_TensorRT):
    """Transformer 策略的 TensorRT 推理封装"""
    pass


def get_policy_onnx(
    args,
    use_trt: bool = False,
    strict_trt: bool = False,
):
    """根据配置创建策略推理对象

    Args:
        args: 配置对象，包含 policy_type, onnx_track, device 等
        use_trt: 是否使用 TensorRT 加速
        strict_trt: 是否严格模式（仅在使用 TensorRT 时有效）

    Returns:
        策略推理对象（MLP 或 Transformer，ONNX 或 TensorRT）

    Raises:
        ValueError: 未知的策略类型
    """
    if args.policy_type == "mlp":
        if use_trt:
            return MLP_Policy_TensorRT(args, strict_trt=strict_trt)
        return MLP_Policy_ONNX(args)
    if args.policy_type == "transformer":
        if use_trt:
            return Transformer_Policy_TensorRT(args, strict_trt=strict_trt)
        return Transformer_Policy_ONNX(args)
    raise ValueError(f"未知的策略类型: {args.policy_type}")
