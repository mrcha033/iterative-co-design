import importlib
import sys
import types
from typing import Any, Tuple

import pytest


class FakeNoGrad:
    def __enter__(self) -> None:  # pragma: no cover - trivial
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - trivial
        return False


class FakeTensor:
    def __init__(self, shape: Tuple[int, ...], dtype: Any = None):
        self.shape = tuple(shape)
        self._dtype = dtype
        self.device = None

    def to(self, device: str | None = None, dtype: Any | None = None) -> "FakeTensor":
        if device is not None:
            self.device = device
        if dtype is not None:
            self._dtype = dtype
        return self

    def dim(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        total = 1
        for s in self.shape:
            total *= s
        return total

    def __getitem__(self, _key) -> "FakeTensor":  # slicing noop
        return self

    @property
    def dtype(self) -> Any:
        return self._dtype


class FakeModel:
    def __init__(self, name: str = "model", torch_dtype: Any | None = None):
        self.name = name
        self.torch_dtype = torch_dtype
        self.device = None
        self.calls = 0

    def to(self, device: str | None = None, dtype: Any | None = None) -> "FakeModel":
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        return self

    def eval(self) -> "FakeModel":
        return self

    def __call__(self, *args, **kwargs) -> None:
        self.calls += 1
        return FakeTensor((1, 1), dtype=self.torch_dtype)


def fake_sequence_loader() -> Tuple[FakeModel, Tuple[FakeTensor, ...]]:
    model = FakeModel("seq_loader")
    example = (FakeTensor((4, 8)), FakeTensor((4, 8)))
    return model, example


def fake_causal_loader() -> Tuple[FakeModel, Tuple[FakeTensor, ...]]:
    model = FakeModel("causal_loader")
    example = (FakeTensor((1, 32)),)
    return model, example


@pytest.fixture()
def fake_torch_env(monkeypatch):
    # torch module
    torch_mod = types.ModuleType("torch")
    torch_mod.float32 = types.SimpleNamespace(itemsize=4)
    torch_mod.float16 = types.SimpleNamespace(itemsize=2)
    torch_mod.bfloat16 = types.SimpleNamespace(itemsize=2)
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_mod.manual_seed = lambda seed: None
    torch_mod.no_grad = FakeNoGrad
    torch_mod.Tensor = FakeTensor

    def _zeros(shape, dtype=None, device=None):
        tensor = FakeTensor(tuple(shape), dtype=dtype)
        if device is not None:
            tensor.device = device
        return tensor

    torch_mod.zeros = _zeros

    # torch.nn module
    nn_mod_root = types.ModuleType("torch.nn")

    class FakeNNModule:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def to(self, *args, **kwargs):  # pragma: no cover - trivial
            return self

        def eval(self):  # pragma: no cover - trivial
            return self

        def parameters(self):  # pragma: no cover - trivial
            return []

    class FakeLinear(FakeNNModule):
        pass

    nn_mod_root.Module = FakeNNModule
    nn_mod_root.Linear = FakeLinear
    torch_mod.nn = nn_mod_root
    monkeypatch.setitem(sys.modules, "torch.nn", nn_mod_root)

    nn_utils_mod = types.ModuleType("torch.nn.utils")
    prune_mod = types.ModuleType("torch.nn.utils.prune")

    def _noop(*args, **kwargs):  # pragma: no cover - trivial
        return None

    prune_mod.global_unstructured = _noop
    prune_mod.l1_unstructured = _noop
    nn_utils_mod.prune = prune_mod

    monkeypatch.setitem(sys.modules, "torch.nn.utils", nn_utils_mod)
    monkeypatch.setitem(sys.modules, "torch.nn.utils.prune", prune_mod)

    def _no_grad():
        return FakeNoGrad()

    torch_mod.no_grad = _no_grad

    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    # torch.profiler
    profiler_mod = types.ModuleType("torch.profiler")
    profiler_mod.ProfilerActivity = types.SimpleNamespace(CPU="cpu", CUDA="cuda")

    class FakeProfile:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self) -> "FakeProfile":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def events(self):
            return [
                types.SimpleNamespace(key="linear"),
                types.SimpleNamespace(key="matmul"),
                types.SimpleNamespace(key="reshape"),
            ]

    profiler_mod.profile = lambda *args, **kwargs: FakeProfile()
    monkeypatch.setitem(sys.modules, "torch.profiler", profiler_mod)

    # torch.fx
    fx_mod = types.ModuleType("torch.fx")

    class FakeNode:
        def __init__(self, op: str, target: str, shape: Tuple[int, ...] | None):
            self.op = op
            self.target = target
            if shape is None:
                meta = None
            else:
                meta = types.SimpleNamespace(shape=shape, dtype=torch_mod.float32)
            self.meta = {"tensor_meta": meta} if meta else {}

    class FakeGraphModule:
        def __init__(self) -> None:
            self.graph = types.SimpleNamespace(nodes=[
                FakeNode("placeholder", "input", None),
                FakeNode("call_function", "linear", (1, 4, 16)),
                FakeNode("call_function", "permute", (1, 4, 16)),
            ])

    fx_mod.symbolic_trace = lambda model: FakeGraphModule()
    monkeypatch.setitem(sys.modules, "torch.fx", fx_mod)

    passes_mod = types.ModuleType("torch.fx.passes")
    monkeypatch.setitem(sys.modules, "torch.fx.passes", passes_mod)
    shp_mod = types.ModuleType("torch.fx.passes.shape_prop")

    class ShapeProp:
        def __init__(self, gm):
            self.gm = gm

        def propagate(self, *args, **kwargs) -> None:
            return None

    shp_mod.ShapeProp = ShapeProp
    monkeypatch.setitem(sys.modules, "torch.fx.passes.shape_prop", shp_mod)

    # transformers module
    transformers_mod = types.ModuleType("transformers")

    class FakeTokenizer:
        def __init__(self, name: str) -> None:
            self.name = name

        @classmethod
        def from_pretrained(cls, name: str) -> "FakeTokenizer":
            return cls(name)

        def __call__(self, texts, return_tensors="pt", padding="max_length", truncation=True, max_length=128):
            if not isinstance(texts, (list, tuple)):
                texts = [texts]
            shape = (len(texts), max_length)
            return {
                "input_ids": FakeTensor(shape, dtype=torch_mod.float32),
                "attention_mask": FakeTensor(shape, dtype=torch_mod.float32),
                "token_type_ids": FakeTensor(shape, dtype=torch_mod.float32),
            }

    class FakeSeqModel(FakeModel):
        @classmethod
        def from_pretrained(cls, name: str, torch_dtype=None):
            return cls(name, torch_dtype=torch_dtype)

    class FakeCausalModel(FakeModel):
        @classmethod
        def from_pretrained(cls, name: str, torch_dtype=None):
            return cls(name, torch_dtype=torch_dtype)

    transformers_mod.AutoTokenizer = FakeTokenizer
    transformers_mod.AutoModelForSequenceClassification = FakeSeqModel
    transformers_mod.AutoModelForCausalLM = FakeCausalModel
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    # torchvision module
    vision_mod = types.ModuleType("torchvision")
    models_mod = types.ModuleType("torchvision.models")

    class FakeWeights:
        def __init__(self, name: str):
            self.name = name

    class ResNet50Weights:
        DEFAULT = FakeWeights("DEFAULT")
        IMAGENET1K_V1 = FakeWeights("IMAGENET1K_V1")
        IMAGENET1K_V2 = FakeWeights("IMAGENET1K_V2")

    def fake_resnet50(weights=None, **kwargs) -> FakeModel:
        model = FakeModel("resnet50", torch_dtype=getattr(weights, "name", None))
        model.weights = weights
        return model

    models_mod.ResNet50_Weights = ResNet50Weights
    models_mod.resnet50 = fake_resnet50
    vision_mod.models = models_mod
    monkeypatch.setitem(sys.modules, "torchvision", vision_mod)
    monkeypatch.setitem(sys.modules, "torchvision.models", models_mod)

    # torch_geometric module
    pyg_mod = types.ModuleType("torch_geometric")
    datasets_mod = types.ModuleType("torch_geometric.datasets")
    nn_mod = types.ModuleType("torch_geometric.nn")
    nn_models_mod = types.ModuleType("torch_geometric.nn.models")

    class FakeGraphData:
        def __init__(self) -> None:
            self.x = FakeTensor((5, 16), dtype=torch_mod.float32)
            self.edge_index = FakeTensor((2, 12), dtype=torch_mod.float32)

    class FakeDataset:
        def __init__(self) -> None:
            self.num_features = 16
            self.num_classes = 3

        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int) -> FakeGraphData:
            return FakeGraphData()

    def fake_arxiv(root=None, **kwargs):
        return FakeDataset()

    def fake_planetoid(root=None, name=None, **kwargs):
        return FakeDataset()

    datasets_mod.OgbnArxiv = fake_arxiv
    datasets_mod.Planetoid = fake_planetoid

    class FakeGCN(FakeModel):
        pass

    class FakeGraphSAGE(FakeModel):
        pass

    def fake_gcn(**kwargs) -> FakeGCN:
        return FakeGCN("gcn", torch_dtype=kwargs.get("dtype"))

    def fake_graphsage(**kwargs) -> FakeGraphSAGE:
        return FakeGraphSAGE("graphsage", torch_dtype=kwargs.get("dtype"))

    nn_models_mod.GCN = fake_gcn
    nn_models_mod.GraphSAGE = fake_graphsage
    nn_mod.models = nn_models_mod
    pyg_mod.datasets = datasets_mod
    pyg_mod.nn = nn_mod

    monkeypatch.setitem(sys.modules, "torch_geometric", pyg_mod)
    monkeypatch.setitem(sys.modules, "torch_geometric.datasets", datasets_mod)
    monkeypatch.setitem(sys.modules, "torch_geometric.nn", nn_mod)
    monkeypatch.setitem(sys.modules, "torch_geometric.nn.models", nn_models_mod)

    # Reload target modules now that stubs exist
    import icd.core.graph_pytorch as gp
    import icd.runtime.runners_hf as rhf
    import icd.experiments.hf as ehf
    import icd.experiments.vision as evision
    import icd.experiments.graph as egraph

    gp = importlib.reload(gp)
    rhf = importlib.reload(rhf)
    ehf = importlib.reload(ehf)
    evision = importlib.reload(evision)
    egraph = importlib.reload(egraph)

    yield types.SimpleNamespace(torch=torch_mod, gp=gp, rhf=rhf, ehf=ehf, evision=evision, egraph=egraph)

    # Cleanup handled by monkeypatch


def test_graph_pytorch_build_w_with_fake_torch(fake_torch_env, monkeypatch):
    gp = fake_torch_env.gp
    monkeypatch.setattr(gp, "_infer_feature_dim_from_fx", lambda gm, fallback=0: fallback or 32, raising=False)
    model = FakeModel("graph")
    inputs = (FakeTensor((1, 4, 16), dtype=fake_torch_env.torch.float32),)
    W = gp.build_w_from_pytorch(model, inputs, hops=2, reuse_decay=0.5, max_len=32, seed=1)
    assert W.meta["source"] == "pytorch"
    assert W.nnz() > 0
    assert "pytorch" in W.meta


def test_hf_sequence_runner_with_cached_model(fake_torch_env):
    rhf = fake_torch_env.rhf
    context = {
        "graph_model": FakeModel("seq"),
        "graph_example_inputs": (FakeTensor((2, 8), dtype=fake_torch_env.torch.float32),),
    }
    result = rhf.hf_sequence_classifier_runner("iterative", context)
    assert result.get("tokens") == 16
    # cached entries stored for reuse
    assert context["_hf_cache"]["model"].name == "seq"


def test_hf_sequence_runner_loader_path(fake_torch_env):
    rhf = fake_torch_env.rhf
    context = {
        "model_loader": "tests.unit.test_torch_stubs:fake_sequence_loader",
    }
    result = rhf.hf_sequence_classifier_runner("linear", context)
    assert result["tokens"] == 32


def test_hf_causal_runner(fake_torch_env):
    rhf = fake_torch_env.rhf
    context = {
        "model_loader": "tests.unit.test_torch_stubs:fake_causal_loader",
    }
    result = rhf.hf_causal_lm_runner("iterative", context)
    assert result["tokens"] == 32


def test_experiment_loaders_with_stubs(fake_torch_env):
    ehf = fake_torch_env.ehf
    model_seq, inputs_seq = ehf.load_hf_sequence_classifier("demo-seq", batch_size=2, sequence_length=4)
    assert isinstance(model_seq, FakeModel)
    assert len(inputs_seq) >= 2

    model_causal, inputs_causal = ehf.load_hf_causal_lm("demo-causal", batch_size=1, sequence_length=8)
    assert isinstance(model_causal, FakeModel)
    assert len(inputs_causal) >= 1


def test_torchvision_resnet_loader(fake_torch_env, monkeypatch):
    evision = fake_torch_env.evision
    gp = fake_torch_env.gp
    monkeypatch.setattr(gp, "_infer_feature_dim_from_fx", lambda gm, fallback=0: fallback or 32, raising=False)
    model, example = evision.load_torchvision_resnet50(batch_size=2, image_size=128, device="cpu")
    assert isinstance(model, FakeModel)
    assert len(example) == 1
    assert example[0].shape == (2, 3, 128, 128)
    assert model.device == "cpu"

    W = gp.build_w_from_pytorch(model, example, hops=1, reuse_decay=0.5, max_len=16, seed=7)
    assert W.meta["source"] == "pytorch"


def test_pyg_gcn_loader(fake_torch_env, monkeypatch):
    egraph = fake_torch_env.egraph
    gp = fake_torch_env.gp
    monkeypatch.setattr(gp, "_infer_feature_dim_from_fx", lambda gm, fallback=0: fallback or 32, raising=False)
    model, example = egraph.load_pyg_gcn(dataset_name="ogbn-arxiv", hidden_channels=64, num_layers=2, device="cpu")
    assert isinstance(model, FakeModel)
    assert len(example) == 2
    x, edge_index = example
    assert x.shape[0] == 5
    assert edge_index.shape[0] == 2

    W = gp.build_w_from_pytorch(model, example, hops=1, reuse_decay=0.5, max_len=16, seed=9)
    assert W.meta["source"] == "pytorch"


def test_pyg_graphsage_loader(fake_torch_env, monkeypatch):
    egraph = fake_torch_env.egraph
    gp = fake_torch_env.gp
    monkeypatch.setattr(gp, "_infer_feature_dim_from_fx", lambda gm, fallback=0: fallback or 32, raising=False)
    model, example = egraph.load_pyg_graphsage(dataset_name="ogbn-arxiv", hidden_channels=32, num_layers=2, device="cpu")
    assert isinstance(model, FakeModel)
    assert len(example) == 2
    W = gp.build_w_from_pytorch(model, example, hops=1, reuse_decay=0.5, max_len=16, seed=11)
    assert W.meta["source"] == "pytorch"
