from __future__ import annotations

import pytest
import torch

from hyset.datasets.synthetic import generate
from hyset.encoders import HashingEncoder
from hyset.inference import InferenceConfig
from hyset.model import ModelConfig
from hyset.negatives import NegativeMix
from hyset.run import save_run
from hyset.train import TrainConfig, train


@pytest.fixture(scope="session")
def tiny_ds():
    return generate(n_train=600, n_val=80, n_test=120, n_filler=20, seed=1)


@pytest.fixture(scope="session")
def encoder():
    return HashingEncoder(256)


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory, tiny_ds, encoder):
    """A small trained run on disk (hashing encoder, a few epochs)."""
    torch.set_num_threads(4)
    data_dir = tmp_path_factory.mktemp("data")
    tiny_ds.save(data_dir)
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(
        epochs=3, lr=1e-3, k_neg=24, patience=5,
        mix=NegativeMix(size_matched=0.3, in_batch=0.3, hard=0.2, model=0.2),
    )  # fmt: skip
    infer_cfg = InferenceConfig(k1=8, k_pool=12)
    result = train(tiny_ds, encoder, model_cfg, train_cfg, infer_cfg)
    run_dir = tmp_path_factory.mktemp("run")
    save_run(run_dir, result, tiny_ds.library, model_cfg=model_cfg, train_cfg=train_cfg,
             infer_cfg=infer_cfg, encoder_spec=encoder.spec, dataset_dir=str(data_dir))  # fmt: skip
    return run_dir, data_dir, result
