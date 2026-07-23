import pytest
import torch


@pytest.fixture(scope="session")
def cpu():
    return torch.device("cpu")


@pytest.fixture
def identity_quat():
    # (w, x, y, z) identity, batched
    return torch.tensor([[1.0, 0.0, 0.0, 0.0]])
