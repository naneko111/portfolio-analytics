import numpy as np, pandas as pd, pytest
from portfolio_analytics import metrics

@pytest.fixture
def rnd():
    np.random.seed(0)
    return pd.Series(np.random.normal(0.0005, 0.01, 1000))

def test_sharpe_sign(rnd):
    assert metrics.sharpe_ratio(rnd) > 0

def test_var_positive(rnd):
    assert metrics.historical_var(rnd) > 0

def test_cvar_ge_var(rnd):
    assert metrics.cvar(rnd) >= metrics.historical_var(rnd)

def test_correlation_diagonal():
    r = pd.DataFrame(np.random.randn(500, 4), columns=list("ABCD"))
    c = metrics.correlation_matrix(r)
    assert np.allclose(np.diag(c), 1.0, atol=1e-8)
