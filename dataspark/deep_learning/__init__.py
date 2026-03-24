try:
    from dataspark.deep_learning.trainer import NeuralNetTrainer
    from dataspark.deep_learning.models import TabularMLP, LSTMForecaster
    __all__ = ["NeuralNetTrainer", "TabularMLP", "LSTMForecaster"]
except ImportError as _exc:
    _msg = (
        "dataspark.deep_learning requires PyTorch. "
        "Install it with:  pip install torch"
    )

    def _missing_dep(*_a, **_kw):
        raise ImportError(_msg)

    class TabularMLP:  # noqa: D101
        def __init__(self, *a, **kw):
            raise ImportError(_msg)

    class LSTMForecaster:  # noqa: D101
        def __init__(self, *a, **kw):
            raise ImportError(_msg)

    NeuralNetTrainer = _missing_dep
    __all__ = ["NeuralNetTrainer", "TabularMLP", "LSTMForecaster"]
