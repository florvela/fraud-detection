"""Tests de las defensas federadas (numpy puro, sin torch/flwr).

Verifican que clipping, agregación robusta y ruido DP se comportan como se espera y,
sobre todo, que la **mediana resiste a un cliente malicioso (outlier)** mucho mejor
que el promedio FedAvg — que es el punto de la clase de seguridad.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

# services/federated no es un paquete instalable: lo agregamos al path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "federated"))

robust = pytest.importorskip("robust")


def _weights(scale: float) -> list[np.ndarray]:
    """Un 'modelo' de 2 capas con todos los valores = scale."""
    return [np.full((3, 2), scale, dtype=np.float32), np.full((2,), scale, dtype=np.float32)]


def test_fedavg_pondera_por_ejemplos():
    a, b = _weights(0.0), _weights(10.0)
    out = robust.aggregate([a, b], num_examples=[3, 1], method="fedavg")
    # promedio ponderado: (3*0 + 1*10)/4 = 2.5
    assert np.allclose(out[0], 2.5)


def test_mediana_resiste_cliente_malicioso():
    # Dos clientes honestos (~1.0) y uno malicioso enorme (1000).
    honest1, honest2, evil = _weights(1.0), _weights(1.1), _weights(1000.0)
    fedavg = robust.aggregate([honest1, honest2, evil], method="mean")
    median = robust.aggregate([honest1, honest2, evil], method="median")
    # El promedio se dispara por el outlier; la mediana se queda cerca de los honestos.
    assert fedavg[0].mean() > 100
    assert np.allclose(median[0], 1.1)


def test_clipping_acota_la_actualizacion():
    global_w = _weights(0.0)
    big = _weights(100.0)  # actualización gigante
    clipped = robust.clip_update(big, global_w, clip_norm=1.0)
    delta = [c - g for c, g in zip(clipped, global_w)]
    norm = np.sqrt(sum(float((d ** 2).sum()) for d in delta))
    assert norm <= 1.0 + 1e-5


def test_ruido_dp_cambia_pesos_pero_mantiene_forma():
    w = _weights(1.0)
    noisy = robust.add_gaussian_noise(w, sigma=0.5, rng=np.random.default_rng(0))
    assert noisy[0].shape == w[0].shape
    assert not np.allclose(noisy[0], w[0])  # el ruido efectivamente cambió los pesos


def test_robust_aggregate_sin_defensa_equivale_a_fedavg():
    global_w = _weights(0.0)
    a, b = _weights(2.0), _weights(4.0)
    out = robust.robust_aggregate(
        [a, b], global_w, num_examples=[1, 1],
        method="fedavg", clip_norm=0.0, dp_sigma=0.0,
    )
    assert np.allclose(out[0], 3.0)
