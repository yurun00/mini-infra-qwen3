"""Unit tests for device synchronization and step latency utilities."""

import time
from unittest.mock import MagicMock, patch

import torch

from mini_infra.device import PendingStepLatencies, StepTimer, synchronize


def test_synchronize_cpu():
    cpu_device = torch.device("cpu")
    # Should be a no-op and not raise
    synchronize(cpu_device)


def test_step_timer_cpu():
    device = torch.device("cpu")
    with StepTimer(device) as timer:
        time.sleep(0.005)

    assert timer.elapsed_ms > 0.0


def test_step_timer_cuda_mock():
    cuda_device = torch.device("cuda:0")

    mock_start = MagicMock()
    mock_end = MagicMock()
    mock_start.elapsed_time.return_value = 12.34

    with (
        patch("torch.cuda.synchronize") as mock_sync,
        patch("torch.cuda.Event", side_effect=[mock_start, mock_end]),
    ):
        with StepTimer(cuda_device) as timer:
            pass

        assert timer.elapsed_ms == 12.34
        assert mock_start.record.called
        assert mock_end.record.called
        assert mock_sync.call_count == 2


def test_pending_step_latencies_cpu():
    device = torch.device("cpu")
    collector = PendingStepLatencies(device)

    # Empty settle
    assert collector.settle() == []

    # Record 3 steps
    for _ in range(3):
        with collector.measure_step():
            time.sleep(0.002)

    latencies = collector.settle()
    assert len(latencies) == 3
    assert all(lat > 0.0 for lat in latencies)

    # Internal list is cleared
    assert collector.settle() == []


def test_pending_step_latencies_cuda_mock():
    cuda_device = torch.device("cuda:0")
    collector = PendingStepLatencies(cuda_device)

    # Empty settle
    assert collector.settle() == []

    events = []
    for _ in range(2):
        s = MagicMock()
        e = MagicMock()
        s.elapsed_time.return_value = 5.67
        events.extend([s, e])

    with (
        patch("torch.cuda.synchronize") as mock_sync,
        patch("torch.cuda.Event", side_effect=events),
    ):
        for _ in range(2):
            with collector.measure_step():
                pass

        latencies = collector.settle()
        assert latencies == [5.67, 5.67]
        mock_sync.assert_called_once_with(cuda_device)
        assert collector.settle() == []

