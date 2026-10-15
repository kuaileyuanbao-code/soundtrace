"""Lifecycle policy checks; real GPU and output comparisons live in validation/ separately."""
import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from model_lifecycle import FAST_COMPONENTS, GIB, fast_state, move_fast, memory_budget, tensor_bytes, cpu_move_groups
import server


class Tensor:
    def __init__(self, device="cuda:0"):
        self.device = device
    def numel(self):
        return 10
    def element_size(self):
        return 4


class Module:
    def __init__(self):
        self.weight = Tensor()
        self.buffer = Tensor()
    def parameters(self):
        return iter([self.weight])
    def buffers(self):
        return iter([self.buffer])
    def to(self, device):
        self.weight.device = self.buffer.device = device
        return self


def owner():
    modules = {name: Module() for name in FAST_COMPONENTS}
    return SimpleNamespace(**modules, kwargs={"device": "cuda:0", "batch_size_s": 60},
                           _base_kwargs_map={"kwargs": {"device": "cuda:0", "batch_size_s": 60}})


class LifecycleTests(unittest.TestCase):
    def test_grouped_cpu_moves_bound_children_and_preserve_values(self):
        import torch
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
        original = {k: v.clone() for k, v in model.state_dict().items()}
        groups = cpu_move_groups(model, limit=300)
        self.assertLessEqual(max(size for _, size in groups), 300)
        self.assertIs(groups[-1][0], model)
        self.assertEqual(groups[-1][1], 0)
        for group, _ in groups:
            group.cpu()
        self.assertTrue(all(torch.equal(original[k], v) for k, v in model.state_dict().items()))

    def test_fast_parking_moves_all_components_and_preserves_configuration(self):
        model = owner()
        config = copy.deepcopy((model.kwargs, model._base_kwargs_map))
        move_fast(model, "cpu")
        self.assertEqual({s["state"] for s in fast_state(model).values()}, {"cpu_loaded"})
        move_fast(model, model.kwargs["device"])
        self.assertEqual({s["state"] for s in fast_state(model).values()}, {"gpu_resident"})
        self.assertEqual((model.kwargs, model._base_kwargs_map), config)
        self.assertEqual(tensor_bytes([model.model, model.model]), 80)

    def test_cuda_buffer_left_after_cpu_move_is_rejected(self):
        model = owner()
        model.spk_model.to = lambda device: None
        with self.assertRaisesRegex(RuntimeError, "未完整释放"):
            move_fast(model, "cpu")

    def test_fast_preparation_parks_live_main_once_and_keeps_ready_state(self):
        with patch.object(server, "memory_budget", return_value={"can_retain": True}), \
             patch.object(server, "stop_moss_worker"), patch.object(server, "stop_diarizen_worker"), \
             patch.object(server, "cache_note"), patch.object(server._main_engine, "ready", True), \
             patch.object(server._main_engine, "state", return_value={"models": {"qwen": {"devices": ["cuda:0"]}}}), \
             patch.object(server._main_engine, "park") as park, patch.object(server._main_engine, "stop") as stop:
            server.prepare_fast_mode()
            park.assert_called_once()
            stop.assert_not_called()
        with patch.object(server, "memory_budget", return_value={"can_retain": True}), \
             patch.object(server, "stop_moss_worker"), patch.object(server, "stop_diarizen_worker"), \
             patch.object(server._main_engine, "ready", True), \
             patch.object(server._main_engine, "state", return_value={"models": {"qwen": {"devices": ["cpu"]}}}), \
             patch.object(server._main_engine, "park") as park:
            server.prepare_fast_mode()
            park.assert_not_called()

    def test_low_memory_or_park_error_releases_main_without_changing_fast_route(self):
        for enough in (False, True):
            with self.subTest(enough=enough), patch.object(server, "memory_budget", return_value={"can_retain": enough}), \
                 patch.object(server, "stop_moss_worker"), patch.object(server, "stop_diarizen_worker"), \
                 patch.object(server, "cache_note"), patch.object(server._main_engine, "ready", True), \
                 patch.object(server._main_engine, "state", return_value={"models": {"qwen": {"devices": ["cuda:0"]}}}), \
                 patch.object(server._main_engine, "park", side_effect=MemoryError("CPU allocation")), \
                 patch.object(server._main_engine, "stop") as stop:
                server.prepare_fast_mode()
                stop.assert_called_once()
                self.assertEqual(server._active_mode, "fast")

    def test_main_entry_keeps_fast_on_cpu_and_review_releases_it(self):
        for preserve in (True, False):
            with self.subTest(preserve=preserve), patch.object(server, "_model", owner()), \
                 patch.object(server, "_speaker_model", None), patch.object(server, "cache_note"), \
                 patch.object(server, "memory_budget", return_value={"can_retain": True}), \
                 patch.object(server, "park_fast_model") as park, patch.object(server, "unload_fast_model") as release:
                server.release_local_models(preserve_fast=preserve)
                self.assertEqual(park.call_count, int(preserve))
                self.assertEqual(release.call_count, int(not preserve))

    def test_idle_memory_guard_only_releases_inactive_owner(self):
        parked = owner()
        move_fast(parked, "cpu")
        with patch.object(server, "memory_budget", return_value={"can_retain": False}), \
             patch.object(server, "cache_note"), patch.object(server, "_model", parked), \
             patch.object(server._main_engine, "ready", True), \
             patch.object(server._main_engine, "state", return_value={"models": {"qwen": {"state": "cpu_loaded"}}}), \
             patch.object(server._main_engine, "stop") as main_release, \
             patch.object(server, "unload_fast_model") as fast_release:
            with patch.object(server, "_active_mode", "fast"):
                server.discard_inactive_cache()
                main_release.assert_called_once()
                fast_release.assert_not_called()
            with patch.object(server, "_active_mode", "main"):
                server.discard_inactive_cache()
                self.assertIsNone(server._model)
                fast_release.assert_not_called()  # CPU-only eviction doesn't invoke CUDA cleanup.

    def test_pressure_monitor_never_evicts_gpu_owner_or_a_locked_constructor(self):
        with patch.object(server, "memory_budget", return_value={"can_retain": False}), \
             patch.object(server, "_active_mode", "main"), patch.object(server, "_model", owner()):
            retained = server._model
            server.discard_inactive_cache()
            self.assertIs(server._model, retained)
            move_fast(retained, "cpu")
            with server._model_lock:
                server.discard_inactive_cache()
                self.assertIs(server._model, retained)

    def test_memory_budget_and_disable_switch_are_observed(self):
        with patch.dict(os.environ, {"SOUNDTRACE_MODEL_REUSE": "0"}):
            self.assertFalse(memory_budget()["can_retain"])
        # Deterministically emulate low free memory without allocating RAM.
        import psutil
        with patch.object(psutil, "virtual_memory", return_value=SimpleNamespace(available=GIB, total=32 * GIB)), \
             patch.dict(os.environ, {"SOUNDTRACE_MODEL_REUSE": "1", "SOUNDTRACE_CACHE_MIN_FREE_GB": "4"}):
            self.assertFalse(memory_budget()["can_retain"])


if __name__ == "__main__":
    unittest.main()
