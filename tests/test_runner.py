"""The shared runner: checkpoints every 5k named by iterations done, and the
wandb run's identity."""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

import sendg1.tasks  # noqa: F401
from sendg1.common.runner import Sendg1Runner
from sendg1.tasks._register import task_info

TASKS = sorted(t for t in list_tasks() if t.startswith(("G1-", "Sendg1-")))


@pytest.mark.parametrize("task", TASKS)
def test_every_task_has_the_runner_and_an_identity(task: str) -> None:
  runner = load_runner_cls(task)
  assert runner is not None and issubclass(runner, Sendg1Runner), task
  info = task_info(load_rl_cfg(task).experiment_name)
  assert info is not None and info.task_id == task.removesuffix("-Eval"), task
  terms = {
    t
    for g in load_rl_cfg(task).obs_groups["actor"]
    for t in load_env_cfg(task).observations[g].terms
  }
  assert info.condition == ("tactile" if "tactile_touch" in terms else "blind"), task


def test_blind_and_tactile_share_a_family() -> None:
  fams: dict[str, set[str]] = {}
  for t in TASKS:
    info = task_info(load_rl_cfg(t).experiment_name)
    assert info is not None
    fams.setdefault(info.family, set()).add(info.condition)
  # The Pedalo Classic has a tactile task only.
  assert {f for f, c in fams.items() if c != {"blind", "tactile"}} == {"pedalo_classic"}


class _Stub(Sendg1Runner):
  """Only the save bookkeeping: no env, no algorithm."""

  def __init__(self, every: int) -> None:  # noqa: D107 -- bypasses the runner's init
    self._save_every, self._last_save_it = every, None
    self.current_learning_iteration = 0
    self.saved: list[tuple[str, int]] = []

  def checkpoint_infos(self) -> dict:
    return {}


def _record(self, path, infos=None):
  self.saved.append((path.rsplit("/", 1)[-1], self.current_learning_iteration))


def test_checkpoints_every_5k_named_by_iterations_done(monkeypatch) -> None:
  from mjlab.rl import MjlabOnPolicyRunner

  monkeypatch.setattr(MjlabOnPolicyRunner, "save", _record)
  r = _Stub(every=5000)
  for it in range(12_345):  # what rsl_rl's loop does with save_interval 1
    r.current_learning_iteration = it
    r.save(f"/x/model_{it}.pt")
  r.save(f"/x/model_{r.current_learning_iteration}.pt")  # the loop's final save
  assert r.saved == [
    ("model_5000.pt", 5000),
    ("model_10000.pt", 10000),
    ("model_12345.pt", 12345),
  ]
  assert r.current_learning_iteration == 12_344


def test_a_run_on_the_grid_ends_in_one_last_checkpoint(monkeypatch) -> None:
  from mjlab.rl import MjlabOnPolicyRunner

  monkeypatch.setattr(MjlabOnPolicyRunner, "save", _record)
  r = _Stub(every=5000)
  for it in range(10_000, 20_000):  # resumed from model_10000.pt
    r.current_learning_iteration = it
    r.save(f"/x/model_{it}.pt")
  r.save(f"/x/model_{r.current_learning_iteration}.pt")
  assert [n for n, _ in r.saved] == ["model_15000.pt", "model_20000.pt", "model_20000.pt"]


def test_explicit_paths_are_saved_as_given(monkeypatch) -> None:
  from mjlab.rl import MjlabOnPolicyRunner

  monkeypatch.setattr(MjlabOnPolicyRunner, "save", _record)
  r = _Stub(every=5000)
  r.current_learning_iteration = 7
  r.save("/x/m.pt")
  assert r.saved == [("m.pt", 7)]


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_training_writes_named_checkpoints_and_resumes(tmp_path) -> None:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper

  task = "G1-RollingCylinder-Blind"
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 8
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    rl = asdict(load_rl_cfg(task))
    rl["logger"], rl["save_interval"], rl["num_steps_per_env"] = "tensorboard", 2, 4
    runner = load_runner_cls(task)(RslRlVecEnvWrapper(env), rl, str(tmp_path), "cuda:0")
    runner.learn(num_learning_iterations=5)
    assert sorted(p.name for p in tmp_path.glob("model_*.pt")) == [
      "model_2.pt", "model_4.pt", "model_5.pt"
    ]
    assert torch.load(tmp_path / "model_5.pt", weights_only=False)["iter"] == 5
    runner.load(str(tmp_path / "model_4.pt"))
    assert runner.current_learning_iteration == 4
  finally:
    env.close()
