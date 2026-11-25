"""The runner every sendg1 task trains with: checkpoints and wandb runs.

Checkpoints. rsl_rl saves ``model_<it>.pt`` with ``it`` the 0-based index of
the iteration just run, so a 20k run ends in ``model_19999.pt``, and a resume
from it runs iteration 19999 a second time. Here a checkpoint is named, and
stamped, with the number of iterations done: ``model_5000.pt`` ...
``model_20000.pt``, every ``SAVE_EVERY`` (``sendg1.common.ppo``), plus the
last iteration if a run stops off that grid. A resume continues at the next
iteration.

wandb. rsl_rl names a run after its log directory and nothing else. Here each
run also carries its task, its condition (blind / tactile), its seed and the
git commit, is grouped by task family (``group``) with the condition as
``job_type``, and is tagged with all four, so the paired runs of a study can
be filtered and grouped in the wandb UI. What is logged per iteration is
rsl_rl's: every reward term (``Episode_Reward/*``), terminations
(``Episode_Termination/*``), curriculum and metrics terms, losses, learning
rate, action std and throughput.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from mjlab.rl import MjlabOnPolicyRunner
from rsl_rl.utils.wandb_log_writer import WandbLogWriter
from torch.utils.tensorboard import SummaryWriter

_LOOP_CHECKPOINT = re.compile(r"^model_\d+\.pt$")
"""The file name rsl_rl's training loop saves under; other paths are saved as given."""
_REPO = Path(__file__).resolve().parents[3]


def _git_state() -> dict[str, str | bool]:
  try:
    sha = subprocess.run(
      ["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
      capture_output=True, text=True, check=True,
    ).stdout.strip()
    dirty = bool(
      subprocess.run(
        ["git", "-C", str(_REPO), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True,
      ).stdout.strip()
    )
  except (OSError, subprocess.CalledProcessError):
    return {}
  return {"git_commit": sha, "git_dirty": dirty}


class Sendg1WandbWriter(WandbLogWriter):
  """rsl_rl's wandb writer, with the run's group, job type, tags and config."""

  def __init__(
    self,
    log_dir: str,
    project_name: str,
    group: str | None = None,
    job_type: str | None = None,
    tags: tuple[str, ...] = (),
    config: dict | None = None,
  ) -> None:
    import wandb

    # WandbLogWriter.__init__ would start the run without these, so it is
    # skipped; this is everything else it does.
    SummaryWriter.__init__(self, log_dir, flush_secs=10)
    # Tags given on the command line (--agent.wandb-tags) sit in wandb's
    # settings until a run starts; keep them.
    cli_tags = tuple(wandb.setup().settings.run_tags or ())
    wandb.init(
      project=project_name,
      entity=os.environ.get("WANDB_USERNAME"),
      name=os.path.basename(log_dir),
      group=group,
      job_type=job_type,
      tags=sorted({*tags, *cli_tags}),
      config={"log_dir": log_dir, **(config or {})},
    )


class Sendg1Runner(MjlabOnPolicyRunner):
  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
    super().__init__(env, train_cfg, log_dir, device)
    # rsl_rl's loop offers a save after every iteration; ``save`` keeps those
    # on the SAVE_EVERY grid (counted in iterations done) and the last one.
    self._save_every = int(self.cfg["save_interval"])
    self.cfg["save_interval"] = 1
    self._last_save_it: int | None = None
    logger = self.cfg.get("logger")
    if isinstance(logger, dict) and logger.get("class_name") == "WandbLogWriter":
      self.cfg["logger"] = self._wandb_logger_cfg(logger["project_name"])

  # -- checkpoints ------------------------------------------------------------

  def checkpoint_infos(self) -> dict:
    """Extra state a task keeps in its checkpoints (override)."""
    return {}

  def save(self, path: str, infos=None) -> None:
    infos = {**self.checkpoint_infos(), **(infos or {})}
    if not _LOOP_CHECKPOINT.match(os.path.basename(path)):
      super().save(path, infos)
      return
    it = self.current_learning_iteration
    # The loop's final save repeats the last iteration's: always keep that one.
    final = it == self._last_save_it
    self._last_save_it = it
    done = it + 1
    if not final and done % self._save_every:
      return
    self.current_learning_iteration = done  # the checkpoint's "iter": resume here
    try:
      super().save(os.path.join(os.path.dirname(path), f"model_{done}.pt"), infos)
    finally:
      self.current_learning_iteration = it

  def load(self, path: str, load_cfg=None, strict: bool = True, map_location=None) -> dict:
    infos = super().load(path, load_cfg, strict, map_location)
    self._last_save_it = None
    return infos

  # -- wandb ------------------------------------------------------------------

  def _wandb_logger_cfg(self, project: str) -> dict:
    from sendg1.tasks._register import task_info

    info = task_info(self.cfg["experiment_name"])
    seed = int(self.cfg["seed"])
    group = info.family if info else self.cfg["experiment_name"]
    condition = info.condition if info else None
    task = info.task_id if info else None
    return {
      "class_name": f"{__name__}:Sendg1WandbWriter",
      "project_name": project,
      "group": group,
      "job_type": condition,
      "tags": tuple(t for t in (group, condition, f"seed{seed}", task) if t),
      "config": {
        "task": task,
        "family": group,
        "condition": condition,
        "seed": seed,
        "num_envs": self.env.num_envs,
        "max_iterations": self.cfg.get("max_iterations"),
        **_git_state(),
      },
    }
