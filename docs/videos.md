# Evaluation videos

Every policy video goes through `sendg1.video.EvalVideo` (`record_video.py`,
`beam_video.py`, `box_push_video.py`, `evaluate_box_push.py`), so clips of
paired policies look the same:

- **Theme** (`--theme`): `studio` (default; dark grid floor fading into a dark
  sky, terrain in greys), `gray`, or `mujoco` (mjlab's stock look). The theme
  only touches the renderer's private model copy; physics and sensors never see it.
- **Caption**, top left: a badge (`TACTILE`, `BLIND`, `CONTACT ORACLE`, read
  off the actor's observation terms), the task, and the script's lines.
  `caption=False` drops it; `logo=` (`IAS_LOGO`, `FLEXOO_LOGO`, or a list,
  from `sendg1.video.LOGOS`) draws those logos there instead, each on a white
  plate.
- **Tactile feet card**, bottom right: both soles from above, toe up, with
  per-taxel normal force, the centre of pressure, per-foot load and a 3 s trace
  of the load on a newton axis, with the robot's body weight as a dashed reference.
  It is live when the actor reads the tactile sole and all zero otherwise:
  a blind policy gets nothing from it, and the clip shows that.
  `sensor_brand=FLEXOO_LOGO` adds a "sensor by <logo>" row under the trace.

A new video script needs `prepare_video_cfg(env_cfg, w, h)` before the env is
built, `EvalVideo(env, rl_cfg, title=...)` after, `video.capture(lines)` per
step and `video.save(path)`.
