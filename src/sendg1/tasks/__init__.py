"""Task registration entry point.

mjlab discovers this module through the ``mjlab.tasks`` entry point declared in
pyproject.toml, and imports it when ``mjlab`` itself is imported. Importing a
task package registers its variants as a side effect, which is what makes
mjlab's own ``train`` / ``play`` / ``list-envs`` CLIs see them.

Adding task N+1: create the directory, add one import line here.
"""

import sendg1.tasks.velocity  # noqa: F401

from sendg1.tasks._register import variants  # noqa: E402

__all__ = ["variants"]
