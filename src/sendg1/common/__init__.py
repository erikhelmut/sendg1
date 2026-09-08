"""Shared library: robot, observation, reward, event and sensor definitions.

Everything here is task-agnostic. Task packages under ``sendg1.tasks`` compose
these pieces; they do not redefine them. This is what keeps "add task N+1" down
to one directory and one registration line.
"""
