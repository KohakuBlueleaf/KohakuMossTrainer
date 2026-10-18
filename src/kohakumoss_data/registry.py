"""Name -> factory registries for the data package (sources, planners, captions)."""

from kohakumoss_common.registry import Registry

SOURCES = Registry("source")
PLANNERS = Registry("planner")
CAPTIONS = Registry("caption template")
