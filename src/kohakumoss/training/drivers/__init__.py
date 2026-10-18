"""Training drivers (own the loop), selected through DRIVERS."""

from kohakumoss.training.drivers.common import Parts, grouped, no_accumulate
from kohakumoss.training.drivers.lightning import LightningDriver
from kohakumoss.training.drivers.native import NativeDriver
from kohakumoss_common.registry import Registry

DRIVERS = Registry("driver")
DRIVERS.add("native", NativeDriver)
DRIVERS.add("lightning", LightningDriver)

__all__ = [
    "DRIVERS",
    "LightningDriver",
    "NativeDriver",
    "Parts",
    "grouped",
    "no_accumulate",
]
