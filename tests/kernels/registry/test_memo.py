"""`TensorMemo`: a value built once per live tensor, key and in-place version, dropped
with its tensor; `dtypes.short`: the dtype names the stored problem keys spell."""

import gc

import pytest
import torch

from kohakumoss_kernels import dtypes
from kohakumoss_kernels.backends._memo import TensorMemo
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def check_memo(memo) -> None:
    builds = []

    def build():
        builds.append(1)
        return len(builds)

    t = torch.zeros(4)
    assert memo.get(t, "a", build) == memo.get(t, "a", build) == 1
    assert memo.get(t, "b", build) == 2, "another key shares the value"
    t.add_(1)
    assert memo.get(t, "a", build) == 3, "an in-place change kept the old value"
    with torch.inference_mode():
        frozen = torch.zeros(2)
        first = memo.get(frozen, "a", build)
        frozen.add_(1)
        again = memo.get(frozen, "a", build)
    assert again != first, "an inference tensor refilled in place kept its value"
    del t, frozen
    gc.collect()
    assert not memo.rows, "entries outlived their tensors"


def test_memo():
    check_memo(TensorMemo())


class _Unversioned(TensorMemo):
    """The broken variant: the version is not checked."""

    def get(self, t, key, build):
        ident = (id(t), key)
        if ident not in self.rows:
            self.rows[ident] = (None, 0, build())
        return self.rows[ident][2]


def test_negative_unversioned_memo():
    must_fail(lambda: check_memo(_Unversioned()), "version ignored")


class _InferenceCached(TensorMemo):
    """The broken variant: an inference tensor is memoized at version 0."""

    def get(self, t, key, build):
        if not t.is_inference():
            return super().get(t, key, build)
        ident = (id(t), key)
        if ident not in self.rows:
            self.rows[ident] = (None, 0, build())
        return self.rows[ident][2]


def test_negative_inference_tensor_memoized():
    must_fail(lambda: check_memo(_InferenceCached()), "inference tensor memoized")


def check_short(short) -> None:
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int8):
        assert short(dtype) == str(dtype)[len("torch.") :], dtype


def test_short_names_match_the_stored_keys():
    check_short(dtypes.short)
    assert dtypes.BY_NAME["bf16"] is torch.bfloat16
    assert dtypes.NAME_OF[torch.float16] == "fp16"


def test_negative_full_dtype_name():
    must_fail(lambda: check_short(str), "torch. prefix kept")
