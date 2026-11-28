"""Independent finite binary16 FMA regression tests (RNE, gradual underflow).

Synthetic operands only. The oracle uses exact signed integers, not the
interpreter's or shader's floating-point helpers. Developed with ChatGPT
assistance; based on the independently exercised issue #115 FMA probe.
"""
import bisect
import random
import struct

import pytest
from emu import rdna2_exec_f16, rdna2_gpu, rdna2_hlsl_f16

COUNT = 65536


def _units24(h):
    """Finite binary16 expressed in integer multiples of 2**-24."""
    e, f = (h >> 10) & 31, h & 1023
    assert e < 31
    magnitude = ((1024 + f) << (e - 1)) if e else f
    return -magnitude if h & 0x8000 else magnitude


_GRID = [_units24(h) << 24 for h in range(0x7C00)]


def _exact_fma(a, b, c):
    product = _units24(a) * _units24(b)
    n = product + (_units24(c) << 24)
    if n == 0:
        return 0x8000 if product == 0 and (a ^ b) & 0x8000 and c == 0x8000 else 0
    sign = 0x8000 if n < 0 else 0
    n = abs(n)
    if n >= 65520 << 48:
        return sign | 0x7C00
    hi = bisect.bisect_left(_GRID, n)
    if hi == len(_GRID):
        return sign | 0x7BFF
    if _GRID[hi] == n:
        return sign | hi
    lo = hi - 1
    dl, dh = n - _GRID[lo], _GRID[hi] - n
    return sign | (hi if dh < dl or (dh == dl and hi % 2 == 0) else lo)


def _cases(count):
    base = [(0x3C01, 0x3E00, 0x8001), (0x3C03, 0x3E00, 0x0001),
            (0x6401, 0x3E00, 0x8400), (0x6403, 0x3E00, 0x0400)]
    out = list(base)
    out += [(a ^ 0x8000, b, c ^ 0x8000) for a, b, c in base]
    out += [(b, a, c) for a, b, c in base]
    out += [(0x0001, 0x3800, 0), (0x8001, 0x3800, 0),
            (0x0003, 0x3800, 0), (0x8003, 0x3800, 0),
            (0x7BFF, 0x4000, 0), (0xFBFF, 0x4000, 0),
            (0x7BFF, 0x3C00, 0x4C00), (0x7BFF, 0x3C00, 0x4BFF),
            (0x3C00, 0x3C00, 0xBC00), (0xBC00, 0x3C00, 0x3C00),
            (0x8000, 0x3C00, 0x8000), (0, 0xBC00, 0x8000),
            (0x8000, 0x3C00, 0), (0, 0x3C00, 0x8000),
            (0x0400, 0x3800, 0), (0x03FF, 0x3C00, 1),
            (0x7BFF, 0x7BFF, 0xFBFF), (0x0001, 0x0001, 0),
            (0x3C01, 0x3C01, 0xBC00), (0x3C00, 0x3C00, 0x1000)]
    rng = random.Random(115)

    def finite():
        return rng.randrange(0x7C00) | (rng.randrange(2) << 15)

    while len(out) < count:
        if len(out) % 2:
            out.append((finite(), finite(), finite()))
        else:
            a = 0x3C00 | (rng.randrange(512) * 2 + 1)
            c = rng.randrange(1, 8) | (rng.randrange(2) << 15)
            out.append((a, 0x3E00, c))
    return out[:count]


@pytest.fixture(scope="module")
def vectors():
    return [(a, b, c, _exact_fma(a, b, c)) for a, b, c in _cases(COUNT)]


def _check(index, row, actual):
    a, b, c, expected = row
    assert actual == expected, (
        f"case {index}: {a:04X} * {b:04X} + {c:04X}, "
        f"actual={actual:04X}, expected={expected:04X}"
    )


def test_independent_f16_fma_interpreter(vectors):
    operation = rdna2_exec_f16.OPERATIONS["v_fma_f16"][1]
    for i, row in enumerate(vectors):
        a, b, c, _ = row
        _check(i, row, operation(a, b, c, 0, 0))


@pytest.mark.skipif(not rdna2_gpu.available(), reason="DXC or anyxboxx_gpu_run.exe unavailable")
def test_independent_f16_fma_gpu(vectors):
    shader = (
        "RWByteAddressBuffer mem : register(u0);\n"
        + rdna2_hlsl_f16.HELPERS
        + """
[numthreads(32, 1, 1)]
void main(uint lane : SV_GroupIndex)
{
    for (uint i = lane; i < CASE_COUNT; i += 32)
    {
        uint p = i * 16;
        uint a = mem.Load(p);
        uint b = mem.Load(p + 4);
        uint c = mem.Load(p + 8);
        mem.Store(p + 12, F16Fma(a, b, c));
    }
}
""".replace("CASE_COUNT", str(len(vectors)))
    )
    memory = b"".join(struct.pack("<4I", a, b, c, 0xDEADBEEF)
                      for a, b, c, _ in vectors)
    output = rdna2_gpu.run(shader, memory)
    assert len(output) == len(memory)
    for i, row in enumerate(vectors):
        actual = struct.unpack_from("<I", output, i * 16 + 12)[0]
        _check(i, row, actual)
