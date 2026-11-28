"""Parity of scalar f16 arithmetic, rounding and conversions between `anyxboxx-rdna2-ref` and `emu/rdna2_exec_f16.py`."""
import random

from emu.rdna2_exec_f16 import OPERATIONS as F16_OPERATIONS
from tests.test_rdna2_ref_parity import (
    EXT_CODES,
    F16_ARITH_NAMES,
    LITERAL,
    OPCODES,
    OPERANDS,
    SEED,
    V0,
    VOP2_WITH_LITERAL,
    assert_same,
    binary,
    extension_word,
    program,
    run_python,
    sop1,
    vop1,
    vop2,
)

UNARY_EXHAUSTIVE = (
    "v_floor_f16",
    "v_ceil_f16",
    "v_trunc_f16",
    "v_rndne_f16",
    "v_fract_f16",
    "v_frexp_mant_f16",
    "v_frexp_exp_i16_f16",
    "v_cvt_f32_f16",
)
SPECIAL_HALVES = (
    0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x0400, 0x8400,
    0x3555, 0x3800, 0xB800, 0x3BFF, 0xBBFF, 0x3C00, 0xBC00, 0x3E00,
    0x4000, 0xC000, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7C01, 0x7E00, 0xFFFF,
)
SPECIAL_WORDS = (
    0x0000_0000, 0x8000_0000, 0x0000_0001, 0x3380_0000, 0x3F80_0000,
    0xBF80_0000, 0x477F_E000, 0x477F_F000, 0xC780_0000, 0x7F80_0000,
    0xFF80_0000, 0x7FC0_0001, 0xFFC0_0042,
)
F16_SEEDED_PROGRAMS = 250


def vop3_f16(name, vdst, sources, *, abs_bits=0, op_sel=0, clamp=0, omod=0, neg_bits=0, literal=None):
    opcode = OPCODES["VOP3", name]
    s0, s1, s2 = (*sources, 128, 128)[:3]
    low = 0x35 << 26 | opcode << 16 | vdst | abs_bits << 8 | op_sel << 11 | clamp << 15
    high = s0 | s1 << 9 | s2 << 18 | omod << 27 | neg_bits << 29
    return [low, high] if literal is None else [low, high, literal]


def test_scalar_f16_fixed_operations_and_refusals_match_the_python_model():
    """Every scalar f16 arithmetic, FMA, rounding and conversion form in VOP1/VOP2/VOP3, plus shared refusals."""
    exe = binary()
    setup = [
        vop1("v_mov_b32", 1, LITERAL), 0xBC00_3E00,   # lo = 1.5, hi = -1.0
        vop1("v_mov_b32", 2, LITERAL), 0x4200_4080,   # lo = 2.25, hi = 3.0
        vop1("v_mov_b32", 3, LITERAL), 0x7E01_3800,   # lo = 0.5, hi = sNaN
        vop1("v_mov_b32", 4, LITERAL), 0xFFFA_0003,   # i16 exponents: +3, -6
        vop1("v_mov_b32", 5, LITERAL), 0x4780_0000,   # 65536.0f32 (overflows f16)
        vop1("v_mov_b32", 6, LITERAL), 0xBF40_0000,   # -0.75f32
        vop1("v_mov_b32", 7, LITERAL), 0xCAFE_3400,   # accumulator with non-zero high half
    ]
    body = [
        vop2("v_add_f16", 8, V0 + 1, 2),
        vop2("v_sub_f16", 9, V0 + 1, 2),
        vop2("v_subrev_f16", 10, V0 + 1, 2),
        vop2("v_mul_f16", 11, 242, 1),                # inline 1.0 (0x3C00)
        vop2("v_fmac_f16", 7, V0 + 1, 2),
        vop2("v_fmamk_f16", 12, V0 + 1, 2), 0xDEAD_4000,
        vop2("v_fmaak_f16", 13, V0 + 1, 2), 0xBEEF_3800,
        *vop3_f16("v_fma_f16", 14, (V0 + 1, V0 + 2, V0 + 3), op_sel=0b1010, neg_bits=0b001),
        *vop3_f16("v_fma_f16", 14, (V0 + 1, V0 + 2, V0 + 3), clamp=1),
        vop2("v_min_f16", 15, LITERAL, 1), 0x0000_7E05,
        vop2("v_max_f16", 16, LITERAL, 1), 0x0000_8000,
        vop2("v_ldexp_f16", 17, V0 + 1, 4),
        *vop3_f16("v_ldexp_f16", 17, (V0 + 1, V0 + 4), op_sel=0b1010, abs_bits=0b001),
        vop2("v_cvt_pkrtz_f16_f32", 18, V0 + 5, 6),
        *vop3_f16("v_cvt_pkrtz_f16_f32", 19, (V0 + 5, V0 + 6), neg_bits=0b01),
        vop1("v_cvt_f16_f32", 20, V0 + 5),
        *vop3_f16("v_cvt_f16_f32", 20, (V0 + 6,), op_sel=0b1000, abs_bits=1),
        vop1("v_cvt_f32_f16", 21, V0 + 3),
        *vop3_f16("v_cvt_f32_f16", 22, (V0 + 3,), op_sel=0b0001),
        *vop3_f16("v_fmamk_f16", 23, (LITERAL, V0 + 2), literal=0x1234_3C00),
        *vop3_f16("v_fmaak_f16", 24, (V0 + 1, LITERAL), literal=0x1234_4000),
    ]
    for idx, name in enumerate(UNARY_EXHAUSTIVE, start=25):
        body.append(vop1(name, idx, V0 + 1))
    for lanes in (32, 64):
        executed = program(*setup, *body)
        assert run_python(lanes, False, [(126, 0x5555_AAAA)], executed).startswith("steps")
        assert_same(exe, lanes, False, [(126, 0x5555_AAAA)], executed)

    # Shared refusals in VOP1/VOP2 and VOP3.
    refusals = [
        program(vop2("v_add_f16", 1, 0xF9, 2), 0x0606_0600),
        program(vop1("v_floor_f16", 1, 0xFA), 0),
        program(*vop3_f16("v_add_f16", 1, (V0 + 1, V0 + 2), omod=1)),
        program(*vop3_f16("v_cvt_f32_f16", 1, (V0 + 1,), clamp=1)),
        program(*vop3_f16("v_cvt_pkrtz_f16_f32", 1, (V0 + 1, V0 + 2), op_sel=0b1000)),
        program(*vop3_f16("v_cvt_f16_f32", 1, (V0 + 1,), op_sel=0b0001)),
        program(*vop3_f16("v_frexp_exp_i16_f16", 1, (V0 + 1,), clamp=1)),
        program(*vop3_f16("v_fmac_f16", 1, (V0 + 1, V0 + 2), op_sel=0b0001)),
        program(*vop3_f16("v_fmamk_f16", 1, (V0 + 1, V0 + 2))),
        program(*vop3_f16("v_fmaak_f16", 1, (V0 + 1, V0 + 2))),
        program(*vop3_f16("v_ldexp_f16", 1, (V0 + 1, V0 + 2), abs_bits=0b010)),
        program(*vop3_f16("v_ldexp_f16", 1, (V0 + 1, 242))),
        program(*vop3_f16("v_add_f16", 1, (242, V0 + 1), op_sel=0b0001)),
        program(*vop3_f16("v_add_f16", 1, (LITERAL, V0 + 1), op_sel=0b0001, literal=0x3C00_3C00)),
    ]
    for bad in refusals:
        assert run_python(32, False, [], bad) == "error untranslated\n"
        assert_same(exe, 32, False, [], bad)


def test_every_unary_f16_rounding_fract_frexp_and_widen_input_matches_the_python_model():
    """All 65,536 binary16 bit patterns across the 8 unary operations, packed into 64-lane waves."""
    exe = binary()
    # In wave64, v0 holds lane 0..63; 32 chunks of 32 VOP2 instructions cover 64 * 1024 = 65,536 inputs.
    for start in range(0, 0x10000, 64 * 32):
        words = []
        for step_idx in range(32):
            base = start + step_idx * 64
            # v1 = base + lane (exact in [0, 65535] since v0 is 0..63 and base is a multiple of 64).
            words += [vop2("v_or_b32", 1, LITERAL, 0), base]
            for dst, name in enumerate(UNARY_EXHAUSTIVE, start=2):
                words.append(vop1(name, dst + (step_idx % 24) * 8, V0 + 1))
        assert_same(exe, 64, False, [], program(*words))


def random_f16_instruction(rng):
    encodings = [
        (family, opcode, name)
        for family, opcode, name in OPCODES_LIST
        if name in F16_ARITH_SET
    ]
    family, opcode, name = rng.choice(encodings)
    kinds, _ = F16_OPERATIONS[name]
    vdst = rng.randrange(1, 16)
    if family in ("VOP1", "VOP2"):
        if rng.random() < 0.08 and not (family == "VOP2" and opcode in VOP2_WITH_LITERAL):
            ext = rng.choice(EXT_CODES)
            word = (0x3F << 25 if family == "VOP1" else opcode << 25) | vdst << 17 | ext
            return [word, extension_word(rng, ext, 0 if family == "VOP2" else None)]
        src0 = rng.choice((*OPERANDS, *range(256, 272)))
        if kinds[0] == "i" and 240 <= src0 <= 248 and rng.random() < 0.8:
            src0 = rng.randrange(256, 272)
        if family == "VOP1":
            word = 0x3F << 25 | vdst << 17 | opcode << 9 | src0
        else:
            word = opcode << 25 | vdst << 17 | rng.randrange(1, 16) << 9 | src0
        if src0 == LITERAL or (family == "VOP2" and opcode in VOP2_WITH_LITERAL):
            lit = rng.choice(SPECIAL_WORDS) if "w" in kinds else (
                rng.choice(SPECIAL_HALVES) << 16 | rng.choice(SPECIAL_HALVES)
            )
            return [word, lit]
        return [word]
    sources = [rng.choice((*OPERANDS, *range(256, 272))) for _ in range(3)]
    plain = rng.random() < 0.75
    if plain:
        for i, kind in enumerate(kinds):
            if kind == "i" and 240 <= sources[i] <= 248:
                sources[i] = rng.randrange(256, 272)
        if name in ("v_fmamk_f16", "v_fmaak_f16") and LITERAL not in sources[:2]:
            sources[0] = LITERAL
    op_sel = 0 if plain else rng.randrange(16)
    if plain and name != "v_fmac_f16" and "w" not in kinds and all(s < 128 or s >= 256 for s in sources[:len(kinds)]):
        op_sel = rng.randrange(16)
    clamp = 0 if (plain and name in ("v_cvt_f32_f16", "v_cvt_pkrtz_f16_f32", "v_frexp_exp_i16_f16")) else rng.randrange(2)
    abs_bits = 0 if (plain and "i" in kinds) else rng.randrange(8)
    neg_bits = 0 if (plain and "i" in kinds) else rng.randrange(8)
    omod = 0 if plain else rng.randrange(4)
    lit = None
    if LITERAL in sources:
        lit = rng.choice(SPECIAL_WORDS) if "w" in kinds else (
            rng.choice(SPECIAL_HALVES) << 16 | rng.choice(SPECIAL_HALVES)
        )
    return vop3_f16(
        name, vdst, sources, abs_bits=abs_bits, op_sel=op_sel,
        clamp=clamp, omod=omod, neg_bits=neg_bits, literal=lit,
    )


F16_ARITH_SET = frozenset(F16_ARITH_NAMES)
OPCODES_LIST = [
    (family, opcode) + (name,)
    for (family, name), opcode in OPCODES.items()
    if family in ("VOP1", "VOP2", "VOP3")
]


def test_seeded_scalar_f16_programs_match_the_python_model():
    exe = binary()
    rng = random.Random(SEED ^ 0xF160)
    outcomes = {"steps": 0, "error": 0}
    for _ in range(F16_SEEDED_PROGRAMS):
        lanes = rng.choice((32, 64))
        scc = rng.random() < 0.5
        scalars = [(i, rng.choice(SPECIAL_HALVES) << 16 | rng.choice(SPECIAL_HALVES)) for i in range(8)]
        if rng.random() < 0.4:
            scalars += [(126, rng.getrandbits(32)), (127, rng.getrandbits(32))]
        init = [sop1("s_mov_b32", 8, LITERAL), rng.choice(SPECIAL_WORDS)]
        for reg in range(1, 6):
            val = rng.choice(SPECIAL_WORDS) if reg == 5 else (
                rng.choice(SPECIAL_HALVES) << 16 | rng.choice(SPECIAL_HALVES)
            )
            init += [vop1("v_mov_b32", reg, LITERAL), val]
        body = [w for _ in range(rng.randrange(1, 7)) for w in random_f16_instruction(rng)]
        words = program(*init, *body)
        outcomes[run_python(lanes, scc, scalars, words).split()[0]] += 1
        assert_same(exe, lanes, scc, scalars, words)
    assert outcomes["steps"] >= 140 and outcomes["error"] >= 20, outcomes
