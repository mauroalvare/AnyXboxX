"""Parity of scalar f16 compares (`v_cmp[x]_*_f16`, `v_cmp[x]_class_f16`) between Rust and Python."""
import random

from emu.rdna2_exec_cmpf16 import COMPAREF16
from tests.test_rdna2_ref_parity import (
    EXT_CODES,
    LITERAL,
    OPCODES,
    OPERANDS,
    SEED,
    V0,
    assert_same,
    binary,
    extension_word,
    program,
    run_python,
    sop1,
    vop1,
    vop2,
)

SPECIAL_HALVES = (
    0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x0400, 0x8400,
    0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x7BFF, 0xFBFF,
    0x7C00, 0xFC00, 0x7C01, 0xFC01, 0x7E00, 0xFE00, 0xFFFF,
)
CMPF16_NAMES = tuple(sorted(COMPAREF16))
CMPF16_SEEDED_PROGRAMS = 250


def vopc_f16(name, src0, vsrc1):
    return 0x3E << 25 | OPCODES["VOPC", name] << 17 | vsrc1 << 9 | src0


def vop3_cmp_f16(name, sdst, sources, *, abs_bits=0, op_sel=0, clamp=0, omod=0, neg_bits=0, literal=None):
    opcode = OPCODES["VOP3", name]
    s0, s1 = sources
    low = 0x35 << 26 | opcode << 16 | sdst | abs_bits << 8 | op_sel << 11 | clamp << 15
    high = s0 | s1 << 9 | 128 << 18 | omod << 27 | neg_bits << 29
    return [low, high] if literal is None else [low, high, literal]


def test_all_thirty_four_f16_compares_in_vopc_and_vop3_match_the_python_model():
    exe = binary()
    # Populate v1 and v2 per lane from SPECIAL_HALVES with noisy high halves so high-half ignoring is tested.
    setup = []
    for lane_mod, reg in ((0, 1), (7, 2)):
        for idx, half in enumerate(SPECIAL_HALVES[:8]):
            noise = (0xA500 + idx * 0x111) << 16
            setup += [
                vop2("v_and_b32", 3, 128 + 7, 0),
                0x3E << 25 | OPCODES["VOPC", "v_cmp_eq_u32"] << 17 | 3 << 9 | (128 + ((idx + lane_mod) & 7)),
                vop2("v_cndmask_b32", reg, V0 + reg, 3),  # keep varying bits
                vop1("v_mov_b32", 4, LITERAL), noise | half,
                vop2("v_cndmask_b32", reg, V0 + reg, 4),
            ]
    for lanes in (32, 64):
        for name in CMPF16_NAMES:
            # VOPC (writes VCC or EXEC) + VOP3 (writes SGPR pair or EXEC, with abs/neg on S0).
            vopc_prog = program(
                *setup,
                vopc_f16(name, V0 + 1, 2),
                *vop3_cmp_f16(name, 20, (242, V0 + 1), abs_bits=1, neg_bits=0 if "class" in name else 0b10),
            )
            scalars = [(126, 0xFFFF_55AA), (127, 0xAA55_FFFF)]
            assert run_python(lanes, False, scalars, vopc_prog).startswith("steps")
            assert_same(exe, lanes, False, scalars, vopc_prog)

    # Shared refusals in VOPC (SDWA/DPP) and VOP3 (op_sel, clamp, omod, abs/neg on class mask S1).
    refusals = [
        program(vopc_f16("v_cmp_eq_f16", 0xF9, 1), 0x0606_0600),
        program(vopc_f16("v_cmp_class_f16", 0xFA, 1), 0),
        program(*vop3_cmp_f16("v_cmp_lt_f16", 10, (V0 + 1, V0 + 2), op_sel=1)),
        program(*vop3_cmp_f16("v_cmpx_ge_f16", 0, (V0 + 1, V0 + 2), op_sel=0b1000)),
        program(*vop3_cmp_f16("v_cmp_eq_f16", 10, (V0 + 1, V0 + 2), clamp=1)),
        program(*vop3_cmp_f16("v_cmp_eq_f16", 10, (V0 + 1, V0 + 2), omod=2)),
        program(*vop3_cmp_f16("v_cmp_class_f16", 10, (V0 + 1, V0 + 2), abs_bits=0b10)),
        program(*vop3_cmp_f16("v_cmpx_class_f16", 0, (V0 + 1, LITERAL), neg_bits=0b10, literal=0x03FF)),
    ]
    for bad in refusals:
        assert run_python(32, False, [], bad) == "error untranslated\n"
        assert_same(exe, 32, False, [], bad)


def test_v_cmp_class_f16_matches_the_python_model_on_every_binary16_input():
    """All 65,536 binary16 inputs tested against each of the 10 IEEE class bits."""
    exe = binary()
    for start in range(0, 0x10000, 64 * 16):
        words = []
        dst_sgpr = 10
        for step_idx in range(16):
            base = start + step_idx * 64
            words += [vop2("v_or_b32", 1, LITERAL, 0), base]
            for bit_idx in range(10):
                if dst_sgpr <= 102:
                    words += vop3_cmp_f16(
                        "v_cmp_class_f16", dst_sgpr, (V0 + 1, LITERAL), literal=1 << bit_idx
                    )
                    dst_sgpr += 2
            if dst_sgpr > 102:
                dst_sgpr = 10
        assert_same(exe, 64, False, [], program(*words))


def random_cmpf16_instruction(rng):
    name = rng.choice(CMPF16_NAMES)
    if rng.random() < 0.5:
        if rng.random() < 0.08:
            ext = rng.choice(EXT_CODES)
            return [vopc_f16(name, ext, rng.randrange(1, 8)), extension_word(rng, ext, 1)]
        src0 = rng.choice((*OPERANDS, *range(256, 264)))
        word = vopc_f16(name, src0, rng.randrange(1, 8))
        if src0 == LITERAL:
            return [word, rng.choice(SPECIAL_HALVES) << 16 | rng.choice(SPECIAL_HALVES)]
        return [word]
    s0 = rng.choice((*OPERANDS, *range(256, 264)))
    s1 = rng.choice((*OPERANDS, *range(256, 264)))
    plain = rng.random() < 0.8
    sdst = rng.randrange(0, 104, 2) if plain else rng.randrange(128)
    abs_bits = (rng.randrange(2) if "class" in name else rng.randrange(4)) if plain else rng.randrange(8)
    neg_bits = (rng.randrange(2) if "class" in name else rng.randrange(4)) if plain else rng.randrange(8)
    op_sel = 0 if plain else rng.randrange(16)
    clamp = 0 if plain else rng.randrange(2)
    omod = 0 if plain else rng.randrange(4)
    lit = (rng.choice(SPECIAL_HALVES) << 16 | rng.choice(SPECIAL_HALVES)) if LITERAL in (s0, s1) else None
    return vop3_cmp_f16(
        name, sdst, (s0, s1), abs_bits=abs_bits, op_sel=op_sel,
        clamp=clamp, omod=omod, neg_bits=neg_bits, literal=lit,
    )


def test_seeded_f16_compare_programs_match_the_python_model():
    exe = binary()
    rng = random.Random(SEED ^ 0xC16F)
    outcomes = {"steps": 0, "error": 0}
    for _ in range(CMPF16_SEEDED_PROGRAMS):
        lanes = rng.choice((32, 64))
        scc = rng.random() < 0.5
        scalars = [(i, rng.choice(SPECIAL_HALVES) << 16 | rng.choice(SPECIAL_HALVES)) for i in range(8)]
        if rng.random() < 0.4:
            scalars += [(126, rng.getrandbits(32)), (127, rng.getrandbits(32))]
        init = [sop1("s_mov_b32", 8, LITERAL), 0x03FF]
        for reg in range(1, 8):
            val = rng.choice(SPECIAL_HALVES) << 16 | rng.choice(SPECIAL_HALVES)
            init += [vop1("v_mov_b32", reg, LITERAL), val]
        body = [w for _ in range(rng.randrange(1, 6)) for w in random_cmpf16_instruction(rng)]
        words = program(*init, *body)
        outcomes[run_python(lanes, scc, scalars, words).split()[0]] += 1
        assert_same(exe, lanes, scc, scalars, words)
    assert outcomes["steps"] >= 140 and outcomes["error"] >= 20, outcomes
