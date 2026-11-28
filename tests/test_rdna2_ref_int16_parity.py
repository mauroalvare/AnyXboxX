"""Parity of the 16-bit integer VOP3 instructions (`emu/rdna2_exec_int16.py`, `execute_vop3`) between the Python
model and the Rust reference interpreter (`crates/anyxboxx-rdna2-ref/src/int16.rs`).

The forms the model runs are compared in every op_sel and clamp combination it accepts, on 64 lanes of
different source values; the forms it refuses are refused by both, with and without active lanes. Which form
is refused is read from the model itself (`vop3_refusal`), so a change on the Python side is compared too."""
import random

from emu import rdna2_exec_int16
from emu.rdna2_exec_vop3 import Fields
from tests.test_rdna2_ref_parity import (
    LITERAL,
    OPERANDS,
    V0,
    assert_same,
    binary,
    program,
    run_python,
    vop1,
    vop3,
)

NAMES = tuple(sorted(rdna2_exec_int16.VOP3))
SOURCES = (1, 2, 3)                  # v1, v2, v3: the sources, filled lane by lane
FIRST_RESULT = 10                    # v10 and up: one destination per form
OLD_DESTINATION = 0xA5A5_5A5A        # what D holds before, so that the kept half shows
INLINE_LANE = 128                    # inline constant 0; lane n is 128 + n
SPECIAL_HALVES = (0x0000, 0x0001, 0x0002, 0x000F, 0x0010, 0x0011, 0x7FFE, 0x7FFF, 0x8000, 0x8001, 0xFFFE, 0xFFFF)
MODIFIERS = ({"abs_bits": 1}, {"abs_bits": 4}, {"neg": 2}, {"neg": 4}, {"omod": 1}, {"omod": 3})
LANES_OFF = [(126, 0), (127, 0)]
SEEDED_PROGRAMS = 200


def word64(words):
    return words[0] | words[1] << 32


def refused(name, **flags):
    """The model's own verdict on this form (op_sel, clamp and the float modifiers)."""
    return rdna2_exec_int16.vop3_refusal(name, Fields(word64(vop3(name, 0, (V0, V0, V0), **flags)))) is not None


def lane_value(rng):
    """A 32-bit source: each half one of the edge values, or random."""
    halves = [rng.choice(SPECIAL_HALVES) if rng.random() < 0.6 else rng.getrandbits(16) for _ in range(2)]
    return halves[0] | halves[1] << 16


def fill(register, values):
    """v_writelane_b32 of each lane's value (a literal) into `register`."""
    words = []
    for lane, value in enumerate(values):
        words += [*vop3("v_writelane_b32", register, (LITERAL, INLINE_LANE + lane, INLINE_LANE)), value]
    return words


def accepted_forms(name):
    """Every op_sel and clamp combination the model runs for `name`, without float modifiers."""
    return [{"op_sel": op_sel, "clamp": clamp} for op_sel in range(16) for clamp in (0, 1)
            if not refused(name, op_sel=op_sel, clamp=clamp)]


def test_every_form_the_model_runs_matches_it_on_64_lanes_of_values():
    exe = binary()
    rng = random.Random(0x16B17)
    for name in NAMES:
        forms = accepted_forms(name)
        assert forms, name
        for lanes in (64, 32):
            words = []
            for register in SOURCES:
                words += fill(register, [lane_value(rng) for _ in range(lanes)])
            for index, flags in enumerate(forms):
                destination = FIRST_RESULT + index
                words += [vop1("v_mov_b32", destination, LITERAL), OLD_DESTINATION]
                words += vop3(name, destination, tuple(V0 + register for register in SOURCES), **flags)
            words = program(*words)
            assert run_python(lanes, False, [], words).startswith("steps"), name
            assert_same(exe, lanes, False, [], words)


def test_the_forms_the_model_refuses_are_refused_by_both_whatever_exec_holds():
    exe = binary()
    sources = tuple(V0 + register for register in SOURCES)
    for name in NAMES:
        variants = list(MODIFIERS)
        variants += [{"op_sel": op_sel, "clamp": clamp} for op_sel in range(16) for clamp in (0, 1)]
        variants = [flags for flags in variants if refused(name, **flags)]
        assert len(variants) >= len(MODIFIERS), name
        for flags in variants:
            words = program(*vop3(name, FIRST_RESULT, sources, **flags))
            for scalars in ([], LANES_OFF):
                assert run_python(32, False, scalars, words) == "error untranslated\n", (name, flags)
                assert_same(exe, 32, False, scalars, words)


def test_constant_sources_are_read_like_registers_in_both_models():
    """Inline integers (sign-extended to 32 bits), inline floats (their float32 bits) and a literal, in each
    source position, through both halves (op_sel); the literal case also takes the high half of the literal."""
    exe = binary()
    constants = (128, 129, 192, 193, 208, 240, 242, 247, LITERAL)
    for name in NAMES:
        forms = accepted_forms(name)
        words = fill(1, [0x8001_7FFF] * 32) + fill(2, [0x0003_FFFC] * 32) + fill(3, [0xFFFF_0002] * 32)
        index = 0
        for position in range(3):
            for constant in constants:
                for flags in (forms[0], forms[-1]):
                    sources = [V0 + register for register in SOURCES]
                    sources[position] = constant
                    destination = FIRST_RESULT + index
                    index += 1
                    words += vop3(name, destination, tuple(sources), **flags)
                    if constant == LITERAL:
                        words.append(0xC0DE_8005)
        assert_same(exe, 32, False, [], program(*words))


def random_instruction(rng, name):
    """A random VOP3 form of `name`: random sources (registers, constants, a literal), and most of the time a
    form the model runs (so that a program goes on past it); otherwise any select bits and clamp, now and then
    with a float modifier."""
    sources = tuple(rng.choice(OPERANDS + list(range(V0, V0 + 8))) for _ in range(3))
    if rng.random() < 0.9:
        flags = dict(rng.choice(accepted_forms(name)))
    else:
        flags = {"op_sel": rng.randrange(16), "clamp": rng.randrange(2)}
        if rng.random() < 0.5:
            flags.update(rng.choice(MODIFIERS))
    words = list(vop3(name, rng.randrange(1, 8), sources, **flags))
    if LITERAL in sources:
        words.append(rng.getrandbits(32))
    return words


def test_seeded_programs_of_16bit_integer_instructions_match_the_python_model():
    exe = binary()
    rng = random.Random(0x1D16)
    for _ in range(SEEDED_PROGRAMS):
        lanes = rng.choice((32, 64))
        scalars = [(index, lane_value(rng)) for index in range(16)]
        words = []
        for register in range(1, 8):
            words += [vop1("v_mov_b32", register, LITERAL), lane_value(rng)]
        for _ in range(rng.randrange(1, 6)):
            words += random_instruction(rng, rng.choice(NAMES))
        assert_same(exe, lanes, False, scalars, program(*words))
