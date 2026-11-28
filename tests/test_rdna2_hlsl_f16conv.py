"""f16 integer and normalized conversions, packs and min3/max3/med3: the HLSL translation against the reference
interpreter, byte for byte.

- Every one of the 20 encodings of emu/rdna2_exec_f16conv.py, with its VOP3 modifier variants (neg/abs, clamp,
  op_sel where it is accepted), inline constants and literals, runs on per-lane f16 and f32 boundary values under
  junk other halves and a partial EXEC that must leave the inactive lanes' destinations untouched (wave32, wave64).
- The 6 unary ops run on all 65,536 16-bit inputs (a loop of 2,048 steps over 32 lanes), each result folded into a
  per-lane checksum, so a single wrong word on any input changes the stored memory.
- The helpers also compile in vertex and pixel shaders (emu/rdna2_hlsl_graphics.py).
"""
import random
import struct

import pytest

import test_rdna2_hlsl_wave64 as wave64
from emu import rdna2, rdna2_exec, rdna2_exec_f16conv, rdna2_gpu, rdna2_hlsl, rdna2_hlsl_graphics
from emu.rdna2_ops import MNEMONICS
from test_rdna2_exec import LITERAL, V, const, sop1, sop2, sopc, sopp, vop1, vop2, vop3
from test_rdna2_exec_memory import INPUT, OUTPUT, global_
from test_rdna2_graphics import PIXEL_SHADER, VERTEX_SHADER, WIDTH
from test_rdna2_hlsl import compare, store_lane, with_lane_offset

HALF, ONE, MINUS_TWO, FOUR, ONE_OVER_TWO_PI, MINUS_ONE, EXEC_LO = 240, 242, 245, 246, 248, 193, 126
SLOTS_PER_BASE = 16                  # 15 x 128 bytes fits the signed 12-bit offset
SLOTS_PER_BASE_64 = 8                # 7 x 256 bytes
FIRST_RESULT = 32                    # v31 holds the store address
PARTIAL_EXEC, PARTIAL_EXEC_HIGH = 0x0F0F_F0F3, 0xF00F_0FF0
HALF_EDGES = (0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3400, 0xB400, 0x3800, 0xB800, 0x3801, 0xB7FF,
              0x3A00, 0x3BFF, 0xBBFF, 0x3C00, 0xBC00, 0x3E00, 0x4100, 0xC100, 0x4500, 0x63FF, 0xE3FF, 0x6400,
              0x77FF, 0x7800, 0xF801, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0x7C01, 0xFE05, 0x3555, 0xB4CD)
FLOAT_EDGES = (0x0000_0000, 0x8000_0000, 0x0000_0001, 0x807F_FFFF, 0x3F80_0000, 0xBF80_0000, 0x3F00_0001,
               0x3EFF_FFFF, 0xBF00_0001, 0x3F80_1FFF, 0x477F_F000, 0x477F_EFFF, 0xC780_0000, 0x7F80_0000,
               0xFF80_0000, 0x7FC0_0000, 0xFF80_0001, 0x3380_0000, 0x337F_FFFF, 0x387F_FFFF, 0x3880_0000,
               0x3780_0080, 0x2F80_0000, 0x3F7F_FFFF, 0x4000_0000)
gpu = pytest.mark.skipif(not rdna2_gpu.available(), reason="dxc or anyxboxx_gpu_run.exe missing")

UNARY = sorted(rdna2_exec_f16conv.UNARY)
MODIFIED = {                         # VOP3 variants of each op: (op_sel, neg, abs, clamp)
    **{name: [(0, 1, 0, 0), (0, 0, 1, 0)] for name in ("v_cvt_u16_f16", "v_cvt_i16_f16", "v_cvt_norm_i16_f16",
                                                        "v_cvt_norm_u16_f16")},
    "v_cvt_f16_u16": [(0, 0, 0, 1)], "v_cvt_f16_i16": [(0, 0, 0, 1)],
}
TWO_SOURCE = {
    "v_cvt_pknorm_i16_f16": [(0, 0, 0, 0), (0b0011, 0b01, 0b10, 0), (0b0001, 0b11, 0, 0)],
    "v_cvt_pknorm_u16_f16": [(0, 0, 0, 0), (0b0010, 0b10, 0b01, 0)],
    "v_pack_b32_f16": [(0, 0, 0, 0), (0b0011, 0b01, 0b10, 0), (0b0010, 0b11, 0, 0)],
}
FLOAT_SOURCE = {
    "v_cvt_pknorm_i16_f32": [(0, 0, 0, 0), (0, 0b10, 0b01, 0)],
    "v_cvt_pknorm_u16_f32": [(0, 0, 0, 0), (0, 0b01, 0, 0)],
}
THREE_SOURCE = {name: [(0, 0, 0, 0), (0b0101, 0b010, 0b001, 0), (0b1110, 0b100, 0, 1), (0b1011, 0, 0b110, 0)]
                for name in ("v_min3_f16", "v_max3_f16", "v_med3_f16")}


def lane_words(seed, count, edges, halves=True):
    """`count` words built from boundary values (one lane in eight fully random)."""
    generator = random.Random(seed)
    words = []
    for lane in range(count):
        if lane % 8 == 7:
            words.append(generator.getrandbits(32))
        elif halves:
            words.append(generator.choice(edges) << 16 | generator.choice(edges))
        else:
            words.append(generator.choice(edges))
    return words


def parity_memory(count=32):
    """Rows of `count` words: v1-v3 f16 pairs, v4 old destination, v5-v6 f32 words."""
    rows = [word for seed in (1, 2, 3) for word in lane_words(seed, count, HALF_EDGES)]
    rows += lane_words(4, count, HALF_EDGES)
    rows += [word for seed in (5, 6) for word in lane_words(seed, count, FLOAT_EDGES, halves=False)]
    image = bytearray(INPUT + 4 * len(rows))
    struct.pack_into(f"<{len(rows)}I", image, INPUT, *rows)
    return bytes(image)


def cases():
    """Instruction words and their literals, one result register each (in order from FIRST_RESULT)."""
    a, b, c, x, y = V + 1, V + 2, V + 3, V + 5, V + 6
    found = []
    for name in UNARY:
        found += [([vop1(name, 0, a)], ()), (list(vop3(name, 0, a)), ())]
        found += [(list(vop3(name, 0, a, op_sel=0, neg=neg, abs_=abs_, clamp=clamp)), ())
                  for _, neg, abs_, clamp in MODIFIED[name]]
    for name, variants in FLOAT_SOURCE.items():
        found += [(list(vop3(name, 0, x, y, op_sel=op_sel, neg=neg, abs_=abs_, clamp=clamp)), ())
                  for op_sel, neg, abs_, clamp in variants]
    for name, variants in TWO_SOURCE.items():
        found += [(list(vop3(name, 0, a, b, op_sel=op_sel, neg=neg, abs_=abs_, clamp=clamp)), ())
                  for op_sel, neg, abs_, clamp in variants]
    for name, variants in THREE_SOURCE.items():
        found += [(list(vop3(name, 0, a, b, c, op_sel=op_sel, neg=neg, abs_=abs_, clamp=clamp)), ())
                  for op_sel, neg, abs_, clamp in variants]
    return found + constant_cases()


def constant_cases():
    """Inline f16 and integer constants and literals on f16, integer and f32 sources."""
    a, b = V + 1, V + 2
    return [
        ([vop1("v_cvt_i16_f16", 0, MINUS_TWO)], ()),
        ([vop1("v_cvt_u16_f16", 0, ONE_OVER_TWO_PI)], ()),
        ([vop1("v_cvt_f16_u16", 0, ONE)], ()),                       # integer source: low half of 0x3F800000
        ([vop1("v_cvt_f16_i16", 0, MINUS_ONE)], ()),
        ([vop1("v_cvt_norm_i16_f16", 0, LITERAL)], (0x1234_B4CD,)),
        ([vop1("v_cvt_norm_u16_f16", 0, const(1))], ()),             # integer 1 = the f16 denormal 0x0001
        (list(vop3("v_cvt_f16_i16", 0, LITERAL)), (0x0000_FFF0,)),
        (list(vop3("v_med3_f16", 0, FOUR, HALF, a)), ()),
        (list(vop3("v_pack_b32_f16", 0, ONE, b, op_sel=0b0010)), ()),
        (list(vop3("v_cvt_pknorm_i16_f16", 0, a, MINUS_TWO, neg=0b10)), ()),
        (list(vop3("v_cvt_pknorm_i16_f32", 0, HALF, LITERAL)), (0xBF00_0001,)),
        (list(vop3("v_cvt_pknorm_u16_f32", 0, LITERAL, ONE)), (0x3F00_0001,)),
        ([vop1("v_cvt_u16_f16", 0, LITERAL)], (0x0000_7BFF,)),
    ]


def with_vdst(words, vdst):
    if len(words) == 1:                                              # VOP1 bits 24:17
        return [words[0] & ~(0xFF << 17) | vdst << 17]
    low, high = words
    return [low & ~0xFF | vdst, high]


def operations(count, first=FIRST_RESULT):
    """Load the sources, preset every destination to the lane's old value, run all cases under a partial EXEC."""
    words = [word for row, register in enumerate((1, 2, 3, 4, 5, 6))
             for word in global_("global_load_dword", addr=31, vdst=register, saddr=0, offset=4 * count * row)]
    listed = cases()
    words += [vop1("v_mov_b32", first + index, V + 4) for index in range(len(listed))]
    words += [sop1("s_mov_b32", EXEC_LO, LITERAL), PARTIAL_EXEC]
    if count == 64:
        words += [sop1("s_mov_b32", EXEC_LO + 1, LITERAL), PARTIAL_EXEC_HIGH]
    for index, (instruction, literal) in enumerate(listed):
        words += [*with_vdst(instruction, first + index), *literal]
    words += [sop1("s_mov_b32", EXEC_LO, MINUS_ONE)]
    if count == 64:
        words += [sop1("s_mov_b32", EXEC_LO + 1, MINUS_ONE)]
    return words, len(listed)


def parity_program():
    words, total = operations(32)
    stores = []
    for slot in range(total):
        if slot and slot % SLOTS_PER_BASE == 0:
            stores += [vop2("v_add_nc_u32", 31, LITERAL, 31), SLOTS_PER_BASE * 128]
        stores += store_lane(FIRST_RESULT + slot, slot % SLOTS_PER_BASE)
    return with_lane_offset(*words, *stores), total


def test_parity_program_fits_the_register_file_and_memory():
    _, total = parity_program()
    assert FIRST_RESULT + total <= 256 and total * wave64.SLOT <= 0x10000
    assert total >= 50


def test_every_encoding_is_in_the_parity_program():
    data = parity_program()[0]
    used = {(i.family, rdna2.mnemonic(i.family, i.opcode)) for i in rdna2.decode(data)}
    wanted = {(family, name) for (family, _), name in MNEMONICS.items() if name in rdna2_exec_f16conv.OPERATIONS}
    assert len(wanted) == 20 and wanted <= used
    text = rdna2_hlsl.translate(data)
    for helper in ("F16ToU16(", "F16ToI16(", "F16Norm(", "F32Norm16(", "F16Med3("):
        assert text.count(helper) >= 2, helper                       # defined and used


def test_parity_program_results_vary_by_lane():
    """Guard for the GPU test: most results differ between lanes, and inactive lanes kept their old value."""
    wave = rdna2_exec.Wave()
    wave.s[0], wave.s[2] = INPUT, OUTPUT
    wave.memory.write(0, parity_memory())
    data, total = parity_program()
    rdna2_exec.run(data, wave)
    varied = [len(set(wave.v[FIRST_RESULT + i])) > 4 for i in range(total)]
    assert sum(varied) > 0.8 * total
    assert all(wave.v[FIRST_RESULT + i][2] == wave.v[4][2] for i in range(total))      # lane 2 inactive


def test_helpers_are_integer_only_and_straight_line():
    from emu.rdna2_hlsl_f16conv import HELPERS
    assert "while" not in HELPERS and "for (" not in HELPERS and "double" not in HELPERS
    assert "uint64_t" not in HELPERS and "clamp(" not in HELPERS


@gpu
def test_f16conv_ops_match_interpreter_on_gpu():
    expected, actual = compare(parity_program()[0], memory=parity_memory())
    assert actual == expected


@gpu
def test_f16conv_ops_match_interpreter_on_gpu_in_wave64():
    words, total = operations(64)
    stores = []
    for slot in range(total):
        if slot and slot % SLOTS_PER_BASE_64 == 0:
            stores += [vop2("v_add_nc_u32", 31, LITERAL, 31), SLOTS_PER_BASE_64 * wave64.SLOT]
        stores += wave64.store_lane(FIRST_RESULT + slot, slot % SLOTS_PER_BASE_64)
    expected, actual = wave64.compare(wave64.with_lane_offset(*words, *stores), memory=parity_memory(64))
    assert actual == expected


# --- every f16 input -----------------------------------------------------------------------------------------

STEPS = 2048                          # 32 lanes x 2048 steps = 65,536 inputs
INPUT_REGISTER, BASE_REGISTER, RESULT_REGISTER, FIRST_SUM = 1, 10, 20, 40
COUNTER = 10                          # s10


def sweep_program():
    """Lane L runs inputs L x 2048 + i (i = 0..2047) through each unary op (VOP1) and folds every 32-bit
    destination word into a per-op sum: sum = rotate_left(sum, 5) + word."""
    head = [vop2("v_lshlrev_b32", BASE_REGISTER, const(11), 0), sop1("s_mov_b32", COUNTER, const(0))]
    body = [vop2("v_add_nc_u32", INPUT_REGISTER, COUNTER, BASE_REGISTER)]
    for index, name in enumerate(UNARY):
        total = FIRST_SUM + index
        body += [vop1(name, RESULT_REGISTER, V + INPUT_REGISTER),
                 *vop3("v_alignbit_b32", total, V + total, V + total, const(27)),
                 vop2("v_add_nc_u32", total, V + RESULT_REGISTER, total)]
    body += [sop2("s_add_u32", COUNTER, COUNTER, const(1)), sopc("s_cmp_lg_u32", COUNTER, LITERAL), STEPS]
    body.append(sopp("s_cbranch_scc1", -(len(body) + 1) & 0xFFFF))
    stores = [word for index in range(len(UNARY)) for word in store_lane(FIRST_SUM + index, index)]
    return with_lane_offset(*head, *body, *stores)


@gpu
def test_unary_ops_on_every_f16_input_match_interpreter_on_gpu():
    expected, actual = compare(sweep_program())
    assert actual == expected
    sums = struct.unpack_from(f"<{32 * len(UNARY)}I", expected, OUTPUT)
    assert all(len(set(sums[32 * index:32 * index + 32])) >= 3 for index in range(len(UNARY)))


# --- graphics stages -------------------------------------------------------------------------------------------

@gpu
def test_helpers_compile_in_vertex_and_pixel_shaders():
    op = struct.pack("<3I", vop1("v_cvt_norm_u16_f16", 10, V + 1), *vop3("v_cvt_pknorm_u16_f32", 11, V + 2, V + 3))
    vertex = rdna2_hlsl_graphics.translate_vertex(op + VERTEX_SHADER)
    pixel = rdna2_hlsl_graphics.translate_pixel(op + PIXEL_SHADER, WIDTH)
    assert "uint F16Norm(" in vertex and "uint F32Norm16(" in pixel
    rdna2_gpu.compile_dxil(vertex, model=rdna2_gpu.VERTEX_MODEL)
    model = rdna2_gpu.PIXEL_WAVE_MODEL if "IsHelperLane()" in pixel else rdna2_gpu.PIXEL_MODEL
    rdna2_gpu.compile_dxil(pixel, model=model)
