"""f16 integer and normalized conversions, packs and min3/max3/med3 (emu/rdna2_exec_f16conv.py): reference
interpreter, bit-exact edges.

Expected words are worked out by hand in the comments from the public sources listed in
emu/rdna2_exec_f16conv.py (AMD RDNA 2 ISA pseudocode, LLVM AMDGPU, Microsoft Learn data conversion rules); the
NaN words follow main's v_min/max_f16 and the labelled assumptions there. The GPU parity of the HLSL translation is in
tests/test_rdna2_hlsl_f16conv.py.
"""
import pytest

from emu import rdna2_exec, rdna2_hlsl
from test_rdna2_exec import LITERAL, V, const, program, vop1, vop3
from test_rdna2_sdwa_dpp_exec import SDWA, sdwa_word

OLD = 0xA5A5_A5A5                    # destination before the instruction: its untouched half must survive
DST = 5
HALF, ONE, MINUS_TWO, FOUR, ONE_OVER_TWO_PI, MINUS_ONE = 240, 242, 245, 246, 248, 193


def run(words, *sources, exec_mask=0xFFFF_FFFF):
    """Lane i reads sources[k][i % len] in v1, v2, v3; v5 starts at OLD. Returns v5."""
    wave = rdna2_exec.Wave()
    for index, values in enumerate(sources):
        wave.v[1 + index] = [values[lane % len(values)] for lane in range(32)]
    wave.v[DST] = [OLD] * 32
    wave.s[126] = exec_mask
    return rdna2_exec.run(program(*words), wave).v[DST]


def unary(name, values, encoding="VOP1", **modifiers):
    words = [vop1(name, DST, V + 1)] if encoding == "VOP1" else list(vop3(name, DST, V + 1, **modifiers))
    return run(words, values)[:len(values)]


def low(name, values, **modifiers):
    """Results of a unary op on the low halves (high halves junk), VOP1 and VOP3 checked equal."""
    words = [0xBEEF_0000 | value for value in values]
    first, second = unary(name, words), unary(name, words, "VOP3", **modifiers)
    assert first == second
    assert all(word >> 16 == OLD >> 16 for word in first), "high half of the destination kept"
    return [word & 0xFFFF for word in first]


# --- integer conversions ------------------------------------------------------------------------------------

def test_cvt_f16_from_16_bit_integers():
    values = [0, 1, 2047, 2049, 2051, 0x7FFF, 0x8000, 0xFFE0, 0xFFEF, 0xFFF0, 0xFFFF]
    # u16 (0.5 ULP, to nearest even): 2049 ties to 2048, 2051 to 2052, 32767 -> 32768, 65504 exact, 65519 -> 65504,
    # 65520 and 65535 -> +inf (IEEE overflow).  i16: 0x8000 = -32768, 0xFFFF = -1.
    assert low("v_cvt_f16_u16", values) == [0x0000, 0x3C00, 0x67FF, 0x6800, 0x6802, 0x7800, 0x7800, 0x7BFF,
                                            0x7BFF, 0x7C00, 0x7C00]
    assert low("v_cvt_f16_i16", values) == [0x0000, 0x3C00, 0x67FF, 0x6800, 0x6802, 0x7800, 0xF800, 0xD000,
                                            0xCC40, 0xCC00, 0xBC00]


def test_cvt_16_bit_integers_from_f16_truncate_and_saturate():
    values = [0x3C00, 0x3FFF, 0xBFFF, 0x77FF, 0x7800, 0x7BFF, 0xF800, 0xF801, 0x7C00, 0xFC00, 0x7E00, 0xFE00,
              0x3800, 0x0001]
    # Truncation and saturation, NaN -> 0 (ISA p. 146; LLVM selects fptoui.sat / fptosi.sat to these opcodes):
    # 1, 1.999 -> 1, -1.999 -> -1, 32752, 32768, 65504, -32768, -32800, +inf, -inf, NaN, -NaN, 0.5, denormal
    assert low("v_cvt_u16_f16", values) == [1, 1, 0, 32752, 32768, 65504, 0, 0, 0xFFFF, 0, 0, 0, 0, 0]
    assert low("v_cvt_i16_f16", values) == [1, 1, 0xFFFF, 32752, 0x7FFF, 0x7FFF, 0x8000, 0x8000, 0x7FFF, 0x8000,
                                            0, 0, 0, 0]


def test_cvt_norm_rounds_the_exact_product_once():
    values = [0x3C00, 0xBC00, 0x3800, 0xB800, 0x3801, 0x0001, 0x8001, 0x7C00, 0xFC00, 0x7E00, 0x4000, 0x8000,
              0x3555]
    # One rounding (0.5 ULP, ISA p. 149) after the D3D clamp, NaN -> 0 ; ties to even (assumed):
    # 0.5 x 32767 = 16383.5 -> 16384, 0.5 x 65535 = 32767.5 -> 32768 ; 0x3801 = 0.5 + 2^-11:
    # x 65535 = 32799.4995 -> 32799, x 32767 = 16399.4995 -> 16399 ; a denormal 2^-24 x 65535 = 0.0039 -> 0 ;
    # 0x3555 = 0.333251953125: x 32767 = 10919.67 -> 10920, x 65535 = 21839.67 -> 21840.
    assert low("v_cvt_norm_i16_f16", values) == [32767, 0x8001, 16384, 0xC000, 16399, 0, 0, 32767, 0x8001, 0,
                                                 32767, 0, 10920]
    assert low("v_cvt_norm_u16_f16", values) == [65535, 0, 32768, 0, 32799, 0, 0, 65535, 0, 0, 65535, 0, 21840]


def test_inline_constants_and_literals_on_conversions():
    wave = rdna2_exec.Wave()
    wave.v[DST] = [OLD] * 32
    for register in range(6, 11):
        wave.v[register] = [OLD] * 32
    rdna2_exec.run(program(
        vop1("v_cvt_i16_f16", 5, MINUS_TWO),                 # f16 -2.0 -> -2 = 0xFFFE
        vop1("v_cvt_u16_f16", 6, ONE_OVER_TWO_PI),           # f16 0x3118 = 0.159 -> 0
        vop1("v_cvt_f16_u16", 7, ONE),                       # 16-bit integer source: low half of 0x3F800000 = 0
        vop1("v_cvt_f16_i16", 8, MINUS_ONE),                 # -1 -> 0xBC00
        vop1("v_cvt_f16_u16", 9, LITERAL), 0x1234_0040,      # literal low half 64 -> 0x5400
        vop1("v_cvt_norm_i16_f16", 10, LITERAL), 0xFFFF_B800), wave)    # -0.5 x 32767 = -16383.5 -> -16384
    assert [wave.v[r][0] for r in range(5, 11)] == [0xA5A5_FFFE, 0xA5A5_0000, 0xA5A5_0000, 0xA5A5_BC00,
                                                    0xA5A5_5400, 0xA5A5_C000]


# --- packs ------------------------------------------------------------------------------------------------------

def test_cvt_pknorm_f32_rounds_the_exact_product_once():
    # 0.5 + 2^-24 (0x3F000001): x 32767 = 16383.5000019 -> 16384 ; x 65535 = 32767.5000039 -> 32768 ;
    # 0.5 - 2^-25 (0x3EFFFFFF): x 32767 = 16383.49999902 -> 16383 (an f32 product would round to 16383.5) ;
    # 1/65535-ish tie: 0x37800080 = 2^-16 x (1 + 2^-16): x 65535 = 1.0000 - ... -> 1.
    pairs = [(0x3F00_0001, 0x3EFF_FFFF), (0xBF00_0001, 0xBEFF_FFFF), (0x7FC0_0000, 0xFFC0_0001),
             (0x4000_0000, 0xC000_0000), (0x7F80_0000, 0xFF80_0000), (0x0000_0001, 0x8000_0001),
             (0x3780_0080, 0x3800_0000)]
    signed = run(list(vop3("v_cvt_pknorm_i16_f32", DST, V + 1, V + 2)), *zip(*pairs))[:len(pairs)]
    unsigned = run(list(vop3("v_cvt_pknorm_u16_f32", DST, V + 1, V + 2)), *zip(*pairs))[:len(pairs)]
    assert signed == [16383 << 16 | 16384, (-16383 & 0xFFFF) << 16 | (-16384 & 0xFFFF), 0, 0x8001_7FFF,
                      0x8001_7FFF, 0, 0x0001_0000]
    assert unsigned == [32767 << 16 | 32768, 0, 0, 0x0000_FFFF, 0x0000_FFFF, 0, 0x0002_0001]


def test_cvt_pknorm_f16_takes_each_half_with_op_sel_and_modifiers():
    # v1 = {1.0, 0.5}, v2 = {-0.5, 0x3801}: S0 = v1 high half (op_sel[0]) = 1.0, S1 = -(v2 low half) = -0x3801.
    words = list(vop3("v_cvt_pknorm_i16_f16", DST, V + 1, V + 2, op_sel=0b0001, neg=0b10))
    assert run(words, [0x3C00_3800], [0xB800_3801])[0] == (-16399 & 0xFFFF) << 16 | 32767
    assert run(list(vop3("v_cvt_pknorm_u16_f16", DST, V + 1, V + 2)),
               [0x0000_3800], [0x0000_3801])[0] == 32799 << 16 | 32768


def test_pack_b32_f16_moves_bits_and_sign():
    # D = {S1.f16, S0.f16} (ISA p. 179); abs then neg act on bit 15 of the half each source gives (assumed bit
    # operations: an sNaN half keeps its payload).
    words = list(vop3("v_pack_b32_f16", DST, V + 1, V + 2, op_sel=0b0010, neg=0b01, abs_=0b10))
    assert run(words, [0x1234_7C01], [0xBC00_5678])[0] == 0x3C00_FC01


# --- min3 / max3 / med3 --------------------------------------------------------------------------------------

def three(name, a, b, c, **modifiers):
    return run(list(vop3(name, DST, V + 1, V + 2, V + 3, **modifiers)), a, b, c)


def test_min3_max3_med3_with_signed_zeros_and_nans():
    a = [0x0000, 0x8000, 0x7E00, 0x7C01, 0x3C00, 0x7E00, 0xBC00]
    b = [0x8000, 0x0000, 0x3C00, 0x7E01, 0x7E00, 0x4000, 0x7C00]
    c = [0xBC00, 0xBC00, 0x4000, 0x7E02, 0x4000, 0x7E00, 0x3C00]
    # min3 = min(min(S0, S1), S2) and max3 alike (ISA p. 181), with v_min/max_f16 of main: -0 < +0, a NaN gives
    # the other operand, two NaN give 0x7E00. med3 (ISA p. 182): with a NaN, min3; else, if max3 equals S0 as a
    # float (+0 == -0), max(S1, S2); if it equals S1, max(S0, S2); else max(S0, S1). So med3(-0, +0, -1) = +0.
    assert [w & 0xFFFF for w in three("v_min3_f16", a, b, c)][:7] == [0xBC00, 0xBC00, 0x3C00, 0x7E00, 0x3C00,
                                                                      0x4000, 0xBC00]
    assert [w & 0xFFFF for w in three("v_max3_f16", a, b, c)][:7] == [0x0000, 0x0000, 0x4000, 0x7E00, 0x4000,
                                                                      0x4000, 0x7C00]
    assert [w & 0xFFFF for w in three("v_med3_f16", a, b, c)][:7] == [0x8000, 0x0000, 0x3C00, 0x7E00, 0x3C00,
                                                                      0x4000, 0x3C00]


def test_min3_max3_med3_nan_and_zero_rules():
    # The rules of main's v_min/max_f16 inside min3/max3 (ISA p. 181: V_MIN3_F16 = V_MIN_F16(V_MIN_F16(S0, S1), S2)):
    # -0 below +0, a single NaN gives the other operand. med3 with a NaN source is min3 (ISA p. 182), so the
    # med3(x, 0, 1.0) clamp gives +0 for a NaN x; LLVM folds llvm.amdgcn.fmed3 the same way.
    a = [0x0000, 0x8000, 0x7E00, 0x0000, 0x7E00, 0x3C00]
    b = [0x8000, 0x0000, 0x0000, 0x3C00, 0x4000, 0x4000]
    c = [0x0000, 0x8000, 0x3C00, 0x7E00, 0x3C00, 0x7E00]
    assert [w & 0xFFFF for w in three("v_min3_f16", a, b, c)][:6] == [0x8000, 0x8000, 0x0000, 0x0000, 0x3C00,
                                                                      0x3C00]
    assert [w & 0xFFFF for w in three("v_max3_f16", a, b, c)][:6] == [0x0000, 0x0000, 0x3C00, 0x3C00, 0x4000,
                                                                      0x4000]
    assert [w & 0xFFFF for w in three("v_med3_f16", a, b, c)][:6] == [0x0000, 0x0000, 0x0000, 0x0000, 0x3C00,
                                                                      0x3C00]


def test_med3_op_sel_and_clamp():
    # op_sel 0b1010: S1 from v2[31:16], result to D[31:16] (D[15:0] kept); clamp caps 2.0 at 1.0.
    words = three("v_med3_f16", [0x0000_4400], [0x4200_0000], [0x0000_4000], op_sel=0b1010)
    assert words[0] == 0x4200_A5A5                    # median(4, 3, 2) = 3.0 = 0x4200
    assert three("v_max3_f16", [0x4000], [0xC000], [0x7E00], clamp=1)[0] == 0xA5A5_3C00


# --- exec, refusals ----------------------------------------------------------------------------------------

def test_inactive_lanes_keep_their_destination():
    result = run([vop1("v_cvt_f16_u16", DST, V + 1)], [7], exec_mask=0x0000_FFF0)
    assert result[:4] == [OLD] * 4 and result[4] == 0xA5A5_4700


@pytest.mark.parametrize("words, message", [
    ([vop1("v_cvt_u16_f16", DST, SDWA), sdwa_word(1)], "SDWA/DPP"),
    (list(vop3("v_cvt_f16_u16", DST, V + 1, omod=1)), "omod"),
    (list(vop3("v_cvt_norm_i16_f16", DST, V + 1, op_sel=0b0001)), "op_sel"),
    (list(vop3("v_cvt_u16_f16", DST, V + 1, clamp=1)), "clamp"),
    (list(vop3("v_cvt_i16_f16", DST, V + 1, clamp=1)), "clamp"),
    (list(vop3("v_cvt_norm_u16_f16", DST, V + 1, clamp=1)), "clamp"),
    (list(vop3("v_cvt_pknorm_u16_f16", DST, V + 1, V + 2, clamp=1)), "clamp"),
    (list(vop3("v_cvt_pknorm_i16_f32", DST, V + 1, V + 2, clamp=1)), "clamp"),
    (list(vop3("v_pack_b32_f16", DST, V + 1, V + 2, clamp=1)), "clamp"),
    (list(vop3("v_cvt_pknorm_i16_f16", DST, V + 1, V + 2, op_sel=0b1000)), "destination"),
    (list(vop3("v_pack_b32_f16", DST, V + 1, V + 2, op_sel=0b0100)), "unused"),
    (list(vop3("v_cvt_pknorm_u16_f32", DST, V + 1, V + 2, op_sel=0b0001)), "32-bit"),
    (list(vop3("v_cvt_f16_u16", DST, V + 1, neg=1)), "integer"),
    (list(vop3("v_min3_f16", DST, ONE, V + 2, V + 3, op_sel=0b0001)), "constant"),
])
def test_unsourced_forms_are_refused_by_name_on_both_sides(words, message):
    code = program(*words)
    with pytest.raises(rdna2_exec.Untranslated, match=message):
        rdna2_exec.run(code)
    with pytest.raises(rdna2_hlsl.Untranslatable, match=message):
        rdna2_hlsl.translate(code)


def test_every_encoding_runs_and_counts_in_coverage():
    from emu import rdna2_exec_f16conv
    from emu.rdna2_ops import MNEMONICS

    found = [(family, name) for (family, _), name in MNEMONICS.items() if name in rdna2_exec_f16conv.OPERATIONS]
    assert len(found) == 20
    assert all(rdna2_exec.executable(family, name) for family, name in found)


def test_constant_source_on_three_operand_ops():
    # Inline f16 constants: 4.0 = 0x4400, 0.5 = 0x3800 ; median(4, 0.5, x).
    assert three_constants() == [0xA5A5_4000, 0xA5A5_3800]


def three_constants():
    wave = rdna2_exec.Wave()
    wave.v[1] = [0x4000, 0x2000] * 16
    wave.v[DST] = [OLD] * 32
    rdna2_exec.run(program(*vop3("v_med3_f16", DST, FOUR, HALF, V + 1)), wave)
    return wave.v[DST][:2]


def test_const_helper_is_an_integer_source():
    # 64 as a 16-bit integer source: v_cvt_f16_i16(64) = 0x5400.
    assert run([vop1("v_cvt_f16_i16", DST, const(64))], [0])[0] == 0xA5A5_5400
