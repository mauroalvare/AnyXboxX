"""f16 integer and normalized conversions, packs and three-operand ops in the RDNA 2 reference interpreter.

20 encodings: v_cvt_f16_{u16,i16}, v_cvt_{u16,i16}_f16 and v_cvt_norm_{i16,u16}_f16 (VOP1 and VOP3), and the
VOP3-only v_cvt_pknorm_{i16,u16}_{f16,f32}, v_pack_b32_f16 and v_min3/max3/med3_f16. Values are computed exactly
(Python floats hold every f16 value, and fractions.Fraction the products) and rounded once.
The conventions of emu/rdna2_exec_f16.py (#218) are kept: a 16-bit result goes to one half of the VGPR and the other
half is kept; inline float constants on an f16 source are their binary16 value; a literal or an inline integer gives
its low 16 bits; f16 denormals are kept (default MODE); a NaN result is 0x7E00; v_min/max_f16 (half_min, half_max)
order -0 below +0 and give the other operand for a single NaN.

Public sources:
- AMD, "RDNA 2" Instruction Set Architecture, November 2020 (printed page numbers): section 12.8 VOP1 opcodes 80-83
  and 99-100 (p. 146-149: cvt_{u16,i16}_f16 "conversion is done with truncation", "supports saturation";
  cvt_f16_{u16,i16} and cvt_norm "0.5ULP accuracy"); section 12.7 VOP2 opcodes 57-58 (p. 136: V_MAX_F16 and
  V_MIN_F16 give the other operand for a NaN and order -0.0 below +0.0, the rule of half_min/half_max; the IEEE_MODE
  rows that quiet a signalling NaN are not modelled); section 12.12 VOP3 opcodes 785-787, 849, 852, 855, 872-873
  (p. 179-185: v_pack_b32_f16 D[31:16] = S1.f16, D[15:0] = S0.f16; the pknorm ops D = {(norm)S1, (norm)S0};
  V_MIN3_F16 = V_MIN_F16(V_MIN_F16(S0, S1), S2) and V_MAX3_F16 likewise; the V_MED3_F16 pseudocode: with a NaN
  source, V_MIN3_F16; else, if V_MAX3_F16 equals S0, V_MAX_F16(S1, S2), if it equals S1, V_MAX_F16(S0, S2), else
  V_MAX_F16(S0, S1)); section 6.2.5 table 21 (p. 44): OPSEL is usable on V_CVT_PKNORM_{I16,U16}_F16,
  V_MIN3/MAX3/MED3_F16 and V_PACK_F16, not on VOP1 opcodes in VOP3.
- LLVM 20 AMDGPU (public code): VOP1Instructions.td selects fptoui.sat/fptosi.sat ("These i16 conversions naturally
  saturate"; NaN -> 0 in the LangRef) to v_cvt_{u16,i16}_f16; AMDGPUInstCombineIntrinsic.cpp folds
  llvm.amdgcn.fmed3 (also on half) with the same median rule as the ISA (an equal compare, so +0 == -0, then
  maxnum) and a NaN source to the minnum of the two others; APFloat's minnum/maxnum order -0 below +0.
- Microsoft Learn, "Data Conversion Rules" (Direct3D): FLOAT -> SNORM/UNORM gives 0 for NaN and clamps to [-1, 1]
  or [0, 1] (infinities included) before scaling by 32767 or 65535.

Assumptions, not measured (listed in the pull request): the pknorm_f32 product is rounded once from its exact value
(like cvt_norm's "0.5ULP accuracy"); norm ties round to nearest even; the VOP1 forms keep the high half like the VOP3
ones; abs/neg on an f16 half are bit operations on bit 15; min3/max3/med3 with every source NaN give 0x7E00 (as
v_min/max_f16 of main).

Everything else is refused by name (Untranslated here, Untranslatable in emu/rdna2_hlsl_f16conv.py): SDWA/DPP forms,
omod, op_sel on the unary ops (table 21) and on 32-bit sources, op_sel on the destination of the 32-bit packs, op_sel
on the high half of a constant, abs/neg on 16-bit integer sources, and clamp on the integer and packed results
(v_cvt_{u16,i16}_f16, v_cvt_norm_*, v_cvt_pknorm_*, v_pack_b32_f16).
"""
from fractions import Fraction
import math

from emu.rdna2_exec_f16 import (EXTENSIONS, INLINE_HALVES, VGPR_FIRST, bits_half, half_bits, half_clamp, half_max,
                                half_min)
from emu.rdna2_exec_ops import bits_float

HALF_SIGN, HALF_EXPONENT, HALF_MAGNITUDE, HALF_INFINITY = 0x8000, 0x7C00, 0x7FFF, 0x7C00
FLOAT_SIGN = 0x8000_0000
INT16_SCALE, UINT16_SCALE = 32767, 65535


def _is_nan(bits):
    return bits & HALF_MAGNITUDE > HALF_INFINITY


def _saturated(bits, low, high):
    """f16 -> integer, truncated toward zero and saturated; NaN -> 0."""
    if _is_nan(bits):
        return 0
    if bits & HALF_EXPONENT == HALF_EXPONENT:          # infinity
        return low if bits & HALF_SIGN else high
    return min(max(math.trunc(bits_half(bits)), low), high)


def _normalized(value, scale, signed):
    """round_even(clamp(value) * scale) from the exact product; NaN -> 0. `value`: exact Fraction, inf or NaN."""
    if isinstance(value, float):                       # infinity or NaN
        if math.isnan(value):
            return 0
        value = Fraction(1 if value > 0 else -1)
    clamped = min(max(value, Fraction(-1 if signed else 0)), Fraction(1))
    return round(clamped * scale) & 0xFFFF            # Fraction.__round__ rounds halves to even


def _half_value(bits):
    x = bits_half(bits)
    return Fraction(x) if math.isfinite(x) else x


def _float_value(bits):
    x = bits_float(bits)
    return Fraction(x) if math.isfinite(x) else x


def min3(a, b, c):
    return half_min(half_min(a, b), c)


def max3(a, b, c):
    return half_max(half_max(a, b), c)


def _same(x, y):
    """Float equality of two non-NaN halves (+0 == -0)."""
    return x == y or (x | y) & HALF_MAGNITUDE == 0


def med3(a, b, c):
    """ISA p. 182: with a NaN, min3; else the max of the two operands other than the first one equal to max3."""
    if any(_is_nan(x) for x in (a, b, c)):
        return min3(a, b, c)
    top = max3(a, b, c)
    if _same(top, a):
        return half_max(b, c)
    if _same(top, b):
        return half_max(a, c)
    return half_max(a, b)


def _pknorm(convert, scale, signed):
    return lambda a, b, c: _normalized(convert(b), scale, signed) << 16 | _normalized(convert(a), scale, signed)


# name -> (source kinds, function(a, b, c), result kind). Source kinds: "h" f16 half, "i" 16-bit integer half,
# "w" 32-bit f32 word. Result kinds: "h" f16 half, "i" 16-bit integer half, "w" whole VGPR.
OPERATIONS = {
    "v_cvt_f16_u16": ("i", lambda a, b, c: half_bits(float(a)), "h"),
    "v_cvt_f16_i16": ("i", lambda a, b, c: half_bits(float(a - 0x10000 if a & HALF_SIGN else a)), "h"),
    "v_cvt_u16_f16": ("h", lambda a, b, c: _saturated(a, 0, 0xFFFF), "i"),
    "v_cvt_i16_f16": ("h", lambda a, b, c: _saturated(a, -0x8000, 0x7FFF) & 0xFFFF, "i"),
    "v_cvt_norm_i16_f16": ("h", lambda a, b, c: _normalized(_half_value(a), INT16_SCALE, True), "i"),
    "v_cvt_norm_u16_f16": ("h", lambda a, b, c: _normalized(_half_value(a), UINT16_SCALE, False), "i"),
    "v_cvt_pknorm_i16_f16": ("hh", _pknorm(_half_value, INT16_SCALE, True), "w"),
    "v_cvt_pknorm_u16_f16": ("hh", _pknorm(_half_value, UINT16_SCALE, False), "w"),
    "v_cvt_pknorm_i16_f32": ("ww", _pknorm(_float_value, INT16_SCALE, True), "w"),
    "v_cvt_pknorm_u16_f32": ("ww", _pknorm(_float_value, UINT16_SCALE, False), "w"),
    "v_pack_b32_f16": ("hh", lambda a, b, c: b << 16 | a, "w"),
    "v_min3_f16": ("hhh", min3, "h"),
    "v_max3_f16": ("hhh", max3, "h"),
    "v_med3_f16": ("hhh", med3, "h"),
}
UNARY = frozenset(name for name, (kinds, _, _) in OPERATIONS.items() if len(kinds) == 1)
# clamp saturates an f16 result to [0, 1] as on the f16 ops of #218; on the integer and packed results it is refused.
CLAMP_SATURATES = frozenset({"v_cvt_f16_u16", "v_cvt_f16_i16", "v_min3_f16", "v_max3_f16", "v_med3_f16"})
DESTINATION_HIGH = frozenset({"v_min3_f16", "v_max3_f16", "v_med3_f16"})       # op_sel[3] writes D[31:16]


def refusal(name, fields):
    """Reason why a VOP3 form is not executed (unsourced), or None. Shared with emu/rdna2_hlsl_f16conv.py."""
    kinds = OPERATIONS[name][0]
    if fields.omod:
        return f"{name}: omod not supported (not modelled)"
    if fields.clamp and name not in CLAMP_SATURATES:
        return f"{name}: clamp not supported on an integer or packed result"
    if name in UNARY and fields.op_sel:
        return f"{name}: op_sel not usable on a VOP1 opcode in VOP3 (ISA section 6.2.5)"
    if fields.op_sel >> 3 & 1 and name not in DESTINATION_HIGH:
        return f"{name}: op_sel on the destination of a 32-bit result not supported"
    if fields.op_sel >> 2 & 1 and len(kinds) < 3:
        return f"{name}: op_sel on an unused source not supported"
    for index, kind in enumerate(kinds):
        if kind == "w" and fields.op_sel >> index & 1:
            return f"{name}: op_sel on a 32-bit source not supported"
        if kind == "i" and (fields.abs | fields.neg) >> index & 1:
            return f"{name}: abs/neg on an integer source not supported"
        code = fields.sources[index]
        if kind != "w" and fields.op_sel >> index & 1 and 128 <= code < VGPR_FIRST:
            return f"{name}: op_sel on the high half of a constant not supported"
    return None


def _refuse(message):
    from emu.rdna2_exec import Untranslated

    raise Untranslated(message)


def source(wave, kind, code, lane, literal, high=False):
    """Source value: the 32-bit word ("w"), or a 16-bit half / constant ("h", "i")."""
    if kind == "w":
        return wave.read_lane(code, lane, literal)
    if code < 128 or code >= VGPR_FIRST:               # SGPR, VCC, M0, EXEC, NULL and VGPR
        word = wave.read_lane(code, lane, literal)
        return word >> 16 if high else word & 0xFFFF
    if code in INLINE_HALVES and kind == "h":
        return INLINE_HALVES[code]
    # Inline integer, literal, or an inline float on a 16-bit integer source: low half of the 32-bit value
    # (the float32 pattern for a float constant: LLVM SIInstrInfo::isInlineConstant on 16-bit integer operands, #226).
    return wave.read_scalar(code, literal) & 0xFFFF


def _modified(value, kind, abs_bit, neg_bit):
    """VOP3 abs and neg: bit 15 of an f16 half, bit 31 of an f32 word, nothing on an integer."""
    if kind == "i":
        return value
    sign = FLOAT_SIGN if kind == "w" else HALF_SIGN
    if abs_bit:
        value &= ~sign
    if neg_bit:
        value ^= sign
    return value


def _write(destination, lane, name, value, high=False):
    result = OPERATIONS[name][2]
    old = destination[lane]
    if result == "w":
        destination[lane] = value & 0xFFFF_FFFF
    elif high:
        destination[lane] = (old & 0xFFFF) | (value & 0xFFFF) << 16
    else:
        destination[lane] = (old & 0xFFFF_0000) | (value & 0xFFFF)


def execute(wave, name, fields, literal):
    """VOP1 (the unary ops): low half in, no modifiers."""
    if fields["src0"] in EXTENSIONS:
        _refuse(f"{name}: SDWA/DPP form (src0 = 0x{fields['src0']:X}) not supported")
    kind, operation, _ = OPERATIONS[name]
    destination = wave.v[fields["vdst"]]
    for lane in wave.active():
        _write(destination, lane, name, operation(source(wave, kind, fields["src0"], lane, literal), 0, 0))


def execute_vop3(wave, name, fields, literal):
    """VOP3: op_sel[0-2] read the high half of a source, op_sel[3] writes D[31:16] (min3/max3/med3)."""
    reason = refusal(name, fields)
    if reason:
        _refuse(reason)
    kinds, operation, _ = OPERATIONS[name]
    destination = wave.v[fields.vdst]
    for lane in wave.active():
        values = [_modified(source(wave, kind, code, lane, literal, fields.op_sel >> index & 1), kind,
                            fields.abs >> index & 1, fields.neg >> index & 1)
                  for index, (kind, code) in enumerate(zip(kinds, fields.sources))] + [0, 0]
        value = operation(*values[:3])
        if fields.clamp and name in CLAMP_SATURATES:
            value = half_clamp(value)
        _write(destination, lane, name, value, fields.op_sel >> 3 & 1)


__all__ = ["OPERATIONS", "execute", "execute_vop3", "refusal"]
