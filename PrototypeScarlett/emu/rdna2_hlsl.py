"""Recompilateur RDNA 2 → HLSL (shader de calcul, puis DXIL avec dxc).

Un programme RDNA 2 (un wave32) devient un shader de calcul HLSL d'un groupe de 32 threads.
Un programme sans branchement s'écrit en ligne droite ; sinon, le contrôle de flux est rendu par une
boucle `while (pc != END_PC)` et une garde `if (pc == ...)` par bloc de base, car les branchements RDNA 2
sont scalaires donc uniformes pour tout le groupe. Pas de `switch (pc)` : sur WARP, un `switch` dans
cette boucle fait planter le lanceur (0xC0000005) quand un bloc contient une barrière de groupe (Ballot64
du wave64), mesuré avec tests/test_rdna2_hlsl_wave64.py ; la chaîne de gardes passe partout. La mémoire du programme est le buffer brut u0 (adresses 32 bits
basses). Chaque instruction traduite est vérifiée contre l'interpréteur de référence
(emu/rdna2_exec*) ; une instruction non couverte lève Untranslatable en la nommant.
"""
import struct

from emu import (rdna2, rdna2_exec_cmp16, rdna2_exec_div, rdna2_exec_pkf16, rdna2_exec_wide, rdna2_hlsl_cmpf16,
                 rdna2_hlsl_div, rdna2_hlsl_dot, rdna2_hlsl_ext, rdna2_hlsl_f16, rdna2_hlsl_format, rdna2_hlsl_image,
                 rdna2_hlsl_int16, rdna2_hlsl_movrel, rdna2_hlsl_pc, rdna2_hlsl_pkf16, rdna2_hlsl_sad,
                 rdna2_hlsl_scalar64, rdna2_hlsl_wide)
from emu import rdna2_hlsl_gather
from emu.rdna2_exec_cmpf16 import COMPAREF16
from emu.rdna2_exec_ops import signed
from emu.rdna2_exec_ops import SIXTEEN_BIT_RESULTS, VECTOR_NO_OPERATION
from emu.rdna2_exec_vop3 import LITERAL_VOP2, MAD64, Fields, float_inputs, float_result
from emu.rdna2_hlsl_clamp import CARRY, carry_write, integer_template, refuse_integer_clamp
from emu import rdna2_hlsl_f16conv
from emu.rdna2_hlsl_memory import memory_statements
from emu.rdna2_hlsl_ops import (BRANCHES, HELPERS as ALU_HELPERS, SCALAR, VECTOR, Untranslatable, source_high,
                                compare_condition, hexa, is_compare, scalar_statement, scalar_target,
                                source, NULL)
from emu.rdna2_operands import FIELDS

ACTIVE = "((s[126] >> lane) & 1u) != 0"
SCALAR_FAMILIES = ("SOP1", "SOP2", "SOPC", "SOPK", "SOPP")
MEMORY_FAMILIES = ("SMEM", "DS", "MUBUF", "MTBUF", "FLAT", "GLOBAL", "SCRATCH")
OUTPUT_SCALE = {1: "2.0f", 2: "4.0f", 3: "0.5f"}
WAVE_SIZES = (32, 64)

PRELUDE = """// Généré par AnyXboxX (emu/rdna2_hlsl.py) : programme RDNA 2 recompilé en HLSL.
#define END_PC 0xFFFFFFFFu
RWByteAddressBuffer g_memory : register(u0);
groupshared uint g_lds[8128];      // D3D12 : 32 Kio de mémoire partagée par groupe au plus
groupshared uint g_ballot;
groupshared uint g_lanes[32];
groupshared uint g_owner[32];

uint Ballot(uint lane, bool bit) {
  GroupMemoryBarrierWithGroupSync();
  if (lane == 0) g_ballot = 0;
  GroupMemoryBarrierWithGroupSync();
  if (bit) InterlockedOr(g_ballot, 1u << lane);
  GroupMemoryBarrierWithGroupSync();
  return g_ballot;
}

uint ReadLane(uint lane, uint value, uint other) {
  GroupMemoryBarrierWithGroupSync();
  g_lanes[lane] = value;
  GroupMemoryBarrierWithGroupSync();
  return g_lanes[other & 31u];
}

uint ForwardLane(uint lane, bool active, uint value, uint target) {
  // ds_permute_b32 : la ligne active de plus grand numéro qui vise une ligne lui donne sa valeur.
  target &= 31u;
  GroupMemoryBarrierWithGroupSync();
  g_owner[lane] = 0u;
  GroupMemoryBarrierWithGroupSync();
  if (active) InterlockedMax(g_owner[target], lane + 1u);
  GroupMemoryBarrierWithGroupSync();
  if (active && g_owner[target] == lane + 1u) g_lanes[target] = value;
  GroupMemoryBarrierWithGroupSync();
  return g_owner[lane] != 0u ? g_lanes[lane] : 0u;
}

[numthreads(32, 1, 1)]
void main(uint lane : SV_GroupIndex) {
  uint s[128];
  uint v[256];
  uint s_null = 0u;
  bool scc = false;
  // Dérouler l'initialisation scalaire évite un résultat incorrect sur WARP (#142).
  [unroll] for (uint i = 0; i < 128; ++i) s[i] = 0u;
  for (uint j = 0; j < 256; ++j) v[j] = 0u;
  v[0] = lane;
  s[126] = 0xFFFFFFFFu;
  // La mémoire groupshared n'est pas remise à zéro par D3D12 : on la vide, comme l'interpréteur.
  for (uint w = lane; w < 8128u; w += 32u) g_lds[w] = 0u;
  GroupMemoryBarrierWithGroupSync();
"""


def _words(data, instruction):
    first = struct.unpack_from("<I", data, instruction.offset)[0]
    pair = int.from_bytes(data[instruction.offset:instruction.offset + 8].ljust(8, b"\0"), "little")
    return first, pair


def _literal(data, instruction, base_words):
    if instruction.size > base_words:
        return struct.unpack_from("<I", data, instruction.offset + base_words * rdna2.DWORD)[0]
    return None


def _fields(family, word):
    return {name: (word >> shift) & mask for name, (shift, mask, _) in FIELDS[family].items()}


def branch_target(data, instruction):
    simm16 = struct.unpack_from("<I", data, instruction.offset)[0] & 0xFFFF
    return instruction.offset + rdna2.DWORD + signed(simm16, 16) * rdna2.DWORD


def _vector(name, fields, literal, active=ACTIVE):
    if name in COMPAREF16:
        target = "s[126]" if name.startswith("v_cmpx") else "s[106]"
        condition = rdna2_hlsl_cmpf16.condition(name, *rdna2_hlsl_cmpf16.vopc_sources(name, fields, literal))
        return [f"{{ bool t = ({active}) && ({condition}); {target} = Ballot(lane, t); }}"]
    a = source(fields["src0"], literal)
    values = {"a": a, "b": f"v[{fields['vsrc1']}]" if "vsrc1" in fields else "0u", "c": "s[106]",
              "d": f"v[{fields.get('vdst', 0)}]", "k": hexa(literal or 0)}
    if name == "v_readfirstlane_b32":
        return [f"s[{fields['vdst']}] = ReadLane(lane, {a}, firstbitlow(s[126]));"]
    if name in VECTOR_NO_OPERATION:
        return []
    if name in rdna2_exec_wide.OPERATIONS:
        return rdna2_hlsl_wide.statements(name, (fields["src0"],), fields["vdst"], active)
    if name in rdna2_hlsl_movrel.VECTOR:
        return rdna2_hlsl_movrel.vector(name, fields["vdst"], fields["src0"], literal, active)
    if name in CARRY:                       # forme VOP2 : retenue lue et écrite dans VCC
        return _carry(name, a, values["b"], "s[106]", fields["vdst"], "s[106]")
    if is_compare(name):
        target = "s[126]" if name.startswith("v_cmpx") else "s[106]"
        condition = compare_condition(name, a, values["b"])
        return [f"{{ bool t = ({active}) && ({condition}); {target} = Ballot(lane, t); }}"]
    if name not in VECTOR:
        raise Untranslatable(f"non traduite en HLSL : {name}")
    return [f"if ({active}) v[{fields['vdst']}] = {VECTOR[name].format(**values)};"]


def _carry(name, a, b, carry_in, vdst, carry_out, clamp=0):
    result, carry = CARRY[name]
    return [f"{{ uint x = {a}, y = {b}, k = ({carry_in} >> lane) & 1u, r = 0u; bool o = false; "
            f"if ({ACTIVE}) {{ r = {result}; o = {carry}; v[{vdst}] = {carry_write(name, clamp)}; }} "
            f"{carry_out} = Ballot(lane, o); }}"]


def _mad64(name, a, b, fields):
    """S0 × S1 + S2 sur 64 bits ; v_mad_i64_i32 corrige le produit non signé par les signes."""
    code = fields.sources[2]
    refuse_integer_clamp(name, fields.clamp)
    if code == 255:
        raise Untranslatable(f"non traduite en HLSL : {name} avec littéral 64 bits")
    low, high = source(code), source_high(code)
    if name == "v_mad_i64_i32":
        if fields.sdst != NULL:
            raise Untranslatable("v_mad_i64_i32 : retenue signée dans SDST non vérifiée")
        product_high = "MulHiI32(x, y)"
    else:
        product_high = "MulHiU32(x, y)"
    return [f"{{ uint x = {a}, y = {b}; uint pl = x * y, ph = {product_high}; uint rl = pl + {low}; "
            f"uint rh = ph + {high}; bool o = rh < ph; uint rh2 = rh + (rl < pl ? 1u : 0u); o = o || rh2 < rh; "
            f"if ({ACTIVE}) {{ v[{fields.vdst}] = rl; v[{fields.vdst + 1}] = rh2; }} "
            f"{scalar_target(fields.sdst)} = Ballot(lane, ({ACTIVE}) && o); }}"]


def modified(code, index, fields, literal, is_float):
    operand = source(code, literal)
    if is_float and fields.abs >> index & 1:
        operand = f"({operand} & 0x7FFFFFFFu)"
    if is_float and fields.neg >> index & 1:
        operand = f"({operand} ^ 0x80000000u)"
    return operand


def _vop3(name, opcode, fields, literal, active=ACTIVE):
    if name in rdna2_exec_pkf16.PK_FMAC:
        return rdna2_hlsl_pkf16.pk_fmac_vop3(name, fields, literal, active)
    if name in rdna2_hlsl_f16conv.TRANSLATED:
        return rdna2_hlsl_f16conv.vop3(name, fields, literal, active)
    if name in rdna2_hlsl_f16.TEMPLATES:
        return rdna2_hlsl_f16.vop3(name, fields, literal, active)
    if name in COMPAREF16:
        target = "s[126]" if name.startswith("v_cmpx") else scalar_target(fields.vdst)
        condition = rdna2_hlsl_cmpf16.condition(name, *rdna2_hlsl_cmpf16.vop3_sources(name, fields, literal))
        return [f"{{ bool t = ({active}) && ({condition}); {target} = Ballot(lane, t); }}"]
    if name in rdna2_hlsl_int16.TRANSLATED:
        return rdna2_hlsl_int16.vop3(name, fields, literal, active)
    is_float = float_result(name)
    a, b, c = (modified(code, i, fields, literal, float_inputs(name)) for i, code in enumerate(fields.sources))
    if opcode < 0x100 and is_compare(name):
        rdna2_exec_cmp16.check_vop3(name, fields.op_sel, Untranslatable)
        target = "s[126]" if name.startswith("v_cmpx") else scalar_target(fields.vdst)
        return [f"{{ bool t = ({active}) && ({compare_condition(name, a, b)}); {target} = Ballot(lane, t); }}"]
    if name in CARRY:
        return _carry(name, a, b, c, fields.vdst, scalar_target(fields.sdst), fields.clamp)
    if name == "v_readlane_b32":
        return [f"{scalar_target(fields.vdst)} = ReadLane(lane, {a}, {b});"]
    if name == "v_readfirstlane_b32":
        return [f"{scalar_target(fields.vdst)} = ReadLane(lane, {a}, firstbitlow(s[126]));"]
    if name in VECTOR_NO_OPERATION:
        return []
    if name in MAD64:
        return _mad64(name, a, b, fields)
    if name in rdna2_exec_wide.OPERATIONS:
        return rdna2_hlsl_wide.statements(name, fields.sources, fields.vdst, active, fields.abs, fields.neg,
                                          fields.clamp, fields.omod)
    if name in rdna2_exec_div.OPERATIONS:
        return rdna2_hlsl_div.statements(name, fields, literal, active)
    if name in rdna2_hlsl_movrel.VECTOR:
        if fields.abs or fields.neg or fields.clamp or fields.omod:
            raise Untranslatable(f"non traduite en HLSL : {name} avec modificateurs VOP3")
        return rdna2_hlsl_movrel.vector(name, fields.vdst, fields.sources[0], literal, active)
    if name == "v_writelane_b32":
        return [f"if (lane == ({b} & 31u)) v[{fields.vdst}] = {a};"]
    if name in rdna2_hlsl_sad.TRANSLATED:
        return rdna2_hlsl_sad.statements(name, fields, literal, active)
    if name not in VECTOR:
        raise Untranslatable(f"non traduite en HLSL : {name} (VOP3)")
    if name in SIXTEEN_BIT_RESULTS and (fields.op_sel or fields.clamp):
        raise Untranslatable(f"non traduite en HLSL : {name} avec op_sel/clamp (VOP3, résultat 16 bits)")
    if name in LITERAL_VOP2 and literal is None:
        raise Untranslatable(f"non traduite en HLSL : {name} en VOP3 sans littéral")
    template = VECTOR[name] if is_float else integer_template(name, fields.clamp, VECTOR[name])
    expression = template.format(a=a, b=b, c=c, d=f"v[{fields.vdst}]", k=hexa(literal or 0))
    if is_float and (fields.clamp or fields.omod):
        scaled = f"asfloat({expression}) * {OUTPUT_SCALE.get(fields.omod, '1.0f')}"
        expression = f"asuint({'saturate(' + scaled + ')' if fields.clamp else scaled})"
    return [f"if ({active}) v[{fields.vdst}] = {expression};"]


def statements(data, instruction, finish="return;", special=None, active=ACTIVE):
    """Instructions HLSL d'une instruction RDNA 2 (hors branchements).

    `finish` : texte de s_endpgm ; `special(famille, nom, mot 64 bits)` traduit les familles
    propres à un étage graphique (VINTRP, EXP) et renvoie None pour les autres ; `active` : condition
    HLSL « la ligne est active » (emu/rdna2_hlsl_wave64.py en donne une sur 64 lignes)."""
    family, opcode = instruction.family, instruction.opcode
    name = rdna2.mnemonic(family, opcode) or f"{family}#{opcode}"
    first, pair = _words(data, instruction)
    if name == "s_endpgm":
        return [finish]
    translated = special(family, name, pair) if special else None
    if translated is not None:
        return translated
    if family == "SOP1" and name in rdna2_hlsl_movrel.SCALAR:
        return rdna2_hlsl_movrel.scalar(name, _fields(family, first), _literal(data, instruction, 1))
    if family in SCALAR_FAMILIES:
        if name in rdna2_hlsl_pc.NAMES:
            return rdna2_hlsl_pc.statements(name, _fields(family, first), instruction)
        if name in rdna2_hlsl_scalar64.SCALAR64:
            return [rdna2_hlsl_scalar64.statement(name, _fields(family, first))]
        if name not in SCALAR:
            raise Untranslatable(f"non traduite en HLSL : {name}")
        line = scalar_statement(name, _fields(family, first), _literal(data, instruction, 1))
        return [line] if line else []
    if family in ("VOP1", "VOP2", "VOPC"):
        fields = _fields(family, first)
        lit = _literal(data, instruction, 1)
        if name in rdna2_exec_pkf16.PK_FMAC and family == "VOP2":
            return rdna2_hlsl_pkf16.pk_fmac_vop2(name, fields, lit, active)
        if name in rdna2_hlsl_f16conv.TRANSLATED:   # before is_ext: SDWA/DPP forms refused by rdna2_hlsl_f16conv
            return rdna2_hlsl_f16conv.vector(name, fields, lit, active)
        if name in rdna2_hlsl_f16.TEMPLATES:        # avant is_ext : les formes SDWA/DPP du f16 sont refusées par rdna2_hlsl_f16
            return rdna2_hlsl_f16.vector(name, fields, lit, active)
        if name in COMPAREF16:                      # before is_ext: SDWA/DPP refused by rdna2_hlsl_cmpf16
            return _vector(name, fields, lit, active)
        if rdna2_hlsl_ext.is_ext(family, fields.get("src0")):
            return rdna2_hlsl_ext.statements(family, name, fields, lit or 0, active)
        return _vector(name, fields, lit, active)
    if family == "VOP3":
        return _vop3(name, opcode, Fields(pair), _literal(data, instruction, 2), active)
    if family == "VOP3P":
        if name in rdna2_hlsl_int16.TRANSLATED:
            return rdna2_hlsl_int16.packed(name, pair, _literal(data, instruction, 2), active)
        if name in rdna2_exec_pkf16.VOP3P:
            return rdna2_hlsl_pkf16.vop3p(name, pair, _literal(data, instruction, 2), active)
        return rdna2_hlsl_dot.statements(name, pair, _literal(data, instruction, 2), active)
    if family in MEMORY_FAMILIES:
        return memory_statements(family, name, pair, active)
    if rdna2_hlsl_gather.supports(family, name):
        end = instruction.offset + instruction.size * rdna2.DWORD
        return rdna2_hlsl_gather.statements(name, int.from_bytes(data[instruction.offset:end], "little"), active)
    if family == "MIMG":
        end = instruction.offset + instruction.size * rdna2.DWORD
        return rdna2_hlsl_image.statements(name, int.from_bytes(data[instruction.offset:end], "little"), active)
    raise Untranslatable(f"non traduite en HLSL : {name} ({family})")


def _leaders(data, instructions):
    leaders = {0}
    for index, instruction in enumerate(instructions):
        name = rdna2.mnemonic(instruction.family, instruction.opcode)
        if name in BRANCHES or name == "s_endpgm":
            leaders.add(branch_target(data, instruction) if name in BRANCHES else 0)
            if index + 1 < len(instructions):
                leaders.add(instructions[index + 1].offset)
    return sorted(leaders)


def translate(data, sgprs=None, wave_size=32):
    """Texte HLSL d'un programme RDNA 2 ; `sgprs` : valeurs initiales {numéro: valeur}.

    `wave_size` : 32 (un groupe de 32 threads) ou 64 (programme wave64, voir emu/rdna2_hlsl_wave64.py)."""
    if wave_size not in WAVE_SIZES:
        raise ValueError(f"taille de wave non prise en charge : {wave_size} (32 ou 64)")
    if wave_size == 64:
        from emu import rdna2_hlsl_wave64
        return rdna2_hlsl_wave64.translate(data, sgprs)
    helpers = "\n".join((ALU_HELPERS, rdna2_hlsl_wide.HELPERS, rdna2_hlsl_div.HELPERS, rdna2_hlsl_image.HELPERS,
                         rdna2_hlsl_format.HELPERS, rdna2_hlsl_dot.HELPERS, rdna2_hlsl_sad.HELPERS,
                         rdna2_hlsl_f16.HELPERS, rdna2_hlsl_cmpf16.HELPERS, rdna2_hlsl_int16.HELPERS,
                         rdna2_hlsl_pkf16.HELPERS,
                         rdna2_hlsl_gather.HELPERS,
                         rdna2_hlsl_f16conv.HELPERS))
    prelude = PRELUDE.replace("[numthreads", helpers + "\n[numthreads", 1)
    return "\n".join([prelude.rstrip("\n"), *body(data, sgprs), "}", ""])


def body(data, sgprs=None, finish="return;", special=None, branches=BRANCHES, translate_one=None):
    """Lignes du corps : SGPR initiaux, puis les instructions en ligne droite si le programme n'a qu'un
    bloc de base, sinon une boucle `while` et une garde `if (pc == ...)` par bloc de base : chaque bloc
    se termine par `continue` vers le suivant, par un branchement ou par `finish`.

    `branches` (conditions HLSL des branchements) et `translate_one` (traduction d'une instruction,
    `statements` par défaut) permettent au wave64 de remplacer ce qui dépend de la taille du wave."""
    translate_one = translate_one or statements
    instructions = [i for i in rdna2.decode(data) if i.family != "tronquée"]
    leaders = _leaders(data, instructions)
    lines = [f"  s[{number}] = {hexa(value)};" for number, value in sorted((sgprs or {}).items())]
    names = [rdna2.mnemonic(instruction.family, instruction.opcode) for instruction in instructions]
    if leaders == [0] and not any(name in branches for name in names):
        for instruction in instructions:
            lines += [f"  {line}" for line in translate_one(data, instruction, finish, special)]
        return lines
    # Cible permise : le début d'une instruction, ou la fin du programme (qui le termine).
    end = instructions[-1].offset + instructions[-1].size * rdna2.DWORD if instructions else 0
    targets = {instruction.offset for instruction in instructions} | {end}
    lines += ["  uint pc = 0;", "  [loop] while (pc != END_PC) {"]
    for index, (instruction, name) in enumerate(zip(instructions, names)):
        if instruction.offset in leaders:
            if index:
                lines.append("    }")
            lines.append(f"    if (pc == {instruction.offset}u) {{")
        if name in branches:
            target = branch_target(data, instruction)
            if target not in targets:
                raise Untranslatable(f"cible de branchement invalide : {target}")
            lines.append(f"      if ({branches[name]}) {{ pc = {target}u; continue; }}")
        else:
            lines += [f"      {line}" for line in translate_one(data, instruction, finish, special)]
        following = instructions[index + 1].offset if index + 1 < len(instructions) else None
        if following in leaders and name != "s_endpgm":
            lines.append(f"      pc = {following}u; continue;")
    return lines + ["    }", f"    {finish}", "  }"]


def coverage_names():
    """Mnémoniques que le recompilateur sait traduire (hors mémoire, comptée à part)."""
    return (set(SCALAR) | set(VECTOR) | set(BRANCHES) | set(CARRY) | VECTOR_NO_OPERATION | MAD64
            | set(rdna2_hlsl_scalar64.SCALAR64)
            | rdna2_hlsl_wide.TRANSLATED | rdna2_hlsl_div.TRANSLATED | rdna2_hlsl_movrel.TRANSLATED
            | rdna2_hlsl_dot.TRANSLATED | rdna2_hlsl_sad.TRANSLATED | rdna2_hlsl_pc.TRANSLATED
            | rdna2_hlsl_int16.TRANSLATED | rdna2_hlsl_pkf16.TRANSLATED
            | set(rdna2_hlsl_f16.TEMPLATES) | rdna2_hlsl_f16conv.TRANSLATED
            | {"s_endpgm", "v_readlane_b32", "v_writelane_b32", "v_readfirstlane_b32"})


__all__ = ["translate", "Untranslatable"]
