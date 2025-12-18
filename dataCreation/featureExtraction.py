import re
from collections import Counter

# -------------------------------------------------
# Your theory/logic definitions (unchanged)
# -------------------------------------------------

# Command-level features (tracked as counts)
command_keywords = [
    'assert', 'declare-fun', 'declare-const', 'declare-sort',
    'define-fun', 'define-fun-rec', 'define-funs-rec', 'define-sort',
    'declare-datatype', 'declare-datatypes',
]

binders = ['exists', 'forall', 'let']

core_theory = [
    'true', 'false', 'Bool', 'ite', 'not', 'or', 'and', '=>', 'xor',
    '=', 'distinct', 'const',
]

arrays_theory = [
    'Array', 'select', 'store', 'store_all', 'eqrange',
]

bitvectors_theory = [
    'BitVec', 'bvempty', 'concat', 'extract', 'repeat',
    'bvnot', 'bvand', 'bvor', 'bvnand', 'bvnor', 'bvxor', 'bvxnor', 'bvcomp',
    'bvneg', 'bvadd', 'bvmul', 'bvudiv', 'bvurem', 'bvsub', 'bvsdiv', 'bvsrem', 'bvsmod',
    'bvult', 'bvule', 'bvugt', 'bvuge', 'bvslt', 'bvsle', 'bvsgt', 'bvsge',
    'bvshl', 'bvlshr', 'bvashr',
    'zero_extend', 'sign_extend', 'rotate_left', 'rotate_right',
    'reduce_and', 'reduce_or', 'reduce_xor',
    'bvite', 'bv1ult', 'bitOf',
    'bvuaddo', 'bvsaddo', 'bvumulo', 'bvsmulo', 'bvusubo', 'bvssubo', 'bvsdivo',
    'bvultbv', 'bvsltbv', 'bvredand', 'bvredor',
    'int2bv', 'bv2nat', 'bvsize',
]

floating_point_theory = [
    'FloatingPoint', 'RoundingMode', 'fp',
    'fp.add', 'fp.sub', 'fp.mul', 'fp.div', 'fp.fma', 'fp.sqrt', 'fp.rem',
    'fp.roundToIntegral', 'fp.min', 'fp.max', 'fp.abs', 'fp.neg',
    'fp.leq', 'fp.lt', 'fp.geq', 'fp.gt', 'fp.eq',
    'fp.isNormal', 'fp.isSubnormal', 'fp.isZero', 'fp.isInfinite', 'fp.isNaN',
    'fp.isPositive', 'fp.isNegative',
    'roundNearestTiesToEven', 'roundNearestTiesToAway', 'roundTowardPositive',
    'roundTowardNegative', 'roundTowardZero',
    'fp.to_ubv', 'fp.to_ubv_total', 'fp.to_sbv', 'fp.to_sbv_total', 'fp.to_real',
    'to_fp', 'to_fp_unsigned', 'to_fp_bv',
]

arithmetic_theory = [
    'Int', 'Real', 'div', 'mod', 'divisible', 'iand', 'int.pow2',
    'div_total', 'mod_total',
    '/', '/_total', '+', '-', '*',
    '<', '<=', '>', '>=',
    'to_real', 'to_int', 'is_int', 'abs', '^',
    'real.pi', 'exp', 'sin', 'cos', 'tan', 'csc', 'sec', 'cot',
    'arcsin', 'arccos', 'arctan', 'arccsc', 'arcsec', 'arccot',
    'sqrt',
]

strings_regex_theory = [
    'String', 'Char', 'RegLan',
    'str.len', 'str.++', 'str.substr', 'str.contains', 'str.replace',
    'str.indexof', 'str.at', 'str.prefixof', 'str.suffixof',
    'str.rev', 'str.unit', 'str.update',
    'str.to_lower', 'str.to_upper',
    'str.to_code', 'str.from_code', 'str.is_digit',
    'str.to_int', 'str.from_int',
    'str.<', 'str.<=',
    'str.replace_all', 'str.replace_re', 'str.replace_re_all', 'str.indexof_re',
    're.allchar', 're.none', 're.all', 're.empty', 'str.to_re',
    're.*', 're.+', 're.opt', 're.comp', 're.range', 're.++',
    're.inter', 're.union', 're.diff', 're.loop',
    'str.in_re',
]

sequences_theory = [
    'seq.empty', 'seq.unit', 'seq.nth', 'seq.len',
]

uf_theory = []

metadata_features = ['maxTermDepth', 'normalizedSize']

logic_to_theories = {
    'QF_UF': [uf_theory, core_theory],
    'QF_BV': [bitvectors_theory, core_theory],
    'QF_IDL': [arithmetic_theory, core_theory],
    'QF_RDL': [arithmetic_theory, core_theory],
    'QF_LIA': [arithmetic_theory, core_theory],
    'QF_LRA': [arithmetic_theory, core_theory],
    'QF_NIA': [arithmetic_theory, core_theory],
    'QF_NRA': [arithmetic_theory, core_theory],
    'QF_AX': [arrays_theory, core_theory],
    'QF_UFIDL': [uf_theory, arithmetic_theory, core_theory],
    'QF_UFBV': [uf_theory, bitvectors_theory, core_theory],
    'QF_UFLIA': [uf_theory, arithmetic_theory, core_theory],
    'QF_UFLRA': [uf_theory, arithmetic_theory, core_theory],
    'QF_UFNIA': [uf_theory, arithmetic_theory, core_theory],
    'QF_UFNRA': [uf_theory, arithmetic_theory, core_theory],
    'QF_ABV': [arrays_theory, bitvectors_theory, core_theory],
    'QF_ALIA': [arrays_theory, arithmetic_theory, core_theory],
    'QF_AUFBV': [arrays_theory, uf_theory, bitvectors_theory, core_theory],
    'QF_AUFLIA': [arrays_theory, uf_theory, arithmetic_theory, core_theory],
    'QF_AUFNIA': [arrays_theory, uf_theory, arithmetic_theory, core_theory],
    'LIA': [arithmetic_theory, core_theory, binders],
    'LRA': [arithmetic_theory, core_theory, binders],
    'NIA': [arithmetic_theory, core_theory, binders],
    'NRA': [arithmetic_theory, core_theory, binders],
    'UFLRA': [uf_theory, arithmetic_theory, core_theory, binders],
    'UFNIA': [uf_theory, arithmetic_theory, core_theory, binders],
    'ALIA': [arrays_theory, arithmetic_theory, core_theory, binders],
    'AUFLIA': [arrays_theory, uf_theory, arithmetic_theory, core_theory, binders],
    'AUFLIRA': [arrays_theory, uf_theory, arithmetic_theory, core_theory, binders],
    'AUFNIRA': [arrays_theory, uf_theory, arithmetic_theory, core_theory, binders],
    'BV': [bitvectors_theory, core_theory, binders],
    'UFBV': [uf_theory, bitvectors_theory, core_theory, binders],
    'ABV': [arrays_theory, bitvectors_theory, core_theory, binders],
    'AUFBV': [arrays_theory, uf_theory, bitvectors_theory, core_theory, binders],
    'QF_FP': [floating_point_theory, core_theory],
    'QF_FPBV': [floating_point_theory, bitvectors_theory, core_theory],
    'QF_UFFP': [uf_theory, floating_point_theory, core_theory],
    'QF_S': [strings_regex_theory, core_theory],
    'QF_SLIA': [strings_regex_theory, arithmetic_theory, core_theory],
    'ALL': [
        core_theory, arrays_theory, bitvectors_theory, floating_point_theory,
        arithmetic_theory, strings_regex_theory, sequences_theory, uf_theory, binders
    ],
}

# -------------------------------------------------
# Tokenization, logic detection, metrics
# -------------------------------------------------

LOGIC_RE = re.compile(r'\(set-logic\s+([A-Za-z0-9_\-+]+)\)', re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z_\.!$%&*+\-/<=>?^|~:]+|\d+|\(|\)")

def strip_comments(s: str) -> str:
    return re.sub(r";[^\n]*", "", s)

def detect_logic(s: str) -> str:
    m = LOGIC_RE.search(s)
    if not m:
        return "ALL"
    logic = m.group(1)
    return logic if logic in logic_to_theories else "ALL"

def compute_structure_metrics(s: str) -> dict:
    s = strip_comments(s)
    depth = 0
    max_depth = 0
    for c in s:
        if c == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif c == ')':
            depth -= 1
    token_count = len(TOKEN_RE.findall(s))
    return {"maxTermDepth": max_depth, "normalizedSize": token_count}

def get_logic_vocabulary(logic: str):
    voc = set()
    for group in logic_to_theories.get(logic, logic_to_theories["ALL"]):
        voc.update(group)
    voc.update(command_keywords)
    voc.update(metadata_features)
    return voc

# -------------------------------------------------
# Main extractors
# -------------------------------------------------

def extract_features_from_text(smt2_text: str, logic: str | None = None) -> dict:
    if logic is None:
        logic = detect_logic(smt2_text)

    vocab = get_logic_vocabulary(logic)
    text_no_comments = strip_comments(smt2_text)
    tokens = TOKEN_RE.findall(text_no_comments)

    counts = Counter(tok for tok in tokens if tok in vocab)

    # structural metrics
    for k, v in compute_structure_metrics(smt2_text).items():
        counts[k] = v

    # ensure all features exist
    result = {feat: int(counts.get(feat, 0)) for feat in vocab}
    result["__logic__"] = logic
    return result


def extract_features_from_path(path: str, logic: str | None = None) -> dict:
    """
    Preferred entry point:
        features = extract_features_from_path("/path/to/file.smt2")
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return extract_features_from_text(text, logic)


features = extract_features_from_path("/home/paul/PycharmProjects/StrategyPrediction/data/ijcai24/benchmarks/QF_NIA/train1/benchmark0.smt2")

for k, v in sorted(features.items()):
    print(k, v)