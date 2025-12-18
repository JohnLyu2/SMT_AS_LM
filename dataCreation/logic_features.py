"""
SMT Catalog Feature Selection by Logic Type

This script provides logic-specific feature filtering for SMT Catalog provided features comprehensive
feature set (~204 operators) + 11 commands. It maps SMT-LIB logics to their relevant operators.
"""

# Command-level features (tracked as counts)
command_keywords = [
    'assert', 'declare-fun', 'declare-const', 'declare-sort',
    'define-fun', 'define-fun-rec', 'define-funs-rec', 'define-sort',
    'declare-datatype', 'declare-datatypes',
]

# Quantifiers and Binders
binders = [
    'exists', 'forall', 'let',
]

# Core Theory (Boolean Logic)
core_theory = [
    'true', 'false', 'Bool', 'ite', 'not', 'or', 'and', '=>', 'xor',
    '=', 'distinct', 'const',
]

# Arrays Theory
arrays_theory = [
    'Array', 'select', 'store', 'store_all', 'eqrange',
]

# Bit-Vectors Theory (Extended - Catalog Database has more BV operators)
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

# Floating-Point Theory (Extended - Catalog Database has total variants)
floating_point_theory = [
    'FloatingPoint', 'RoundingMode',
    'fp',
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

# Arithmetic Theory (Extended - Catalog Database has transcendentals)
arithmetic_theory = [
    'Int', 'Real',
    'div', 'mod', 'divisible', 'iand', 'int.pow2',
    'div_total', 'mod_total',
    '/', '/_total',
    '+', '-', '*',
    '<', '<=', '>', '>=',
    'to_real', 'to_int', 'is_int',
    'abs', '^',
    # Transcendental functions (nonlinear arithmetic)
    'real.pi', 'exp', 'sin', 'cos', 'tan', 'csc', 'sec', 'cot',
    'arcsin', 'arccos', 'arctan', 'arccsc', 'arcsec', 'arccot',
    'sqrt',
]

# Strings and Regular Expressions Theory (Extended)
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

# Sequences Theory (Catalog Database specific)
sequences_theory = [
    'seq.empty', 'seq.unit', 'seq.nth', 'seq.len',
]

# Uninterpreted Functions
uf_theory = [
    # UF doesn't have specific operators beyond core theory
]

# Metadata features (structural metrics, always included regardless of logic)
metadata_features = [
    'maxTermDepth', 'normalizedSize',
]

# Logic mapping to relevant theories
logic_to_theories = {
    # Quantifier-Free Logics
    'QF_UF': [uf_theory, core_theory],
    'QF_BV': [bitvectors_theory, core_theory],
    'QF_IDL': [arithmetic_theory, core_theory],
    'QF_RDL': [arithmetic_theory, core_theory],
    'QF_LIA': [arithmetic_theory, core_theory],
    'QF_LRA': [arithmetic_theory, core_theory],
    'QF_NIA': [arithmetic_theory, core_theory],
    'QF_NRA': [arithmetic_theory, core_theory],
    'QF_AX': [arrays_theory, core_theory],
    
    # Combinations of Quantifier-Free Logics
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
    
    # Quantified Logics
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
    
    # Floating-point logics
    'QF_FP': [floating_point_theory, core_theory],
    'QF_FPBV': [floating_point_theory, bitvectors_theory, core_theory],
    'QF_UFFP': [uf_theory, floating_point_theory, core_theory],
    
    # String logics
    'QF_S': [strings_regex_theory, core_theory],
    'QF_SLIA': [strings_regex_theory, arithmetic_theory, core_theory],
    
    # All theories (for unknown logics or ALL)
    'ALL': [core_theory, arrays_theory, bitvectors_theory, floating_point_theory,
            arithmetic_theory, strings_regex_theory, sequences_theory, uf_theory, binders],
}


def get_relevant_features(logic, include_commands=True, include_metadata=True):
    """
    Get the list of relevant features for Catalog Database based on the SMT logic.

    Args:
        logic (str): The SMT logic (e.g., 'QF_BV', 'AUFLIA', 'ALL')
        include_commands (bool): Whether to include command-level features
        include_metadata (bool): Whether to include metadata features (maxTermDepth, normalizedSize)

    Returns:
        list: Combined list of all relevant features for the logic
    """
    relevant_features = []

    # Always include command keywords if requested
    if include_commands:
        relevant_features.extend(command_keywords)

    # Add theory-specific operators based on the logic
    if logic in logic_to_theories:
        theories = logic_to_theories[logic]
    else:
        # If logic is unknown, use all theories
        print(f"Warning: Unknown logic '{logic}', using all features")
        theories = logic_to_theories['ALL']

    # Collect all features from relevant theories
    seen = set(relevant_features)  # Track seen features to avoid duplicates
    for theory in theories:
        for feature in theory:
            if feature not in seen:
                relevant_features.append(feature)
                seen.add(feature)

    # Add metadata features if requested
    if include_metadata:
        for feature in metadata_features:
            if feature not in seen:
                relevant_features.append(feature)
                seen.add(feature)

    return relevant_features


def print_relevant_features(logic, include_commands=True, format='list'):
    """
    Print relevant features for a given logic in specified format.
    
    Args:
        logic (str): The SMT logic
        include_commands (bool): Whether to include command-level features
        format (str): Output format - 'list' or 'numbered'
    """
    features = get_relevant_features(logic, include_commands)
    
    if format == 'list':
        print(f"\nRelevant features for {logic}:")
        print(", ".join(features))
    elif format == 'numbered':
        print(f"\nRelevant features for {logic} (numbered):")
        for i, feature in enumerate(features, 1):
            print(f"{i}|{feature}")
    
    print(f"\nTotal features: {len(features)}")


# Example usage
if __name__ == "__main__":
    # Test with different logics
    test_logics = ['QF_BV', 'QF_AUFLIA', 'NRA', 'QF_FP', 'QF_S', 'ALL']
    
    for logic in test_logics:
        print("\n" + "="*60)
        print_relevant_features(logic, include_commands=True, format='numbered')
        print("="*60)