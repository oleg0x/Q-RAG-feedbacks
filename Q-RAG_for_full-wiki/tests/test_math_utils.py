from utils.process_latex import (
    extract_boxed_expression,
    process_latex_for_cmp,
    simplify_latex,
)


def test_boxed_answer_helpers():
    assert extract_boxed_expression(r"work \\boxed{\\frac{1}{2}}") == r"\\frac{1}{2}"
    assert simplify_latex(r" \\dfrac{1}{2} ") == r"\\frac12"
    assert process_latex_for_cmp(r"\\boxed{\\sqrt{2}}") == r"\\sqrt2"
