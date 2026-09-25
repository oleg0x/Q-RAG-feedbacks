"""Helpers for extracting and normalizing final answers from MATH outputs."""

import re


def extract_boxed_expression(text: str) -> str:
    """Return the contents of the last ``\\boxed`` expression, if present."""
    idx = text.rfind(r"\boxed")
    if idx == -1:
        return ""
    remaining = text[idx + len(r"\boxed") :].lstrip()
    if not remaining:
        return ""
    if remaining[0] != "{":
        return remaining[0]

    depth = 1
    pos = 1
    while pos < len(remaining) and depth:
        if remaining[pos] == "{":
            depth += 1
        elif remaining[pos] == "}":
            depth -= 1
        pos += 1
    return remaining[1 : pos - 1] if depth == 0 else ""


def remove_sqrt_braces(text: str) -> str:
    return re.sub(r"\\sqrt\{(\d)\}", r"\\sqrt\1", text)


def simplify_frac(latex: str) -> str:
    """Canonicalize frac/dfrac/tfrac and remove braces around one-digit args."""
    command = re.compile(r"\\(?:frac|dfrac|tfrac)(?![a-zA-Z])")

    def parse_group(value: str, start: int):
        depth = 1
        pos = start + 1
        while pos < len(value) and depth:
            depth += (value[pos] == "{") - (value[pos] == "}")
            pos += 1
        return value[start + 1 : pos - 1], pos

    def parse_arg(value: str, start: int):
        while start < len(value) and value[start].isspace():
            start += 1
        if start >= len(value):
            return None, False, start
        if value[start] == "{":
            parsed, end = parse_group(value, start)
            return parsed, True, end
        if value[start] == "\\":
            end = start + 1
            while end < len(value) and value[end].isalpha():
                end += 1
            return value[start:end], False, end
        return value[start], False, start + 1

    result = []
    cursor = 0
    while True:
        match = command.search(latex, cursor)
        if match is None:
            result.append(latex[cursor:])
            break
        result.append(latex[cursor : match.start()])
        numerator, numerator_braced, pos = parse_arg(latex, match.end())
        denominator, denominator_braced, end = parse_arg(latex, pos)
        if numerator is None or denominator is None:
            result.append(latex[match.start() : end])
            cursor = end
            continue
        numerator = simplify_frac(numerator) if numerator_braced else numerator
        denominator = simplify_frac(denominator) if denominator_braced else denominator
        if numerator_braced and not (len(numerator) == 1 and numerator.isdigit()):
            numerator = "{" + numerator + "}"
        if denominator_braced and not (len(denominator) == 1 and denominator.isdigit()):
            denominator = "{" + denominator + "}"
        result.append(r"\frac" + numerator + denominator)
        cursor = end
    return "".join(result)


def clean_latex_expression(text: str) -> str:
    text = text.replace(" ", "").replace(r"\left", "").replace(r"\right", "")
    return re.sub(r"\\text\{([^}]*)\}", r"\1", text)


def simplify_latex(text: str) -> str:
    return clean_latex_expression(simplify_frac(remove_sqrt_braces(text)))


def process_latex_for_cmp(text: str) -> str:
    return simplify_latex(extract_boxed_expression(text))
