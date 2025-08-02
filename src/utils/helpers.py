# src/utils/helpers.py
def fmt(tpl: str, text: str, hyp: str, pred: str = "", type_field: str = "") -> str:
    return tpl.format(text=text, hypothesis=hyp, predicate=pred, type=type_field)

def extract_label(label_str: str) -> str:
    if not label_str:
        return "R"
    for ch in reversed(label_str.strip()):
        up = ch.upper()
        if up in {"T","F","U"}:
            return up
    return "R"
