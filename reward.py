import re
import tiktoken

BOX_PATTERN = re.compile(r'\\box\{([^}]*)\}')

STEP_BLOCKS = [
    re.compile(r"Step\s*1\s*—\s*Section.*?Brief Justification:\s*(\S.*)?\n.*?Decision:\s*\\box\{[^}]*\}", re.S),
    re.compile(r"Step\s*2\s*—\s*Class.*?Brief Justification:\s*(\S.*)?\n.*?Decision:\s*\\box\{[^}]*\}", re.S),
    re.compile(r"Step\s*3\s*—\s*Subclass.*?Brief Justification:\s*(\S.*)?\n.*?Decision:\s*\\box\{[^}]*\}", re.S),
]

def extract_solution(solution_str):
    """Extract up to three \box{...} values; pad with None if fewer than 3."""
    items = [m.strip() for m in BOX_PATTERN.findall(solution_str or "")]
    items = items[:3] + [None] * max(0, 3 - len(items))
    return items

def token_count(text, enc_name="cl100k_base"):
    enc = tiktoken.get_encoding(enc_name)
    return len(enc.encode(text or ""))

def length_reward(n_tok, lo=100, hi=400, hard_lo=10, hard_hi=512,
                  bonus=0.00, w_short=0.1, w_long=0.1, cap_pen=0.0):
    if n_tok < lo:
        return -w_short * (lo - n_tok) / lo
    if n_tok <= hi:
        return bonus
    over = min(n_tok, hard_hi) - hi
    span = max(1, hard_hi - hi)
    r = -w_long * (over / span)
    if n_tok >= hard_hi or n_tok <= hard_lo:
        r -= cap_pen
    return r


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    ans = extract_solution(solution_str)
    r = 0.0

    # —— main——
    if ans[2] and ans[2] == ground_truth[:4]:
        r += 1.0

    # —— length shaping ——
    n_tok = token_count(solution_str)
    r += length_reward(n_tok)

    return r
