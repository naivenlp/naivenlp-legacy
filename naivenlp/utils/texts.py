import re


def char_half2full(uchar):
    """Convert characters from half-width to full-width."""
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e:  # 不是半角字符就返回原来的字符
        return uchar
    if inside_code == 0x0020:  # 除了空格其他的全角半角的公式为:半角=全角-0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return chr(inside_code)


def string_half2full(s):
    return "".join([char_half2full(c) for c in s])


def b2q(s):
    """Convert character(s) from half-width to full-width."""
    if not s:
        return ""
    if len(s) == 1:
        return char_half2full(s)
    return string_half2full(s)


def char_full2half(uchar):
    """Convert characters from full-width to half-width."""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)


def string_full2half(ustring):
    """Convert string from full-width to half-width."""
    return "".join([char_full2half(uchar) for uchar in ustring])


def q2b(s):
    """Convert character(s) from full-width to half-width."""
    if not s:
        return ""
    if len(s) == 1:
        return char_full2half(s)
    return string_full2half(s)


def split_sentence(text, reg=None):
    """Split long text to a list of sentence."""
    p = reg if reg is not None else '[\\n\\t\\s；。：;]+'
    return re.split(p, text)
