import re
import string

SEPARATOR_CHAR_SET = set(string.whitespace + string.punctuation)

def count_tokens(string):
    previous_sep = False
    separator_count = 0
    for char in string:
        if char in SEPARATOR_CHAR_SET:
            if previous_sep:
                continue
            previous_sep = True
            separator_count += 1
        else:
            previous_sep = False
    # Do not count trailing separator
    if previous_sep:
        separator_count -= 1
    return separator_count + 1


def filter_by_num_tokens(string, minimum, maximum):
    num_tokens = count_tokens(string)
    return minimum <= num_tokens and num_tokens <= maximum


def deitalize(string):
    return string.replace('_', '')


def reduce_whitespace(string):
    return re.sub('\s+', ' ', string)


def normalize_text(string):
    if string is None:
        return ''
    replaced_refs = string.replace('_USER_REFERENCE_', '[NAME]')
    replaced_urls = replaced_refs.replace('_URL_/', '[URL]').replace('_URL_', '[URL]')
    undo_italics = deitalize(replaced_urls)
    parse_whitespace = reduce_whitespace(undo_italics).strip()
    return parse_whitespace
