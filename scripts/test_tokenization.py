# %%
import numpy as np
import random
from itertools import zip_longest
import string

from anthropic import Anthropic, AsyncAnthropic
import nest_asyncio

from src.anthropic_tokenizer import tokenize_text

nest_asyncio.apply()

an_client_async = AsyncAnthropic()
an_client_sync = Anthropic()

# tokenizers based on different models aren't the same
# haiku and opus are the same, sonnet different on: s4 didn't repeat enough unicode, s5 broke words into bigger pieces
# sometimes is different if combine ascii and unicode; for s10 haiku had extra ' ' on end but ran again and fine

s1 = (
    "We hold these truths to be self-evident, that all men are created equal, that they are"
    " endowed, by their Creator, with certain unalienable rights, that among these are life,"
    " liberty, and the pursuit of happiness."
)
s2 = """We believe AI will have a vast impact on the world. Anthropic is dedicated to building systems that people can rely on and generating research about the opportunities and risks of AI. We Build Safer Systems."""
# random unicode
s3 = """Ↄ⊙◵⳺⸬⩋ⷒ⊨⃺ⴾ❭⬫⼀⠹⍓⩁⤋┝⢫⁴⨂☍⭬➖⒛⯩ⓓ⁊⬬✟⮧⇡⍰⪤⚉⣶⃏⍮ⷕ⤽⸍⩽⤍➦➟⊞⁶⟈⬹⠤ⲵ‸◢⤽⤷∺⻊ⓨ➑⊳⍖✚☌‎⾹ⲡ✂⾠⩾ⵟ≑⎃⪆◻╾⶗ⅴ➙ⴗ▌❗◿☎ⴻ▱⍗⃌⹃⹕➨⌚⊝∆⠐ⳋ⍆⫭⃃⤽⻀"""

# words with random unicode
s4 = "".join([c for i, j in zip_longest(s1.split(" "), list(s3), fillvalue="") for c in (i, j)])
# S5: 'Ⓡ' vs  'ⓘ' vs '⊙' for opus,sonet,haiku; plus haiku change a lot of others
s5 = "".join([c for i, j in zip_longest(s2.split(" "), list(s3), fillvalue="") for c in (i, j)])

# random char words
s6 = " ".join(
    "".join(random.choice(string.ascii_letters) for _ in range(random.randint(1, 10)))
    for _ in range(20)
)
s7 = " ".join(
    "".join(random.choice(string.ascii_letters) for _ in range(random.randint(1, 10)))
    for _ in range(20)
)

# random ascii letters
s8 = "".join(random.choice(string.ascii_letters) for _ in range(100))
s9 = "".join(random.choice(string.ascii_letters) for _ in range(100))

# I don't think these are all valid unicode, but using the full valid unicode still gives errors
unicode_range = [0x0000, 0xD7FF]

# random if ascii letters or unicode; sometimes trips up
s10 = "".join(
    (
        random.choice(string.ascii_letters)
        if random.random() < 0.5
        else chr(np.random.randint(*unicode_range))
    )
    for _ in range(100)
)
s11 = "".join(
    (
        random.choice(string.ascii_letters)
        if random.random() < 0.5
        else chr(np.random.randint(*unicode_range))
    )
    for _ in range(100)
)

# random ascii
s12 = "".join(chr(random.choice(range(256))) for _ in range(100))
s13 = "".join(chr(random.choice(range(256))) for _ in range(100))


s14 = (  # all 3 slightly different, 'ᶑ' vs 'ᶁ', 'ế' vs 'ė' vs 'ć'; plus opus extra space on end
    "Ḽơᶉëᶆ ȋṕšᶙṁ ḍỡḽǭᵳ ʂǐť ӓṁệẗ, ĉṓɲṩḙċťᶒțûɾ ấɖḯƥĭṩčįɳġ ḝłįʈ, șếᶑ ᶁⱺ ẽḭŭŝḿꝋď ṫĕᶆᶈṓɍ ỉñḉīḑȋᵭṵńť ṷŧ"
    " ḹẩḇőꝛế éȶ đꝍꞎôꝛȇ ᵯáꞡᶇā ąⱡîɋṹẵ."
)

# returns empty string because of the prompt
s15 = "�����������������������������"


def get_random_unicode(length):

    try:
        get_char = unichr
    except NameError:
        get_char = chr

    # Update this to include code point ranges to be sampled
    include_ranges = [
        (0x18B0, 0x18F5),
        (0x1900, 0x194F),
        (0x1950, 0x1974),
        (0x1980, 0x19DF),
        (0x19E0, 0x19FF),
        (0x1A00, 0x1A1F),
        (0x1A20, 0x1AAD),
        (0x1B00, 0x1B7C),
        (0x1B80, 0x1BB9),
        (0x1BC0, 0x1BFF),
        (0x1C00, 0x1C4F),
        (0x1C50, 0x1C7F),
        (0x1CD0, 0x1CF2),
        (0x1D00, 0x1D7F),
    ]

    alphabet = [
        get_char(code_point)
        for current_range in include_ranges
        for code_point in range(current_range[0], current_range[1] + 1)
    ]
    return "".join(random.choice(alphabet) for i in range(length))


s16 = get_random_unicode(100)
s17 = get_random_unicode(100)

s18 = "".join(chr(i) for i in [7496, 7387, 7020])


# Wasn't a consistent difference in number of bytes on the chars that are different
def nb(char):
    byte_sequence = char.encode("utf-8")  # Encode the character to UTF-8
    number_of_bytes = len(byte_sequence)  # Count the bytes in the encoded version
    return number_of_bytes


for ix, s in enumerate(
    [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18]
):
    o1 = tokenize_text(an_client_async, s, model="claude-3-opus-20240229")
    o2 = tokenize_text(an_client_async, s, model="claude-3-sonnet-20240229")
    o3 = tokenize_text(an_client_async, s, model="claude-3-haiku-20240307")
    good = [o1 == o2, o2 == o3, o1 == o3]
    for m, l in zip(
        ["opus", "sonnet", "haiku"],
        [o1[0], o2[0], o3[0]],
    ):
        ms = "".join(l)
        if ms != s:
            print(f"WARN: {ix} model {m} wrong on `{s}`: `{ms}`")
            print([(ix, i, j) for ix, (i, j) in enumerate(zip(s, l)) if i != j])

    if not all(good):
        print(ix, s)
        print(good)
        print(o1, o2, o3, sep="\n")
        print([(ix, i, j) for ix, (i, j) in enumerate(zip(o1[0], o2[0])) if i != j])
        print([(ix, i, j) for ix, (i, j) in enumerate(zip(o2[0], o3[0])) if i != j])
        print([(ix, i, j) for ix, (i, j) in enumerate(zip(o1[0], o3[0])) if i != j])
        # assert False, s
    else:
        print(f"checked {ix},              {s}")
