# %%
import numpy as np
import random
from itertools import zip_longest
import string

from anthropic import Anthropic, AsyncAnthropic

from src.anthropic_tokenizer import tokenize_text

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


# Wasn't a consistent difference in number of bytes on the chars that are different
def nb(char):
    byte_sequence = char.encode("utf-8")  # Encode the character to UTF-8
    number_of_bytes = len(byte_sequence)  # Count the bytes in the encoded version
    return number_of_bytes


for ix, s in enumerate([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14]):
    o1 = tokenize_text(an_client_async, s, model="claude-3-opus-20240229")
    o2 = tokenize_text(an_client_async, s, model="claude-3-sonnet-20240229")
    o3 = tokenize_text(an_client_async, s, model="claude-3-haiku-20240307")
    good = [o1 == o2, o2 == o3, o1 == o3]
    for m, l in zip(
        ["opus", "sonnet", "haiku"],
        [o1[0], o2[0], o3[0]],
    ):
        if "".join(l) != s:
            print(f"WARN: {ix} model {m} wrong on `{s}`")

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
