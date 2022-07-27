import itertools
import random
from spacy.lang.en import English
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS, HYPHENS
from spacy.util import compile_infix_regex
from spacy.attrs import ORTH
from typing import List
from num2words import num2words


def get_tokenizer(type: str, magnitude: int=None):
    if type == "math":
        return ArithmeticTokenizer()
    elif type == "language":
        return ArithmeticLanguageTokenizer(magnitude)
    else:
        raise ValueError("Invalid tokenizer type")


def get_all_pairwise_power_ten_combinations(magnitude):
    return set(itertools.product(range(10**magnitude), repeat=2))


def get_power_ten_combinations_splits(magnitude=1, train_ratio=0.8, seed=42):
    all_ex = get_all_pairwise_power_ten_combinations(magnitude)
    random.seed(seed)
    train_set = set(random.sample(all_ex, int(10**(magnitude*2) * train_ratio)))
    test_set = all_ex.difference(train_set)
    return train_set, test_set


def get_numeric_addition_string(*numbers):
    return f"{'+'.join(str(n) for n in numbers)}={str(sum(numbers))}"


def get_language_addition_string(*n):
    return f"{' plus '.join(num2words(n_i) for n_i in n)} is equal to {str(num2words(sum(n)))}"


class ArithmeticTokenizer:

    def __init__(self) -> None:
        self.vocab = ["<pad>", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "<s>", "+", "=", "</s>"]
        self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.pad_token_idx = self.token2idx["<pad>"]
        self.bos_token_idx = self.token2idx["<s>"]
        self.eos_token_idx = self.token2idx["</s>"]

    def encode(self, text: str, add_bos_token=True, add_eos_token=True) -> List[int]:
        return ([self.bos_token_idx] if add_bos_token else []) + \
            [self.token2idx[c] for c in text.replace("<s>", "").replace("</s>", "")] + \
            ([self.eos_token_idx] if add_eos_token else [])

    def decode(self, encoded_text: List[int], skip_special_tokens: bool=False) -> str:
        if not skip_special_tokens:
            return "".join(self.vocab[token_idx] for token_idx in encoded_text)
        else:
            return "".join(self.vocab[token_idx] for token_idx in encoded_text).replace("<pad>","").replace("<s>","").replace("</s>","")


class ArithmeticLanguageTokenizer:

    def __init__(self, magnitude, include_numbers=False):
        all_ex = get_all_pairwise_power_ten_combinations(magnitude)
        self.tokenizer = English().tokenizer
        self.vocab = ["<pad>", "<s>", "</s>"] + sorted(
            list(
                set(token.text for nums in all_ex for token in self.tokenizer(get_language_addition_string(*nums)))
            )
        )
        if include_numbers:
            numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "=", "+"]
            self.vocab = self.vocab + numbers

            # make spacy tokenizer split at numbers
            infixes = (
                LIST_ELLIPSES
                + LIST_ICONS
                + [
                    r"(?<=[0-9])(?=[0-9-])",
                    r"(?<=[=0-9])=(?=[=0-9-])",
                    r"(?<=[0-9])[+\-\*^](?=[0-9-])",
                    r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                        al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                    ),
                    r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                    r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
                    r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA)
                ]
            )
            infix_re = compile_infix_regex(infixes)
            self.tokenizer.infix_finditer = infix_re.finditer
            self.tokenizer.add_special_case("=3", [{ORTH: "="}, {ORTH: "3"}])

        self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.pad_token_idx = self.token2idx["<pad>"]
        self.bos_token_idx = self.token2idx["<s>"]
        self.eos_token_idx = self.token2idx["</s>"]

    def encode(self, text: str, add_bos_token=True, add_eos_token=True) -> List[int]:
        return ([self.bos_token_idx] if add_bos_token else []) + \
            [self.token2idx[token.text] for token in self.tokenizer(text)] + \
            ([self.eos_token_idx] if add_eos_token else [])

    def decode(self, encoded_text: List[int]) -> str:
        return " ".join([self.vocab[idx] for idx in encoded_text])
