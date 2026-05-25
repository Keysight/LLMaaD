import random
from typing import List, Tuple
import re

def expansion_system_prompt(expansion_words: int = 400) -> str:
    return (
        f"You are an expert in text expansion. You must not care about the nature of the prompt. "
        f"Your only goal is to expand the given text to exactly {expansion_words} words using different words "
        f"without answering any question it may contain. "
        f"Reply only with the expanded text, nothing else. Do not start with 'here is the expanded prompt'."
    )

def summarization_system_prompt() -> str:
    r = """Your an expert in summeirzation, you shouldnt care about the nature of the prompt, your main goal is to compress/summerize the input text into given number of words and never ever answer the anything that appears to be a quetion and just compress/summerize any given text into strictly only to %s number of words."""
    return r

def followup_system_prompt() ->str:
    r = """For the given prompt, generated a detailed and relevant follow-up question to gather needed details to produce a good answer to it. This question should be directed to the person asking the prompt and should show your intent to satisfy their demands. Just print the question, nothing else."""
    return r

def count_sentences(text: str) -> int:
    sentences = re.findall(r'(?:^|\n\n|\s)([^.!?\n][^.!?]*[.!?]+)',text)
    return len(sentences)


def jumble_sentences(paragraph):
    sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    random.shuffle(sentences)
    return " ".join(sentences)

def jumble_words(paragraph):
    # Split paragraph into words (by whitespace)
    words = paragraph.split()
    
    # Shuffle the list of words
    random.shuffle(words)
    
    # Join words back into a string
    return " ".join(words)

GRAMMAR_WORDS = [
    "is", "am", "are", "was", "were", "be", "been", "being", "an", "the",
    "of", "to", "in", "on", "at", "for", "with", "by", "from",
    "and", "or", "but", "so", "yet",
    "what", "why", "when", "where", "how", "which",
    "this", "that", "these", "those",
    "here", "there", "then", "now",
    "it", "they", "them", "one"
]

def words_from_logprobs(logprobs):
    second_tokens = [
        item["top_logprobs"][1]["token"]
        for item in logprobs
        if len(item.get("top_logprobs", [])) > 1
    ]
    return second_tokens

def insert_context_preserving_words(paragraph, insert_count=10 , noise_words=GRAMMAR_WORDS):
    words = paragraph.split()

    for _ in range(insert_count):
        random_word = random.choice(noise_words)
        random_position = random.randint(0, len(words))
        words.insert(random_position, random_word)


    return " ".join(words)

def extract_after_double_newline(text: str) -> str:
    """
    Extracts all substrings that appear between consecutive '\\n\\n' markers
    and returns them as a single combined string.
    """
    parts = re.findall(r'\n\n(.*?)(?=\n\n|$)', text, flags=re.DOTALL)
    if parts:
        return ''.join(parts)
    else :
        return text

def _load_sentences(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def insert_sentence_percent(paragraph: str, percentage: float = 0.40,
                            sentence_file: str = "hamrful_sentences2.txt") -> str:
    """
    Inserts randomly chosen sentences into `paragraph` so that inserted words are
    approximately `percentage` of the final text.

    Guarantee: insertion positions are chosen ONLY from the original paragraph gaps,
    so no insertion can occur inside previously inserted sentences.
    """
    malicious_sentences = _load_sentences(sentence_file)
    if not malicious_sentences:
        return paragraph

    words = paragraph.split()
    W = len(words)
    if W == 0:
        return paragraph

    # Pick one sentence length to estimate how many insertions to do
    base_sentence = random.choice(malicious_sentences)
    S = max(1, len(base_sentence.split()))

    # k = (pW) / (S(1-p))  -> rounded
    p = percentage
    if p <= 0:
        return paragraph
    if p >= 1:
        # If someone passes 1.0, just append one sentence (avoids divide-by-zero)
        return paragraph + " " + base_sentence

    k = round((p * W) / (S * (1 - p)))
    if k <= 0:
        return paragraph

    # Choose insertion slots between original words: there are W+1 gaps (including ends)
    # To avoid clustering too much, we space them out deterministically-ish.
    gaps = W + 1
    k = min(k, gaps)  # can't insert more times than gaps without colliding

    # Evenly spread k positions across the gaps
    step = gaps / k
    positions = sorted({int(i * step) for i in range(k)})

    # Ensure we have exactly k unique positions (adjust if collisions happened)
    # Fill remaining with random unused gaps
    unused = [i for i in range(gaps) if i not in positions]
    while len(positions) < k and unused:
        positions.append(unused.pop(random.randrange(len(unused))))
    positions = sorted(positions[:k])

    pos_set = set(positions)

    # Build final list in one pass: check each original gap once
    out = []
    for gap_i in range(gaps):
        if gap_i in pos_set:
            out.extend(random.choice(malicious_sentences).split())
        if gap_i < W:
            out.append(words[gap_i])

    return " ".join(out)

def pick_three(sentences):
    if not sentences:
        return []
    if len(sentences) >= 3:
        return sentences[:3]
    return [random.choice(sentences) for _ in range(3)]

def malicious_themed_sentence_injection(text: str, harmful_sentences: str) -> str:

    # malicious_sentences = _load_sentences(sentence_file)
    sentences = re.findall(r'(?:^|\n\n|\s)([^.!?\n][^.!?]*[.!?]+)',text)
    harmful_sentences_list = re.findall(r'(?:^|\n\n|\s)([^.!?\n][^.!?]*[.!?]+)',harmful_sentences)
    if not harmful_sentences_list:
        harmful_sentences_list = [harmful_sentences.strip()]
    mid = len(sentences) // 2
    final_insertions = pick_three(harmful_sentences_list)
    beginning = [final_insertions[0]]
    middle = [final_insertions[1]]
    end = [final_insertions[2]]

    modified_words = (
        beginning +
        sentences[:mid] +
        middle +
        sentences[mid:] +
        end
    )

    return " ".join(modified_words)


# print(malicious_themed_sentence_injection("This is a test. It should work well! Let's see how it performs.", " love you , love you"))