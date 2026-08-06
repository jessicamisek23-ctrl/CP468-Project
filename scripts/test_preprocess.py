from src.preprocess import Vocabulary, clean_text, tokenize


def main() -> None:
    sample_sentences = [
        "She don't like apples.",
        "She doesn't like apples.",
        "here are some sample outputs!",
        "Here are some sample outputs!",
    ]

    print("Cleaning and tokenization:\n")

    for sentence in sample_sentences:
        print("Original:", sentence)
        print("Cleaned: ", clean_text(sentence))
        print("Tokens:  ", tokenize(sentence))
        print()

    vocabulary = Vocabulary(
        minimum_frequency=1,
        maximum_size=100,
    )

    vocabulary.build(
        tokenize(sentence)
        for sentence in sample_sentences
    )

    print("Vocabulary size:", len(vocabulary))
    print("PAD index:", vocabulary.pad_index)
    print("UNK index:", vocabulary.unk_index)
    print("SOS index:", vocabulary.sos_index)
    print("EOS index:", vocabulary.eos_index)

    tokens = tokenize("She doesn't like apples.")

    encoded = vocabulary.encode(
        tokens,
        add_sos=True,
        add_eos=True,
    )

    decoded = vocabulary.decode(
        encoded,
        skip_special_tokens=False,
    )

    print("\nTokens:", tokens)
    print("Encoded:", encoded)
    print("Decoded:", decoded)


if __name__ == "__main__":
    main()