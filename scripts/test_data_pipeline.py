from src.dataset import build_data_pipeline


def main() -> None:
    pipeline = build_data_pipeline()

    train_loader = pipeline["train_loader"]
    vocabulary = pipeline["vocabulary"]

    source, source_lengths, target = next(iter(train_loader))

    print("Source shape:", source.shape)
    print("Source lengths shape:", source_lengths.shape)
    print("Target shape:", target.shape)
    print("Vocabulary size:", len(vocabulary))

    print("\nSpecial token indices:")
    print("PAD:", vocabulary.pad_index)
    print("UNK:", vocabulary.unk_index)
    print("SOS:", vocabulary.sos_index)
    print("EOS:", vocabulary.eos_index)

    first_source = source[:, 0].tolist()
    first_target = target[:, 0].tolist()

    print("\nDecoded incorrect sentence:")
    print(
        " ".join(
            vocabulary.decode(
                first_source,
                skip_special_tokens=True,
                stop_at_eos=True,
            )
        )
    )

    print("\nDecoded corrected sentence:")
    print(
        " ".join(
            vocabulary.decode(
                first_target,
                skip_special_tokens=True,
                stop_at_eos=True,
            )
        )
    )


if __name__ == "__main__":
    main()