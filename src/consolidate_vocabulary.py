import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab_file",
        help="The file where the vocab is stored",
        required=False,
        default="anthropic_vocab.jsonl",
    )
    args = parser.parse_args()

    with open(args.vocab_file, "r") as f:
        tokens = [json.loads(line)["token"] for line in f]

    tokens = set(tokens)

    with open(args.vocab_file, "w") as f:
        for t in tokens:
            f.write(json.dumps({"token": t}) + "\n")
