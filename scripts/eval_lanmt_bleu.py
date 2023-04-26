import argparse
import fileinput
import json

import requests
import sacrebleu
import tqdm


def translate(texts, src_lang, tgt_lang):
    global url
    params = {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "domain": "general",
        "data": {"input": texts},
    }

    response = requests.post(url, json=params)
    return response.json()["data"]["translation"]


def main():
    global url
    parser = argparse.ArgumentParser()
    parser.add_argument("url", type=str, help="URL of the Lan-MT server")
    parser.add_argument("input", type=str, help="Input file. jsonl format.")
    parser.add_argument("--tgt-lang", type=str, choices=["en", "zh"], default="en")
    args = parser.parse_args()
    url = args.url

    hyp_docs = []
    ref_docs = []
    for line in tqdm.tqdm(fileinput.input(args.input)):
        data = json.loads(line)
        instruction = data["instruction"]
        if instruction.endswith("英文:"):
            src_lang = "zh"
            tgt_lang = "en"
        else:
            src_lang = "en"
            tgt_lang = "zh"

        if tgt_lang != args.tgt_lang:
            continue

        ref_docs.append(data["output"])
        src = data["input"].split("\n")
        tgt = data["output"].split("\n")
        assert len(src) == len(tgt)
        translations = translate(src, src_lang, tgt_lang)
        translations = "\n".join(translations)

        hyp_docs.append(translations)

    tokenizer = "zh" if args.tgt_lang == "zh" else "13a"
    bleu = sacrebleu.corpus_bleu(hyp_docs, [ref_docs], tokenize=tokenizer)
    print(bleu.score)


if __name__ == "__main__":
    main()
