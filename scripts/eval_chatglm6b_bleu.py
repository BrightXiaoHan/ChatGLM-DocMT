import argparse
import fileinput
import json

import sacrebleu
import tqdm
from transformers import AutoModel, AutoTokenizer


def translate(texts, src_lang, tgt_lang):
    if tgt_lang == "en":
        lang = "英文"
    else:
        lang = "中文"
    prompt = f"""
    请将下面的一段文本翻译成{lang}:

    """ + texts
    
    global model
    global tokenizer
    response, _ = model.chat(tokenizer, prompt, history=[])
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file. jsonl format.")
    parser.add_argument("--tgt-lang", type=str, choices=["en", "zh"], default="en")
    args = parser.parse_args()

    global model
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "THUDM/chatglm-6b", trust_remote_code=True
    )

    model = (
        AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
        .half()
        .cuda()
    )

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
        translations = translate(data["input"], src_lang, tgt_lang)

        hyp_docs.append(translations)

    tokenizer = "zh" if args.tgt_lang == "zh" else "13a"
    bleu = sacrebleu.corpus_bleu(hyp_docs, [ref_docs], tokenize=tokenizer)
    print(bleu.score)


if __name__ == "__main__":
    main()
