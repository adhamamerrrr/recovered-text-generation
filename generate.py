"""Local text generation: a stdlib Markov baseline and optional GPT-Neo inference."""
import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re


@dataclass(frozen=True)
class Options:
    max_new_tokens: int = 60
    seed: int = 42
    temperature: float = 0.8

    def validate(self):
        if not 1 <= self.max_new_tokens <= 512:
            raise ValueError('max-new-tokens must be between 1 and 512')
        if not math.isfinite(self.temperature) or not 0 < self.temperature <= 2:
            raise ValueError('temperature must be greater than 0 and at most 2')


class MarkovGenerator:
    """A first-order word model; repeated transitions retain their frequency."""
    def __init__(self, text):
        tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
        if len(tokens) < 2:
            raise ValueError('corpus must contain at least two tokens')
        self.transitions = defaultdict(list)
        for a, b in zip(tokens, tokens[1:]):
            self.transitions[a].append(b)

    def generate(self, prompt, options):
        options.validate()
        if not prompt.strip():
            raise ValueError('prompt cannot be empty')
        rng = random.Random(options.seed)
        parts = re.findall(r"\w+(?:'\w+)?|[^\w\s]", prompt)
        token = parts[-1]
        result = []
        for _ in range(options.max_new_tokens):
            candidates = self.transitions.get(token)
            if not candidates:
                break
            token = rng.choice(candidates)
            result.append(token)
        continuation = ' '.join(result)
        continuation = re.sub(r'\s+([.,!?;:])', r'\1', continuation)
        return continuation


def gpt_neo_generate(prompt, options, model_id='EleutherAI/gpt-neo-125m', device='cpu'):
    options.validate()
    if not prompt.strip():
        raise ValueError('prompt cannot be empty')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; use --device cpu')
    set_seed(options.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=False)
    model.to(device).eval()
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    length = inputs['input_ids'].shape[1]
    limit = getattr(model.config, 'max_position_embeddings', 2048)
    if length + options.max_new_tokens > limit:
        raise ValueError(f'prompt plus continuation exceeds model context of {limit} tokens')
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=options.max_new_tokens,
                                do_sample=True, temperature=options.temperature,
                                top_p=0.95, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, length:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--backend', choices=['markov', 'gpt-neo'], default='markov')
    parser.add_argument('--corpus', type=Path, default=Path(__file__).with_name('sample_corpus.txt'))
    parser.add_argument('--model', default='EleutherAI/gpt-neo-125m')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--max-new-tokens', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=0.8)
    args = parser.parse_args()
    options = Options(args.max_new_tokens, args.seed, args.temperature)
    try:
        if args.backend == 'markov':
            continuation = MarkovGenerator(args.corpus.read_text(encoding='utf-8')).generate(args.prompt, options)
        else:
            continuation = gpt_neo_generate(args.prompt, options, args.model, args.device)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f'Generation failed: {exc}\n')
    print(json.dumps({'backend': args.backend, 'prompt': args.prompt,
                      'continuation': continuation, 'seed': args.seed}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
