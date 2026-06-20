#!/usr/bin/env python3
"""Make one OpenRouter/LiteLLM call to test forced-choice logprobs support."""

from __future__ import annotations

import argparse
import asyncio
import os

from litellm import acompletion

from utility_behavior_gap.io_utils import load_env_file
from utility_behavior_gap.paths import ROOT


async def probe(model: str) -> None:
    load_env_file(ROOT / ".env")
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise RuntimeError("OPENROUTER_API_KEY is required in .env or the shell environment")

    response = await acompletion(
        model=f"openrouter/{model}",
        messages=[
            {
                "role": "user",
                "content": (
                    "Choose the better outcome. Respond with only A or B.\n\n"
                    "A: 10 people are saved from terminal illness.\n"
                    "B: 1 person is saved from terminal illness.\n\n"
                    "Answer:"
                ),
            }
        ],
        max_tokens=1,
        temperature=1.0,
        logprobs=True,
        top_logprobs=20,
    )
    choice = response.choices[0]
    print(f"model: openrouter/{model}")
    print(f"content: {choice.message.content!r}")
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None) if logprobs is not None else None
    if not content:
        print("logprobs: missing")
        return
    top = getattr(content[0], "top_logprobs", None)
    if not top:
        print("top_logprobs: missing")
        return
    print("top_logprobs:")
    for item in top[:10]:
        print(f"  {item.token!r}: {item.logprob}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="xiaomi/mimo-v2.5-pro")
    args = parser.parse_args()
    asyncio.run(probe(args.model))


if __name__ == "__main__":
    main()
