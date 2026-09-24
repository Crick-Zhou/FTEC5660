#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_deepseek import ChatDeepSeek

QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    system_prompt="""
    
    Extract accounting fields from one supermarket receipt image.

    Return JSON only, using exactly this structure:
    {
      "amount_paid_after_rounding": "102.30",
      "subtotal_after_discounts_before_rounding": "102.31",
      "discounts": ["-5.39"]
    }

    Rules:
    - amount_paid_after_rounding is the final payable or charged amount
      after ROUNDING.
    - Do not confuse it with cash tendered, change, card balance, points,
      savings, or transaction numbers.
    - subtotal_after_discounts_before_rounding is the printed SUBTOTAL
      before ROUNDING.
    - discounts contains every actual discount, promotion, coupon,
      member discount, app discount, or percentage discount amount.
    - Use the actual applied monetary amount, not a percentage or
      promotional description.
    - Never include ROUNDING in discounts.
    - If there is no discount, return an empty list.
    - Return valid JSON only, with no Markdown and no explanation.

    """
    prompt=ChatPromptTemplate.from_messages(
        [SystemMessage(content=system_prompt),
         MessagesPlaceholder(variable_name="messages"),
         ]
    )

    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
        max_retries=3,
    )
    return prompt | model


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    inputs = []
    for path in images:
        message=HumanMessage(
            content=[{
                "type":"text",
                "text":"Please Extract the required receipt fields as JSON",
            },
            {
                "type":"image_url",
                "image_url":{
                    'url':image_data_url(path),
                },
            }
            ]
        )
        inputs.append({"messages":[message]})

    model_results= chain.batch(inputs,config={"max_concurrency":3})

    paid_total = Decimal("0.00")
    without_discount_total = Decimal("0.00")

    for path,result in zip(images,model_results):
        text = response_text(result)
        data = json.loads(text)
        paid = Decimal(data["amount_paid_after_rounding"])
        subtotal = Decimal(data["subtotal_after_discounts_before_rounding"])
        discounts = [Decimal(value)
                     for value in data["discounts"]
        ]
        paid_total+=paid
        without_discount_total+=(
            subtotal+sum((abs(value) for value in discounts),Decimal("0.00"))
        )
    paid_total= paid_total.quantize(Decimal("0.01"))
    without_discount_total=without_discount_total.quantize(Decimal("0.00"))
    return{
        QUERY_1: f"HK${paid_total:.2f}",
        QUERY_2: f"HK${without_discount_total:.2f}",
    }

    # _ = (chain, images)
    # return {QUERY_1: DUMMY_RESPONSE, QUERY_2: DUMMY_RESPONSE}


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
