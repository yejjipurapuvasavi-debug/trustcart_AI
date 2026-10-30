
import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from config import REQUIRED_COLUMNS, VALID_STOCK


@dataclass
class Product:
    product_id: str
    title: str
    brand: str
    category: str
    sub_category: str
    description: str
    attributes: dict[str, Any]
    price_inr: int
    mrp_inr: int
    stock_status: str
    rating: float | None
    review_count: int


@dataclass
class Response:
    action: str
    answer_text: str
    cited_product_ids: list[str] = field(default_factory=list)
    cited_policy_sections: list[str] = field(default_factory=list)
    stated_prices: list[dict[str, Any]] = field(default_factory=list)
    stated_facts: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DataValidationError(ValueError):
    """Raised when challenge files do not match the published schema."""


def load_catalogue(path: Path) -> list[Product]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing catalogue: {path}")

    frame = pd.read_csv(path, dtype={"product_id": "string"})
    missing = REQUIRED_COLUMNS - set(frame.columns)

    if missing:
        raise DataValidationError(
            f"Missing catalogue columns: {sorted(missing)}"
        )

    if len(frame) != 5000:
        raise DataValidationError(
            f"Expected 5,000 catalogue rows; found {len(frame)}."
        )

    if frame.product_id.isna().any() or frame.product_id.duplicated().any():
        raise DataValidationError("Product IDs must be present and unique.")

    products = []

    for row in frame.to_dict(orient="records"):
        attrs = json.loads(row["attributes"]) if row["attributes"] else {}

        if not isinstance(attrs, dict):
            raise DataValidationError(
                f"Invalid attributes for {row['product_id']}"
            )

        stock = str(row["stock_status"])

        if stock not in VALID_STOCK:
            raise DataValidationError(
                f"Invalid stock status for {row['product_id']}: {stock}"
            )

        price = int(row["price_inr"])
        mrp = int(row["mrp_inr"])

        if price < 0 or mrp < price:
            raise DataValidationError(
                f"Invalid price/MRP for {row['product_id']}"
            )

        rating = None if pd.isna(row["rating"]) else float(row["rating"])

        products.append(Product(
            product_id=str(row["product_id"]),
            title=str(row["title"]),
            brand=str(row["brand"]),
            category=str(row["category"]),
            sub_category=str(row["sub_category"]),
            description=str(row["description"]),
            attributes=attrs,
            price_inr=price,
            mrp_inr=mrp,
            stock_status=stock,
            rating=rating,
            review_count=int(row["review_count"]),
        ))

    return products


def load_policy(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing policy: {path}")

    sections = {}
    current_id = None
    title = ""
    lines = []

    def save():
        if current_id:
            sections[current_id] = {
                "title": title,
                "text": "\n".join(lines).strip()
            }

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            save()
            heading = line[4:].strip().split(" ", 1)
            current_id = heading[0]
            title = heading[1] if len(heading) > 1 else ""
            lines = []
        elif current_id and not line.startswith("## "):
            lines.append(line)

    save()

    if not sections:
        raise DataValidationError("No numbered policy sections found.")

    return sections


STOP = set("""
a an the is are was were of for to in on me se ka ki ke hai h hain
kya kitne kitna kitni kaunsa kaunsi konsa konsi wala wali wale
price rate batao do pls please aur ya or and with i my mujhe
chahiye milega milegi hoga ho kar sakte sakti abhi toh bhi jo
ye yeh woh wo what which how much cost costs show find have you
do product item hai kya mujhe ek dikhao dikha options
""".split())


SYNONYMS = {
    "tshirt": "t shirt",
    "facewash": "face wash",
    "powerbank": "power bank",
    "ladies": "women",
    "mens": "men",
    "shoe": "shoes",
    "litre": "l",
    "liter": "l",
    "earbud": "earbuds",
    "earphone": "earphones",
    "colour": "color",
    "krida": "kridaa",
    "kridaa": "kridaa",
    "halogrip": "halo grip",
    "astraair": "astra air",
    "vistaair": "vista air",
}


def tokens(text: str) -> list[str]:
    # Split joined model names: RippleFlex -> Ripple Flex
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)

    text = text.casefold().replace("-", " ").replace("_", " ")
    words = re.findall(r"[a-z0-9]+", text)

    output = []

    for word in words:
        word = SYNONYMS.get(word, word)
        output.extend(
            part for part in word.split()
            if part not in STOP
        )

    return output


class ProductIndex:
    def __init__(self, products: list[Product]):
        self.products = products
        self.docs = []
        document_frequency = Counter()

        for product in products:
            attribute_text = []

            for key, value in product.attributes.items():
                attribute_text.append(key)

                if isinstance(value, list):
                    attribute_text.extend(map(str, value))
                else:
                    attribute_text.append(str(value))

            text = " ".join([
                product.title,
                product.brand,
                product.category,
                product.sub_category,
                product.description,
                " ".join(attribute_text),
            ])

            terms = set(tokens(text))
            self.docs.append(terms)
            document_frequency.update(terms)

        total = max(1, len(products))

        self.idf = {
            term: math.log((total + 1) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str) -> list[tuple[float, Product]]:
        query_terms = set(tokens(query))

        if not query_terms:
            return []

        scored = []

        for product, document in zip(self.products, self.docs):
            common = query_terms & document

            if not common:
                continue

            score = sum(self.idf.get(term, 0) for term in common)

            # Exact brand/title matches receive a small ranking bonus.
            title_terms = set(tokens(product.title + " " + product.brand))
            score += 2 * len(query_terms & title_terms)

            scored.append((score, product))

        return sorted(
            scored,
            key=lambda item: (-item[0], item[1].product_id)
        )


# ---------------- POLICY ROUTING ----------------

POLICY_RULES = [
    (r"\b(opened|seal|sealed|hygiene seal|earbuds|earphones)\b", ["6.2"]),
    (r"\b(face wash|facewash|shampoo|hair oil|perfume|sunscreen|beauty|opened.*return)\b", ["6.1"]),
    (r"\b(beard trimmer|trimmer used|used trimmer)\b", ["6.3"]),
    (r"\b(warranty claim|claim warranty|claim process)\b", ["3.4"]),
    (r"\b(warranty cover|covered by warranty)\b", ["3.2"]),
    (r"\b(warranty exclude|not covered.*warranty)\b", ["3.3"]),
    (r"\b(warranty|guarantee)\b", ["3.1"]),
    (r"\b(damaged|defective|wrong item|wrong product)\b", ["1.7"]),
    (r"\b(refund|money back|refund time|refund method)\b", ["1.5"]),
    (r"\b(exchange)\b", ["1.6"]),
    (r"\b(restocking|free return|return pickup|pickup charge)\b", ["1.2", "1.4"]),
    (r"\b(return condition|unused|unwashed|original tags)\b", ["1.3"]),
    (r"\b(start.*return|how.*return|return process)\b", ["1.4"]),
    (r"\b(return window|days.*return|return.*days)\b", ["1.1"]),
    (r"\b(return)\b", ["1.1"]),
    (r"\b(free delivery|delivery charge|shipping charge|delivery fee)\b", ["2.1"]),
    (r"\b(delivery time|when.*deliver|how long.*deliver)\b", ["2.2"]),
    (r"\b(international|outside india|ship abroad)\b", ["2.3"]),
    (r"\b(track order|tracking)\b", ["2.4"]),
    (r"\b(low stock|out of stock|backorder|pre.order|stock status)\b", ["2.5"]),
    (r"\b(cash on delivery|\bcod\b)\b", ["4.2"]),
    (r"\b(payment method|upi|debit card|credit card|net banking|wallet)\b", ["4.1"]),
    (r"\b(failed payment|money debited|payment failed)\b", ["4.3"]),
    (r"\b(cancel.*dispatch|before dispatch)\b", ["5.1"]),
    (r"\b(after dispatch|cancel.*shipped)\b", ["5.2"]),
    (r"\b(cancel)\b", ["5.1", "5.2"]),
    (r"\b(gst|mrp|inclusive of tax|tax included)\b", ["7.1"]),
    (r"\b(price change|price changes)\b", ["7.2"]),
    (r"\b(coupon|coupons|combine.*offer)\b", ["7.3"]),
    (r"\b(price match|match.*price)\b", ["7.4"]),
    (r"\b(sale|discounted item|discount.*return)\b", ["7.5"]),
    (r"\b(customer support|contact support|helpline|support number)\b", ["8.1"]),
]


def _policy_sections(query: str, policy: dict) -> list[str]:
    low = query.casefold()
    found = []

    for pattern, section_ids in POLICY_RULES:
        if re.search(pattern, low):
            for section_id in section_ids:
                if section_id in policy and section_id not in found:
                    found.append(section_id)

    # A warranty claim for a faulty product also invokes the defective-item rule.
    if re.search(r"\bwarranty\b", low) and re.search(
        r"\b(broken|faulty|damaged|defective|kharab)\b", low
    ):
        for section_id in ("3.4", "1.7"):
            if section_id in policy and section_id not in found:
                found.append(section_id)

    # A claim for a faulty product should use the claim process,
    # not the general warranty coverage section.
    if "3.4" in found and "3.1" in found:
        found.remove("3.1")

    return found


def _policy_section(query: str, policy: dict) -> str | None:
    sections = _policy_sections(query, policy)
    return sections[0] if sections else None


def _is_policy(query: str) -> bool:
    return bool(re.search(
        r"\b(return|refund|exchange|shipping|delivery|warranty|policy|"
        r"coupon|cod|cancel|payment|gst|restocking|tracking|"
        r"international|support|mrp|dispatch|seal|opened|"
        r"low stock|backorder)\b",
        query,
        re.I
    ))


# ---------------- PRODUCT MATCHING ----------------

def _is_ooc(query: str) -> bool:
    low = query.casefold()

    # These are laptop-carrying accessories, not laptops.
    if any(term in low for term in (
        "laptop bag", "laptop sleeve", "laptop backpack"
    )):
        return False

    unsupported = (
        "iphone", "smartphone", "mobile phone", "redmi note",
        "samsung galaxy phone", "laptop", "television", "smart tv",
        "groceries", "grocery", "maggi noodles", "noodles",
        "crocin", "medicine", "medicines", "tablet medicine",
        "jewellery", "jewelry", "payal", "bike price",
        "motorcycle", "bajaj pulsar", "car price",
        "printer cartridge", "hp printer cartridge",
    )

    return any(term in low for term in unsupported)


def _budget(query: str) -> int | None:
    match = re.search(
        r"(?:under|below|less than|within|ke andar|andar)"
        r"\s*(?:₹|rs\.?|inr)?\s*([\d,]+)",
        query,
        re.I
    )

    if not match:
        return None

    return int(match.group(1).replace(",", ""))


def _attribute(query: str) -> str | None:
    patterns = [
        ("water_resistant", r"waterproof|water resistant|paani me"),
        ("dishwasher_safe", r"dishwasher"),
        ("induction_compatible", r"induction"),
        ("paraben_free", r"paraben"),
        ("battery_life_hours", r"battery life|battery backup"),
        ("microwave_safe", r"microwave"),
        ("bpa_free", r"bpa.?free"),
        ("leak_proof", r"leak.?proof"),
        ("warranty_months", r"warranty"),
        ("sizes", r"\bsize\b"),
        ("shoe_compartment", r"shoe compartment"),
        ("connectivity", r"bluetooth"),
        ("thickness_mm", r"thick|moti|mm"),
    ]

    for key, pattern in patterns:
        if re.search(pattern, query, re.I):
            return key

    return None


def _numeric_constraint(query: str):
    patterns = [
        (r"(\d+(?:\.\d+)?)\s*(?:watt|watts|w)\b",
         ("output_w", "wattage_w")),
        (r"(\d+(?:\.\d+)?)\s*(?:litre|liter|l)\b",
         ("capacity_l",)),
        (r"(\d+(?:\.\d+)?)\s*(?:mm|millimetre|millimeter)\b",
         ("thickness_mm",)),
        (r"(\d+)\s*(?:pieces|piece|pcs)\b",
         ("pieces",)),
        (r"(\d+)\s*(?:pack|packs)\b",
         ("pack_of",)),
        (r"(\d+)\s*(?:ml|millilitre|milliliter)\b",
         ("volume_ml", "capacity_ml")),
    ]

    for pattern, keys in patterns:
        match = re.search(pattern, query, re.I)

        if match:
            return float(match.group(1)), keys

    return None


COLOURS = {
    "blue", "navy", "teal", "black", "white", "red",
    "green", "pink", "purple", "brown", "yellow",
    "beige", "orange", "charcoal", "lavender", "silver",
    "grey", "gray", "gold"
}

MATERIALS = {
    "cotton", "leather", "steel", "silicone", "plastic",
    "wood", "mesh", "fleece", "wool", "linen", "tpe",
    "rubber", "pvc", "polyester"
}


def _attribute_text(product: Product, keys: tuple[str, ...]) -> str:
    values = []

    for key in keys:
        value = product.attributes.get(key)

        if value is not None:
            values.append(str(value).casefold())

    return " ".join(values)


def _matches_constraints(
    query: str,
    product: Product,
    use_colour: bool = True
) -> bool:
    attrs = product.attributes

    # Exact numeric attribute constraints.
    numeric = _numeric_constraint(query)

    if numeric:
        wanted, keys = numeric
        actual_values = []

        for key in keys:
            if key in attrs:
                try:
                    actual_values.append(float(attrs[key]))
                except (TypeError, ValueError):
                    pass

        if not actual_values or not any(
            abs(value - wanted) < 0.001 for value in actual_values
        ):
            return False

    # Budget is a hard constraint.
    budget = _budget(query)

    if budget is not None and product.price_inr > budget:
        return False

    query_words = set(tokens(query))

    # Colour is checked against the actual catalogue attribute.
    wanted_colours = query_words & COLOURS

    if use_colour and wanted_colours:
        actual_colour = _attribute_text(product, ("colour", "color"))

        if not any(
            colour in actual_colour for colour in wanted_colours
        ):
            return False

    # Material is checked against actual material attributes.
    wanted_materials = query_words & MATERIALS

    if wanted_materials:
        actual_material = _attribute_text(
            product,
            ("material", "upper_material", "sole_material", "shell")
        )

        if not all(
            material in actual_material for material in wanted_materials
        ):
            return False

    return True


def _is_price_question(query: str) -> bool:
    return bool(re.search(
        r"\b(price|cost|rate|kitne|kitna|kitni|rupees|rs)\b|"
        r"kitne ka|kitne ki|ka price|ki price|sasta|sasti",
        query,
        re.I
    ))


def _is_stock_question(query: str) -> bool:
    return bool(re.search(
        r"\b(stock|available|availability|milega|milegi)\b|"
        r"stock me|stock hai",
        query,
        re.I
    ))


def _is_cheapest(query: str) -> bool:
    return bool(re.search(
        r"\b(cheapest|lowest|sabse sasta|sabse sasti)\b",
        query,
        re.I
    ))


def _is_expensive(query: str) -> bool:
    return bool(re.search(
        r"\b(costliest|most expensive|highest priced|mehenga|mehngi)\b",
        query,
        re.I
    ))


def _is_browse(query: str) -> bool:
    return bool(re.search(
        r"\b(show|options|dikhao|dikha|list|suggest|recommend)\b",
        query,
        re.I
    ))


def _is_short_followup(query: str) -> bool:
    words = tokens(query)

    return (
        len(words) <= 3
        and bool(re.search(
            r"\b(yeh|ye|woh|wo|same|another|aur|lavender|blue|red|"
            r"purple|teal|black|white|pink|green|brown|wala|wali)\b",
            query,
            re.I
        ))
    )


def _format_product(product: Product) -> str:
    return (
        f"{product.title} ({product.product_id}) — "
        f"₹{product.price_inr:,}; stock: {product.stock_status} "
        f"[Product {product.product_id}]"
    )


def _make_product_response(
    action: str,
    message: str,
    selected: list[Product]
) -> Response:
    return Response(
        action=action,
        answer_text=message,
        cited_product_ids=[p.product_id for p in selected],
        stated_prices=[
            {"product_id": p.product_id, "price_inr": p.price_inr}
            for p in selected
        ],
        stated_facts=[
            {
                "product_id": p.product_id,
                "field": "stock_status",
                "value": p.stock_status
            }
            for p in selected
        ]
    )


# ---------------- MAIN ANSWERING FUNCTION ----------------

def answer_query(
    query: str,
    history: list[str],
    products: list[Product],
    index: ProductIndex,
    policy: dict[str, dict[str, str]]
) -> Response:

    start = time.perf_counter()
    q = query.strip()

    if not q or len(q) > 500:
        result = Response(
            "refuse",
            "Please enter a question between 1 and 500 characters."
        )

    elif _is_ooc(q):
        result = Response(
            "refuse",
            "I can't verify that item in the supplied catalogue, "
            "so I won't invent its price or availability."
        )

    elif _is_policy(q):
        section_ids = _policy_sections(q, policy)

        if section_ids:
            parts = []

            for section_id in section_ids:
                section = policy[section_id]
                parts.append(
                    f"Policy section {section_id} — "
                    f"{section['title']}: {section['text']} "
                    f"[Policy {section_id}]"
                )

            result = Response(
                "answer",
                "\n\n".join(parts),
                cited_policy_sections=section_ids
            )
        else:
            result = Response(
                "refuse",
                "The supplied policy does not contain enough "
                "information to confirm that."
            )

    else:
        # Use only the immediately preceding question for a short
        # follow-up, rather than mixing several old questions.
        context = q

        if history and _is_short_followup(q):
            context = history[-1] + " " + q

        hits = index.search(context)

        # Apply exact catalogue constraints.
        hits = [
            (score, product)
            for score, product in hits
            if _matches_constraints(context, product)
        ]

        if not hits:
            result = Response(
                "refuse",
                "I couldn't find a sufficiently supported match "
                "in the supplied catalogue. I won't guess."
            )

        else:
            top_score = hits[0][0]

            # Retain close matches so variants are not silently discarded.
            close_hits = [
                (score, product)
                for score, product in hits
                if score >= top_score * 0.85
            ]

            candidates = [product for _, product in close_hits]

            asks_price = _is_price_question(context)
            asks_stock = _is_stock_question(context)
            asks_compare = bool(re.search(
                r"\b(cheaper|sasta|sasti|kaunsa.*sasta|compare|"
                r"which.*cheap)\b",
                q,
                re.I
            ))
            cheapest = _is_cheapest(q)
            expensive = _is_expensive(q)
            browse = _is_browse(q) or _budget(context) is not None

            # Product attribute questions must be supported by that
            # exact product's own catalogue entry.
            attr = _attribute(q)

            # A price comparison is answered only when multiple
            # supported variants are present.
            if asks_compare and len(candidates) >= 2:
                selected = sorted(
                    candidates,
                    key=lambda p: p.price_inr
                )[:5]

                result = _make_product_response(
                    "answer",
                    "Price comparison of matching catalogue products:\n"
                    + "\n".join(_format_product(p) for p in selected),
                    selected
                )

            elif cheapest or expensive:
                pool = candidates

                chosen = (
                    min(pool, key=lambda p: p.price_inr)
                    if cheapest
                    else max(pool, key=lambda p: p.price_inr)
                )

                result = _make_product_response(
                    "answer",
                    "Best matching "
                    + ("lowest-priced" if cheapest else "highest-priced")
                    + " product: "
                    + _format_product(chosen),
                    [chosen]
                )

            elif browse:
                selected = candidates[:5]

                result = _make_product_response(
                    "answer",
                    "Matching catalogue options:\n"
                    + "\n".join(_format_product(p) for p in selected),
                    selected
                )

            elif attr:
                # Use the best-supported product, but refuse if the
                # requested attribute is not listed for that product.
                chosen = candidates[0]

                if attr not in chosen.attributes:
                    result = Response(
                        "refuse",
                        f"The catalogue does not list {attr} for "
                        f"{chosen.title} ({chosen.product_id}), "
                        "so I can't confirm it.",
                        cited_product_ids=[chosen.product_id]
                    )
                else:
                    value = chosen.attributes[attr]

                    result = Response(
                        "answer",
                        f"{chosen.title} ({chosen.product_id}): "
                        f"{attr} = {value}. [Product {chosen.product_id}]",
                        cited_product_ids=[chosen.product_id],
                        stated_facts=[{
                            "product_id": chosen.product_id,
                            "field": attr,
                            "value": value
                        }]
                    )

            # Ask the customer to choose when close variants differ
            # and the requested answer would otherwise be ambiguous.
            elif len(candidates) > 1 and (asks_price or asks_stock):
                selected = candidates[:5]

                result = _make_product_response(
                    "clarify",
                    "Several catalogue variants match. "
                    "Please choose the exact variant:\n"
                    + "\n".join(_format_product(p) for p in selected),
                    selected
                )

            else:
                chosen = candidates[0]

                # Answer price and stock together whenever requested.
                if asks_price or asks_stock:
                    details = []

                    if asks_price:
                        details.append(
                            f"selling price: ₹{chosen.price_inr:,}"
                        )

                    if asks_stock:
                        details.append(
                            f"stock status: {chosen.stock_status}"
                        )

                    result = Response(
                        "answer",
                        f"{chosen.title} ({chosen.product_id}) "
                        + "; ".join(details)
                        + f". [Product {chosen.product_id}]",
                        cited_product_ids=[chosen.product_id],
                        stated_prices=(
                            [{
                                "product_id": chosen.product_id,
                                "price_inr": chosen.price_inr
                            }]
                            if asks_price else []
                        ),
                        stated_facts=(
                            [{
                                "product_id": chosen.product_id,
                                "field": "stock_status",
                                "value": chosen.stock_status
                            }]
                            if asks_stock else []
                        )
                    )

                else:
                    result = Response(
                        "answer",
                        f"I found {chosen.title} ({chosen.product_id}) "
                        f"in the catalogue. [Product {chosen.product_id}]",
                        cited_product_ids=[chosen.product_id]
                    )

    result.latency_ms = round(
        (time.perf_counter() - start) * 1000,
        3
    )
    result.tokens_in = 0
    result.tokens_out = 0

    return result
