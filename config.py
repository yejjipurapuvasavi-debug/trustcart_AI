from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT
CATALOGUE = DATA / "catalogue.csv"
POLICY = DATA / "policy.md"
REQUIRED_COLUMNS = {
    "product_id", "title", "brand", "category", "sub_category",
    "description", "attributes", "price_inr", "mrp_inr",
    "stock_status", "rating", "review_count",
}
VALID_STOCK = {"in_stock", "low_stock", "out_of_stock"}
