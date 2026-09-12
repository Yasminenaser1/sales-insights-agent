"""Level-2 reliability: pull the key numbers out of the analysis into labeled
'facts' with clean names. The report model will reference these by name instead
of retyping numbers, so it can never garble a digit - code owns the numbers.
"""
import pandas as pd


def build_facts(path: str) -> dict:
    """Compute a labeled set of facts directly from the data (not from the
    model's output), so every value is exact. Returns {slot_name: (value, display)}."""
    df = pd.read_csv(path, parse_dates=["order_date"])
    facts = {}

    def add(name, value, display):
        facts[name] = {"value": value, "display": display}

    # Region-level sales facts
    by_region_sales = df.groupby("region")["sales"].sum().sort_values(ascending=False)
    top_r, top_v = by_region_sales.index[0], by_region_sales.iloc[0]
    low_r, low_v = by_region_sales.index[-1], by_region_sales.iloc[-1]
    add("top_region", top_r, top_r)
    add("top_region_sales", top_v, f"${top_v:,.0f}")
    add("bottom_region", low_r, low_r)
    add("bottom_region_sales", low_v, f"${low_v:,.0f}")

    # Profit share of the top region
    prof = df.groupby("region")["profit"].sum()
    top_prof_region = prof.idxmax()
    share = prof.max() / prof.sum() * 100
    add("top_profit_region", top_prof_region, top_prof_region)
    add("top_profit_share", round(share, 1), f"{share:.1f}%")

    # Best and worst month by sales
    monthly = df.set_index("order_date")["sales"].resample("ME").sum()
    best_m, worst_m = monthly.idxmax(), monthly.idxmin()
    add("best_month", best_m.strftime("%B %Y"), best_m.strftime("%B %Y"))
    add("best_month_sales", monthly.max(), f"${monthly.max():,.0f}")
    add("worst_month", worst_m.strftime("%B %Y"), worst_m.strftime("%B %Y"))
    add("worst_month_sales", monthly.min(), f"${monthly.min():,.0f}")

    # Top category by sales
    by_cat = df.groupby("category")["sales"].sum().sort_values(ascending=False)
    add("top_category", by_cat.index[0], by_cat.index[0])
    add("top_category_sales", by_cat.iloc[0], f"${by_cat.iloc[0]:,.0f}")

    return facts


if __name__ == "__main__":
    facts = build_facts("data/sales.csv")
    print("Facts the report can reference (name -> display value):\n")
    for name, f in facts.items():
        print(f"  {{{name}}} = {f['display']}")
