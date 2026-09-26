import numpy as np
import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parent
folder = base / "data"

folder.mkdir(exist_ok=True)

np.random.seed(42)

low = pd.DataFrame({
    "purchase_frequency": np.random.randint(1, 5, 34),
    "monthly_spending": np.random.randint(70, 190, 34)
})

medium = pd.DataFrame({
    "purchase_frequency": np.random.randint(4, 9, 33),
    "monthly_spending": np.random.randint(220, 380, 33)
})

high = pd.DataFrame({
    "purchase_frequency": np.random.randint(7, 12, 33),
    "monthly_spending": np.random.randint(400, 600, 33)
})

df = pd.concat(
    [low, medium, high],
    ignore_index=True
)

df.insert(
    0,
    "customer_id",
    range(1, 101)
)

df.to_csv(
    folder / "ecommerce_customers.csv",
    index=False
)

print("Dataset created")
print(f"Records: {len(df)}")
print(df.head())