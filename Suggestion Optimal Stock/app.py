import math
import pandas as pd

# Sample data for demonstration
data = {
    'Product': ['Product A', 'Product B', 'Product C'],
    'Annual_Demand': [5000, 3000, 8000],  # units
    'Ordering_Cost': [50, 75, 60],        # cost per order
    'Holding_Cost_Per_Unit': [2.5, 3.0, 2.0]  # cost per unit per year
}

# Create a DataFrame
df = pd.DataFrame(data)

# Function to calculate EOQ
def calculate_eoq(demand, order_cost, holding_cost):
    return math.sqrt((2 * demand * order_cost) / holding_cost)

# Add a new column 'EOQ' to the DataFrame
df['EOQ'] = df.apply(
    lambda row: calculate_eoq(row['Annual_Demand'], row['Ordering_Cost'], row['Holding_Cost_Per_Unit']),
    axis=1
)

# Display the DataFrame with EOQ results
print("Optimal Order Quantity (EOQ) for each product:")
print(df)
