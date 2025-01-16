import pandas as pd
import folium
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report


def convert_float(inp):
    splitted_data = inp.split(",")
    return float(splitted_data[-2]), float(splitted_data[-1])

# Step 1: Load and Process Data
file_path = r'C:\Users\ahmad\Documents\tamz\My Proj\ericson\data\datatam.xlsx'  # Update with your file path
df = pd.read_excel(file_path)

# Clean column names
df.columns = df.columns.str.strip()

# Add derived features (example: total utilization)
df['Total Utilization'] = df['Util Link 1'] + df['Util Link 2']

# Extract tower coordinates (replace with actual mapping logic)
tower_coordinates = df[['Tower ID', 'Coordinates']].dropna()

print(tower_coordinates)

# df["Source Coordinates"] = df["Tower ID"].map(tower_coordinates['Coordinates'])
# df["Target Coordinates"] = df["ROH"].map(tower_coordinates['Coordinates'])

df["Source Coordinates"] = tower_coordinates['Coordinates']
df["Target Coordinates"] = tower_coordinates['Coordinates']

print(df["Source Coordinates"])
print(df["Target Coordinates"])

# # Step 2: Prepare Data for Machine Learning
# features = df[["COUNT hop", "Link Qty", "Util Link 1", "Util Link 2", "Total Utilization"]].fillna(0)

# # Targets: Congestion and High Traffic
# target_congestion = (df["Util Link 1"] > 80).astype(int)  # Congested if Util Link 1 > 80%
# target_high_traffic = (df["Util Link 1"] > 50).astype(int)  # High traffic if Util Link 1 > 50%

# # Train-test split
# X_train, X_test, y_train_congestion, y_test_congestion = train_test_split(
#     features, target_congestion, test_size=0.2, random_state=42
# )
# _, _, y_train_traffic, y_test_traffic = train_test_split(
#     features, target_high_traffic, test_size=0.2, random_state=42
# )

# Step 3: Train Models
# Congestion Model
# model_congestion = RandomForestClassifier(random_state=42)
# model_congestion.fit(X_train, y_train_congestion)
# y_pred_congestion = model_congestion.predict(X_test)
# print("Congestion Model Accuracy:", accuracy_score(y_test_congestion, y_pred_congestion))
# print("Congestion Model Report:\n", classification_report(y_test_congestion, y_pred_congestion))

# # High Traffic Model
# model_traffic = RandomForestClassifier(random_state=42)
# model_traffic.fit(X_train, y_train_traffic)
# y_pred_traffic = model_traffic.predict(X_test)
# print("High Traffic Model Accuracy:", accuracy_score(y_test_traffic, y_pred_traffic))
# print("High Traffic Model Report:\n", classification_report(y_test_traffic, y_pred_traffic))

# # Utilization Prediction Model (for alternative routes)
# model_utilization = RandomForestRegressor(random_state=42)
# model_utilization.fit(X_train, features["Total Utilization"])

# Predict future utilization
# future_utilization = model_utilization.predict(X_test)

# Step 2: Prepare Data for Machine Learning
# Fill missing values to avoid errors
features = df[["COUNT hop", "Link Qty", "Util Link 1", "Util Link 2", "Total Utilization"]].fillna(0)

# Targets: Congestion and High Traffic
target_congestion = (df["Util Link 1"] > 80).astype(int)  # Congested if Util Link 1 > 80%
target_high_traffic = (df["Util Link 1"] > 50).astype(int)  # High traffic if Util Link 1 > 50%

# Debug shapes before splitting
print("Features Shape:", features.shape)
print("Target Congestion Shape:", target_congestion.shape)
print("Target High Traffic Shape:", target_high_traffic.shape)

# Step 2: Split the data in one go
X_train, X_test, y_train_congestion, y_test_congestion, y_train_traffic, y_test_traffic = train_test_split(
    features, target_congestion, target_high_traffic, test_size=0.2, random_state=42
)



# Step 3: Train Models
# Congestion Model
model_congestion = RandomForestClassifier(random_state=42)
model_congestion.fit(X_train, y_train_congestion)
y_pred_congestion = model_congestion.predict(X_test)

# Check accuracy for congestion prediction
print("Congestion Model Accuracy:", accuracy_score(y_test_congestion, y_pred_congestion))
print("Congestion Model Report:\n", classification_report(y_test_congestion, y_pred_congestion))

# High Traffic Model
model_traffic = RandomForestClassifier(random_state=42)
model_traffic.fit(X_train, y_train_traffic)
y_pred_traffic = model_traffic.predict(X_test)

# Check accuracy for high traffic prediction
print("High Traffic Model Accuracy:", accuracy_score(y_test_traffic, y_pred_traffic))
print("High Traffic Model Report:\n", classification_report(y_test_traffic, y_pred_traffic))


print(f"Features Shape: {features.shape}")
print(f"Target Congestion Shape: {target_congestion.shape}")
print(f"Target High Traffic Shape: {target_high_traffic.shape}")
print(f"X_train Shape: {X_train.shape}")
print(f"y_train_congestion Shape: {y_train_congestion.shape}")
print(f"y_train_traffic Shape: {y_train_traffic.shape}")

# Utilization Prediction Model (optional)
model_utilization = RandomForestRegressor(random_state=42)
model_utilization.fit(X_train, features["Total Utilization"][:7431])
future_utilization = model_utilization.predict(X_test)


# Step 4: Add Predictions to Data
df["Predicted Congestion"] = model_congestion.predict(features)
df["Predicted High Traffic"] = model_traffic.predict(features)
df["Forecasted Utilization"] = model_utilization.predict(features)

# Step 5: Visualize on Map
# Initialize the map
mymap = folium.Map(location=[-6.2146, 106.8451], zoom_start=12)

# Add tower markers
for _, row in df.iterrows():
    source_coords = row["Source Coordinates"]
    print ({source_coords})
    if source_coords:
        icon_color = "blue" if "Fiber" in row["Tower ID"] else "black"
        folium.Marker(
            location=convert_float(source_coords),
            popup=(
                f"Tower ID: {row['Tower ID']}<br>Hops: {row['COUNT hop']}<br>"
                f"Links: {row['Link Qty']}"
            ),
            icon=folium.Icon(color=icon_color, icon="info-sign")
        ).add_to(mymap)

# Add routes with predictions
for _, route in df.iterrows():
    source_coords = route["Source Coordinates"]
    target_coords = route["Target Coordinates"]
    if source_coords and target_coords:
        # Color by prediction
        if route["Predicted Congestion"] == 1:
            line_color = "red"
        elif route["Predicted High Traffic"] == 1:
            line_color = "orange"
        else:
            line_color = "green"

        folium.PolyLine(
            locations=[convert_float(source_coords), convert_float(target_coords)],
            color=line_color,
            weight=2,
            popup=(
                f"Route: {route['Link 1']}<br>"
                f"Predicted Congestion: {route['Predicted Congestion']}<br>"
                f"Predicted High Traffic: {route['Predicted High Traffic']}"
            )
        ).add_to(mymap)

# Save the map to an HTML file
output_file = "route_path_optimization_map.html"
mymap.save(output_file)
print(f"Map saved to {output_file}")