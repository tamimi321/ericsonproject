import pandas as pd
from math import sin, cos, sqrt, atan2, radians
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import gmplot

# Google Maps API key
GMAPS_API_KEY = "AIzaSyCF-PhIuh4xM7kFuG0BLaDrj69r_5KBIKI"

# # Load Excel file
# file_path = r'C:\Users\ahmad\Documents\tamz\My Proj\ericson\data\Copy of Decongestion-2024W44H_2024_Central_New.xlsb'
# df = pd.read_excel(file_path)

# # Clean column names
# df.columns = df.columns.str.strip()

# # Check column names
# print("Columns in DataFrame:", df.columns)

# # Add missing columns if necessary
# if "COUNT hop" not in df.columns:
#     df["COUNT hop"] = 1
# if "Link Qty" not in df.columns:
#     df["Link Qty"] = 1

# # Features for machine learning
# features = df[["COUNT hop", "Link Qty", "Util Link 1", "Util Link 2"]]
# print(features.head())

# Load the dataset
data = {
    "Tower ID": ["JAW-JB-CJR-0475", "JAW-JB-CJR-4320", "JAW-JB-CJR-4322", "JAW-JB-CJR-4341", "JAW-JB-PRG-3881"],
    "Route Path": [
        "JAW-JB-CJR-0547/JAW-JB-CJR-0579/JAW-JB-CJR-0524",
        "JAW-JB-CJR-0547/JAW-JB-CJR-0579/JAW-JB-CJR-0524",
        "JAW-JB-CJR-0547/JAW-JB-CJR-0579/JAW-JB-CJR-0524",
        "JAW-JB-CJR-0641/JAW-JB-CJR-3288/JAW-JB-CJR-0687",
        "JAW-JB-CMS-0806/JAW-JB-CMS-0834/JAW-JB-PRG-0831"
    ],
    "Link 1": ["JAW-JB-CJR-0547/JAW-JB-CJR-0579", "JAW-JB-CJR-0547/JAW-JB-CJR-0579", "JAW-JB-CJR-0547/JAW-JB-CJR-0579",
               "JAW-JB-CJR-0641/JAW-JB-CJR-3288", "JAW-JB-CMS-0806/JAW-JB-CMS-0834"],
    "Util Link 1": [67.06, 67.06, 67.06, 99.97, 66.07],
    "Link 2": ["JAW-JB-CJR-0579/JAW-JB-CJR-0524", "JAW-JB-CJR-0579/JAW-JB-CJR-0524", "JAW-JB-CJR-0579/JAW-JB-CJR-0524",
               "JAW-JB-CJR-3288/JAW-JB-CJR-0687", "JAW-JB-CMS-0834/JAW-JB-PRG-0831"],
    "Util Link 2": [5.55, 5.55, 5.55, 7.68, 29.74],
    "COUNT hop": [4, 2, 4, 4, 4],
    "Link Qty": [4, 2, 4, 4, 4]
}

df = pd.DataFrame(data)

# Assign coordinates to towers (example coordinates)
tower_coordinates = {
    "JAW-JB-CJR-0475": (37.7749, -122.4194),
    "JAW-JB-CJR-4320": (37.8044, -122.2711),
    "JAW-JB-CJR-4322": (37.6879, -122.4702),
    "JAW-JB-CJR-4341": (37.7833, -122.4167),
    "JAW-JB-PRG-3881": (37.7648, -122.463),
    "JAW-JB-CJR-0547": (37.7512, -122.4478),
    "JAW-JB-CJR-0579": (37.7693, -122.4294),
    "JAW-JB-CJR-0524": (37.7873, -122.4098),
    "JAW-JB-CJR-0641": (37.7934, -122.3988),
    "JAW-JB-CJR-3288": (37.7802, -122.4096),
    "JAW-JB-CJR-0687": (37.7666, -122.4522),
    "JAW-JB-CMS-0806": (37.7551, -122.4681),
    "JAW-JB-CMS-0834": (37.7513, -122.4627),
    "JAW-JB-PRG-0831": (37.7455, -122.4752)
}

# Categorize links by utilization thresholds
threshold_high_traffic = 50  # High traffic utilization threshold
threshold_congested = 75  # Congestion threshold

df["High Traffic 1"] = df["Util Link 1"] > threshold_high_traffic
df["High Traffic 2"] = df["Util Link 2"] > threshold_high_traffic
df["Congested 1"] = df["Util Link 1"] > threshold_congested
df["Congested 2"] = df["Util Link 2"] > threshold_congested

# Plot data on Google Maps
gmap = gmplot.GoogleMapPlotter(37.7749, -122.4194, 12, apikey=GMAPS_API_KEY)

# Add tower markers
for tower_id, (lat, lon) in tower_coordinates.items():
    gmap.marker(lat, lon, title=tower_id)

# Plot routes
for _, row in df.iterrows():
    for link, congested, high_traffic in [
        (row["Link 1"], row["Congested 1"], row["High Traffic 1"]),
        (row["Link 2"], row["Congested 2"], row["High Traffic 2"])
    ]:
        source, target = link.split("/")
        if source in tower_coordinates and target in tower_coordinates:
            source_coords = tower_coordinates[source]
            target_coords = tower_coordinates[target]
            color = "red" if congested else "purple" if high_traffic else "blue"
            gmap.plot(
                [source_coords[0], target_coords[0]],
                [source_coords[1], target_coords[1]],
                color=color,
                edge_width=2
            )

# Prepare data for machine learning
features = df[["COUNT hop", "Link Qty", "Util Link 1", "Util Link 2"]]
targets = df["Util Link 1"]  # Predicting Util Link 1 as an example

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Train a regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Model Mean Squared Error: {mse}")

# Predict future utilization for all links
df["Forecasted Util Link 1"] = model.predict(features)

# Categorize predicted data
df["Forecasted High Traffic 1"] = df["Forecasted Util Link 1"] > threshold_high_traffic
df["Forecasted Congested 1"] = df["Forecasted Util Link 1"] > threshold_congested

# Visualize updated data with forecasted categories
gmap = gmplot.GoogleMapPlotter(37.7749, -122.4194, 12, apikey=GMAPS_API_KEY)

# Add tower markers
for tower_id, (lat, lon) in tower_coordinates.items():
    gmap.marker(lat, lon, title=tower_id)

# Plot existing and forecasted routes
for _, row in df.iterrows():
    for link, congested, high_traffic, forecast_congested, forecast_high_traffic in [
        (row["Link 1"], row["Congested 1"], row["High Traffic 1"],
         row["Forecasted Congested 1"], row["Forecasted High Traffic 1"]),
        (row["Link 2"], row["Congested 2"], row["High Traffic 2"],
         False, False)  # Assuming no forecast for Link 2 in this example
    ]:
        source, target = link.split("/")
        if source in tower_coordinates and target in tower_coordinates:
            source_coords = tower_coordinates[source]
            target_coords = tower_coordinates[target]

            # Choose color based on forecast or existing conditions
            if forecast_congested:
                color = "orange"  # Predicted congestion
            elif forecast_high_traffic:
                color = "green"  # Predicted high traffic
            elif congested:
                color = "red"  # Existing congestion
            elif high_traffic:
                color = "purple"  # Existing high traffic
            else:
                color = "blue"  # Normal traffic

            gmap.plot(
                [source_coords[0], target_coords[0]],
                [source_coords[1], target_coords[1]],
                color=color,
                edge_width=2
            )

# Save the updated map with forecasts
output_file_forecast = r"C:\Users\ahmad\Documents\tamz\My Proj\ericson\data\tower_routes_forecast_map.html"
gmap.draw(output_file_forecast)

output_file_forecast

# Save map to HTML file
output_file = r"C:\Users\ahmad\Documents\tamz\My Proj\ericson\data\tower_routes_map.html"
gmap.draw(output_file)

output_file
