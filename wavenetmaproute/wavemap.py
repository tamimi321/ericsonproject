from flask import Flask, render_template
import folium

app = Flask(__name__)

# Sample data
towers = [
    {"id": 1, "type": "fiber", "name": "Tower 1", "location": [-7.667310, 109.667954]},
    {"id": 2, "type": "microwave", "name": "Tower 2", "location": [-7.674200, 109.662117]},
    {"id": 3, "type": "microwave", "name": "Tower 3", "location": [-7.682111, 109.667280]},
    {"id": 4, "type": "fiber", "name": "Tower 4", "location": [-7.682111, 109.667267]},
    {"id": 5, "type": "microwave", "name": "Tower 5", "location": [-7.675391, 109.655680]},
    {"id": 6, "type": "fiber", "name": "Tower 6", "location": [-7.687895, 109.661431]},
    {"id": 7, "type": "microwave", "name": "Tower 7", "location": [-7.690192, 109.670786]},
    {"id": 8, "type": "fiber", "name": "Tower 8", "location": [-7.686789, 109.677223]},
    {"id": 9, "type": "fiber", "name": "Tower 9", "location": [-7.666715, 109.668640]},
    {"id": 10, "type": "fiber", "name": "Tower 10", "location": [-7.666715, 109.669640]},
]

routes = [
    {"start": 1, "end": 2, "type": "existing", "traffic": "high"},
    {"start": 2, "end": 3, "type": "optimal", "traffic": "low"},
    {"start": 1, "end": 3, "type": "alternate", "traffic": "high"},
    {"start": 4, "end": 5, "type": "existing", "traffic": "high"},
    {"start": 5, "end": 6, "type": "existing", "traffic": "low"},
    {"start": 7, "end": 9, "type": "existing", "traffic": "low"},
    {"start": 8, "end": 6, "type": "alternate", "traffic": "low"},
    {"start": 6, "end": 5, "type": "optimal", "traffic": "low"},
    {"start": 8, "end": 9, "type": "congested", "traffic": "low"},
    {"start": 9, "end": 7, "type": "alternate", "traffic": "low"},
    {"start": 10, "end": 1, "type": "existing", "traffic": "low"},
]

@app.route('/map')
def map():
    return render_template("map.html")

@app.route('/')
def index():
    # Initialize map
    map = folium.Map(location=[-7.6784533,109.6384279], zoom_start=13)

    # Add towers
    for tower in towers:
        icon = folium.Icon(
            icon="tower-cell" if tower["type"] == "microwave" else "tower-broadcast",prefix="fa",
            color="blue" if tower["type"] == "fiber" else "orange",
        )
        folium.Marker(location=tower["location"], popup=tower["name"], icon=icon).add_to(map)

    # Add routes
    for route in routes:
        start_tower = next(t for t in towers if t["id"] == route["start"])
        end_tower = next(t for t in towers if t["id"] == route["end"])
        line_style = {
            "existing": {"color": "black", "dash_array": None},
            "optimal": {"color": "green", "dash_array": None},
            "alternate": {"color": "gray", "dash_array": "10, 10"},
            "congested": {"color": "red", "dash_array": "10, 10"},
        }

        folium.PolyLine(
            locations=[start_tower["location"], end_tower["location"]],
            color=line_style[route["type"]]["color"],
            dash_array=line_style[route["type"]]["dash_array"],
            weight=2.5,
            tooltip=f"{route['type'].capitalize()} Route",
        ).add_to(map)

        # Add traffic icons to existing routes
        if route["type"] == "existing" and route["traffic"]:
            midpoint = [
                (start_tower["location"][0] + end_tower["location"][0]) / 2,
                (start_tower["location"][1] + end_tower["location"][1]) / 2,
            ]
            icon = folium.Icon(
                icon="warning" if route["traffic"] == "high" else "check",prefix= 'fa',
                color="yellow" if route["traffic"] == "high" else "green",
            )
            folium.Marker(location=midpoint,popup=route["traffic"], icon=icon).add_to(map)

    # Save the map as an HTML file
    map.save("templates/map.html")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
