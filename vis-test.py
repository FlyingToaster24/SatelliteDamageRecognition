import matplotlib.pyplot as plt
from PIL import Image
import json
import matplotlib.patches as patches
from shapely import wkt

from training_model import parse_json

name = "palu-tsunami_00000124_post_disaster"

json_name = "data/labels/" + name + ".json"

d = open(json_name, "r")

image = Image.open("data/images/" + name + ".png").convert('RGB')

fig, ax = plt.subplots(1)
ax.imshow(image)

data = json.load(d)

for feature in data["features"]["xy"]:
    class_label_to_color = {
        'building-no-damage': 'green',
        'building-unknown': 'yellow',
        'building-un-classified': 'blue',
        'building-minor-damage': 'orange',
        'building-major-damage': 'red',
        'building-destroyed': 'white'
    }

    shape = wkt.loads(feature["wkt"])
    feature_type = feature["properties"]["feature_type"] + \
        "-" + feature["properties"]["subtype"]

    if shape.geom_type == "Polygon" and feature_type in class_label_to_color:
        rect = patches.Rectangle((shape.bounds[0], shape.bounds[1]), shape.bounds[2] - shape.bounds[0],
                                 shape.bounds[3] - shape.bounds[1], linewidth=1, edgecolor=class_label_to_color[feature_type], fill=False)
        ax.add_patch(rect)


plt.show()
