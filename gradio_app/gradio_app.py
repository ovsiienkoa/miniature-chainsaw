import gradio as gr
from google.cloud.aiplatform import Endpoint

import json
import os
from parse_crypto import parse_crypto_directly
from parse_steam import parse_steam_directly
from case import Case


with open("train_to_artifact.json", "r") as f:
    cfg = json.load(f)

def prediction(entity, days):
    if entity == "BTC":
        data = parse_crypto_directly(days = 32)

    if entity == "CS | Dreams & Nightmares Case":
        entity = entity[5:]
        data = parse_steam_directly(name = entity, days = 32)

    data = Case.transform_dates(data)
    data.drop(columns = ['date'], inplace = True)
    data = data.values.tolist()

    endpoint = Endpoint(endpoint_name = f"projects/{cfg["PROJECT"]}/locations/europe-west9/endpoints/{cfg["ENDPOINT"]}")

    entity = entity.replace(" ", "_")
    entity = entity.replace("&", "_")

    request = [{"entity":entity, "days": int(days), "data": data}]
    result = endpoint.predict(instances = request)
    result = result.predictions[0]
    return result

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    demo = gr.Interface(
        prediction,
        [gr.Radio(["BTC", "CS | Dreams & Nightmares Case"], label="entity"),
         gr.Slider(1, 5, value=1, label="Days to predict", step=1)],
        "text"
    )
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )