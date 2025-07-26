import gradio as gr
from google.cloud.aiplatform import Endpoint

import json
from parse_crypto import parse_crypto_directly
from case import Case


with open("train_to_artifact.json", "r") as f:
    cfg = json.load(f)

def prediction(entity, days):
    if entity == "BTC":
        data = Case.transform_dates(parse_crypto_directly(days = 32))
        data.drop(columns = ['date'], inplace = True)
        data = data.values.tolist()

    endpoint = Endpoint(endpoint_name = f"projects/{cfg["PROJECT"]}/locations/europe-west9/endpoints/{cfg["ENDPOINT"]}")

    request = [{"entity":entity, "days": int(days), "data": data}]
    result = endpoint.predict(instances = request)
    result = result.predictions[0]
    return result

demo = gr.Interface(
    prediction,
    [gr.Radio(["BTC"], label="entity"),
    gr.Slider(1, 5, value=1, label="Days to predict", step=1)],
    "text"
)

demo.launch(share = True)