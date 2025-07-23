import gradio as gr

from parse_crypto import parse_crypto_directly
from case import Case

def prediction(entity, days):
    if entity == "BTC":
        data = Case.transform_dates(parse_crypto_directly(days = 32))
        data.drop(columns = ['date'], inplace = True)
        data = data.to_numpy()

    #get endpoint

    #result = endpoint({"entity":entity, "days":days, "data":data})
    return result

demo = gr.Interface(
    prediction,
    [gr.Radio(["BTC", "CS2 | Fracture Case"], label="entity"),
    gr.Slider(1, 5, value=1, label="Days to predict")],
    "text"
)

demo.launch()  # Share your demo with just 1 extra parameter 🚀