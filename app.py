from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

templates = Jinja2Templates(
    directory=os.path.join(BASE_DIR, "templates")
)

MODEL_PATH = os.path.join(BASE_DIR, "house_price_model.pkl")

model = joblib.load(MODEL_PATH)


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/predict")
def predict(
    request: Request,
    area: float = Form(...),
    bedrooms: int = Form(...),
    bathrooms: int = Form(...)
):
    input_data = pd.DataFrame(
        [[area, bedrooms, bathrooms]],
        columns=["Area", "Bedrooms", "Bathrooms"]
    )

    prediction = model.predict(input_data)

    price = round(float(prediction[0]), 2)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"prediction": price}
    )