import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd

# Friendly error page for invalid form input (e.g. negative numbers) 
from fastapi.exceptions import RequestValidationError
from fastapi.requests import Request as StarletteRequest


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Loan Eligibility Prediction API", version="1.0")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

model = joblib.load(os.path.join(BASE_DIR, "model", "loan_model.pkl"))


class LoanApplication(BaseModel):
    Gender: Literal["Male", "Female"]
    Married: Literal["Yes", "No"]
    Dependents: Literal["0", "1", "2", "3+"]
    Education: Literal["Graduate", "Not Graduate"]
    Self_Employed: Literal["Yes", "No"]
    ApplicantIncome: float = Field(..., ge=0)
    CoapplicantIncome: float = Field(..., ge=0)
    LoanAmount: float = Field(..., ge=0)
    Loan_Amount_Term: float = Field(..., ge=0)
    Credit_History: float = Field(..., ge=0, le=1)
    Property_Area: Literal["Urban", "Semiurban", "Rural"]


def run_prediction(data: dict):
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    return prediction, probability


# JSON API (Swagger /docs, other apps) 
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(application: LoanApplication):
    prediction, probability = run_prediction(application.dict())
    return {
        "loan_status": "Approved" if prediction == 1 else "Rejected",
        "approval_probability": round(float(probability), 4)
    }


# Web UI (human fills out a form in the browser) 
@app.get("/", response_class=HTMLResponse)
def show_form(request: Request):
    return templates.TemplateResponse(request, "index.html", {"result": None})


@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(
    request: Request,
    Gender: str = Form(...),
    Married: str = Form(...),
    Dependents: str = Form(...),
    Education: str = Form(...),
    Self_Employed: str = Form(...),
    ApplicantIncome: float = Form(..., ge=0),          
    CoapplicantIncome: float = Form(..., ge=0),        
    LoanAmount: float = Form(..., ge=0),                
    Loan_Amount_Term: float = Form(..., ge=0),          
    Credit_History: float = Form(..., ge=0, le=1),      
    Property_Area: str = Form(...),
):

    data = {
        "Gender": Gender, "Married": Married, "Dependents": Dependents,
        "Education": Education, "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome, "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount, "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History, "Property_Area": Property_Area,
    }
    prediction, probability = run_prediction(data)
    result = "Approved" if prediction == 1 else "Rejected"
    return templates.TemplateResponse(request, "index.html", {
        "result": result,
        "probability": round(float(probability) * 100, 2)
    })



@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: StarletteRequest, exc: RequestValidationError):
    if request.url.path == "/predict-form":
        messages = []
        for err in exc.errors():
            field = err["loc"][-1]
            messages.append(f"{field.replace('_', ' ')}: {err['msg']}")
        return templates.TemplateResponse(request, "index.html", {
            "result": None,
            "error": " | ".join(messages)
        }, status_code=200)
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=422, content={"detail": exc.errors()})