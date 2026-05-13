from fastapi import FastAPI

from pydantic import BaseModel

from prediction import predict

app = FastAPI(title="Loan Defaulter Prediction API")

class LoanDefaultInput(BaseModel):
    loanLimit: int
    gender: int
    approvInAdv: int
    loanType: int
    loanPurpose: int
    creditWorthiness: int
    openCredit: int
    businessOrCommercial: int
    loanAmount: int
    rateOfInterest: float
    interestRateSpread: float
    term: int
    negAmmortization: int
    interestOnly: int
    lumpSumPayment: int
    propertyValue: float
    constructionType: int
    occupancyType: int
    securedBy: int
    totalUnits: int
    income: int
    creditType: int
    creditScore: int
    coApplicantCreditType: int
    age: int
    submissionOfApplication: int
    ltv: float
    region: int
    securityType: int
    dtir1: float

@app.get("/")
def home():
    return {"message": "Loan Default Prediction API is running."}

@app.post("/predict")
def predict_diabetes(input_data: LoanDefaultInput):
    ordered_data = {
        "loanLimit": input_data.loanLimit,
        "gender": input_data.gender,
        "approvInAdv": input_data.approvInAdv,
        "loanType": input_data.loanType,
        "loanPurpose": input_data.loanPurpose,
        "creditWorthiness": input_data.creditWorthiness,
        "openCredit": input_data.openCredit,
        "businessOrCommercial": input_data.businessOrCommercial,
        "loanAmount": input_data.loanAmount,
        "rateOfInterest": input_data.rateOfInterest,
        "interestRateSpread": input_data.interestRateSpread,
        "term": input_data.term,
        "negAmmortization": input_data.negAmmortization,
        "interestOnly": input_data.interestOnly,
        "lumpSumPayment": input_data.lumpSumPayment,
        "propertyValue": input_data.propertyValue,
        "constructionType": input_data.constructionType,
        "occupancyType": input_data.occupancyType,
        "securedBy": input_data.securedBy,
        "totalUnits": input_data.totalUnits,
        "income": input_data.income,
        "creditType": input_data.creditType,
        "creditScore": input_data.creditScore,
        "coApplicantCreditType": input_data.coApplicantCreditType,
        "age": input_data.age,
        "submissionOfApplication": input_data.submissionOfApplication,
        "ltv": input_data.ltv,
        "region": input_data.region,
        "securityType": input_data.securityType,
        "dtir1": input_data.dtir1
    }
    result = predict(ordered_data)
    return result

