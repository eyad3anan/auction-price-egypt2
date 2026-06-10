"""app/main.py — FastAPI prediction service for Egyptian Auction Price Predictor."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

app = FastAPI(
    title="Egyptian Auction Price Predictor",
    description="""
Predicts the **final selling price (EGP)** for items listed on Egyptian online auctions.

## Model
Ensemble of Random Forest + LightGBM + XGBoost (log-target blend)  
**Test R² = 0.9377 | RMSLE = 0.2655 | Bundle size = 4.6 MB**

## Features Guide

| Feature | Type | Description | Allowed Values |
|---------|------|-------------|----------------|
| `category` | string | Main product category | Electronics, Fashion, Home & Garden, Sports, Vehicles, Books, Collectibles, Other |
| `subcategory` | string | Sub-category within main category | Any subcategory string from the dataset |
| `brand` | string | Product brand name | Any brand string (e.g. Apple, Samsung, Generic) |
| `condition` | string | Physical condition of item | For Parts, Poor, Fair, Good, Very Good, Excellent, Like New, New |
| `product_age` | int | Age of product in months | 0 to 240 (0 = brand new, 240 = 20 years old) |
| `starting_price` | float | Starting bid price in EGP | Any positive number (e.g. 100.0, 5000.0) |
| `auction_duration` | int | Duration of auction in days | 1 to 30 |
| `listing_day_of_week` | string | Day the auction was listed | Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday |
| `listing_hour` | int | Hour of day the listing was posted (24h) | 0 to 23 |
| `seller_rating` | float | Seller rating score | 0.0 to 5.0 |
| `seller_total_sales` | int | Total number of sales by seller | 0 to 10000+ |
| `seller_account_age` | int | Seller account age in months | 0 to 240 |
| `verified_seller` | int | Whether seller is verified | 0 (not verified) or 1 (verified) |
    """,
    version="1.0.0"
)

class AuctionListing(BaseModel):
    category: str = Field(
        ...,
        example="Electronics",
        description="Main product category. Options: Electronics, Fashion, Home & Garden, Sports, Vehicles, Books, Collectibles, Other"
    )
    subcategory: str = Field(
        ...,
        example="Laptops",
        description="Sub-category within the main category (e.g. Laptops, Shirts, Furniture)"
    )
    brand: str = Field(
        ...,
        example="Apple",
        description="Product brand name (e.g. Apple, Samsung, Sony, Generic)"
    )
    condition: str = Field(
        ...,
        example="Like New",
        description="Physical condition. Must be one of: For Parts | Poor | Fair | Good | Very Good | Excellent | Like New | New"
    )
    product_age: int = Field(
        ..., ge=0, le=240,
        example=12,
        description="Age of product in months. 0 = brand new, 12 = 1 year old, 24 = 2 years old"
    )
    starting_price: float = Field(
        ..., gt=0,
        example=5000.0,
        description="Starting auction bid price in EGP. Must be positive."
    )
    auction_duration: int = Field(
        ..., ge=1, le=30,
        example=7,
        description="How many days the auction runs. Range: 1 to 30 days."
    )
    listing_day_of_week: str = Field(
        ...,
        example="Saturday",
        description="Day the auction was listed. Options: Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday"
    )
    listing_hour: int = Field(
        ..., ge=0, le=23,
        example=20,
        description="Hour of day the listing was posted (24-hour format). 0=midnight, 12=noon, 20=8pm"
    )
    seller_rating: float = Field(
        ..., ge=0.0, le=5.0,
        example=4.5,
        description="Seller rating from 0.0 (worst) to 5.0 (best)"
    )
    seller_total_sales: int = Field(
        ..., ge=0,
        example=50,
        description="Total number of completed sales by this seller. 0 = new seller."
    )
    seller_account_age: int = Field(
        ..., ge=0,
        example=24,
        description="How old the seller account is in months. 0 = brand new account."
    )
    verified_seller: int = Field(
        ..., ge=0, le=1,
        example=1,
        description="Whether the seller is verified. 0 = not verified, 1 = verified."
    )

class PredictionResponse(BaseModel):
    predicted_final_selling_price_egp: float
    currency: str = "EGP"
    model: str = "RF + LightGBM + XGBoost Ensemble"
    model_version: str = "1.0.0"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str

@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Egyptian Auction Price Predictor",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict"
    }

@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    try:
        from scripts.predict import load_pipeline
        _, _, _, _, _, bundle = load_pipeline()
        return HealthResponse(status="ok", model_loaded=True, model_type=bundle["type"])
    except Exception as e:
        return HealthResponse(status=f"error: {e}", model_loaded=False, model_type="unknown")

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(listing: AuctionListing):
    """
    Predict the final selling price for an auction listing.

    **All 13 features are required.** See the schema below for allowed values.

    Returns the predicted final selling price in EGP.
    """
    try:
        from scripts.predict import predict_single
        price = predict_single(listing.model_dump())
        return PredictionResponse(predicted_final_selling_price_egp=price)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")