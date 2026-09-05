"""Contratos Pydantic de la API: entrada (transaccion) y salida (prediccion)"""

from enum import Enum

from pydantic import BaseModel, Field


class Category(str, Enum):
    entertainment = "entertainment"
    food_dining = "food_dining"
    gas_transport = "gas_transport"
    grocery_net = "grocery_net"
    grocery_pos = "grocery_pos"
    health_fitness = "health_fitness"
    home = "home"
    kids_pets = "kids_pets"
    misc_net = "misc_net"
    misc_pos = "misc_pos"
    personal_care = "personal_care"
    shopping_net = "shopping_net"
    shopping_pos = "shopping_pos"
    travel = "travel"


class Gender(str, Enum):
    F = "F"
    M = "M"


class Transaction(BaseModel):
    amt: float = Field(..., gt=0, description="Monto de la transacción", examples=[120.5])
    category: Category = Field(..., description="Rubro del comercio", examples=["grocery_pos"])
    gender: Gender = Field(..., description="Genero del titular", examples=["F"])
    city_pop: int = Field(..., ge=0, description="Población de la ciudad del titular", examples=[50000])
    lat: float = Field(..., ge=-90, le=90, description="Latitud del titular", examples=[40.1])
    long: float = Field(..., ge=-180, le=180, description="Longitud del titular", examples=[-74.5])
    merch_lat: float = Field(..., ge=-90, le=90, description="Latitud del comercio", examples=[40.3])
    merch_long: float = Field(..., ge=-180, le=180, description="Longitud del comercio", examples=[-74.2])
    hour: int = Field(..., ge=0, le=23, description="Hora del día de la transacción", examples=[2])
    age: int = Field(..., ge=0, le=120, description="Edad del titular", examples=[35])

    model_config = {"extra": "forbid"}


class PredictionResponse(BaseModel):
    is_fraud: bool = Field(..., description="True si el modelo predice fraude")
    probability: float = Field(..., description="Probabilidad estimada de fraude (0 a 1)")
    model_version: str = Field(..., description="Versión del modelo que respondió")


class HealthResponse(BaseModel):
    status: str
    model_version: str
