from typing import List, Dict
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
import joblib

# Load the ExtraTreesClassifier model
model = joblib.load("model/best_model.joblib")

def get_features_dict(model):
    """
    Retrieve feature names and their types (defaulting to float) for the ExtraTreesClassifier model.
    """
    if hasattr(model, "feature_names_in_"):
        # Map each feature name to a type annotation
        return {name: (float, ...) for name in model.feature_names_in_}
    else:
        raise AttributeError("The model does not have the 'feature_names_in_' attribute.")

def create_input_features_class(model):
    """
    Dynamically create the InputFeatures class based on model features.
    """
    features_dict = get_features_dict(model)
    # Use create_model to dynamically generate a Pydantic class
    return create_model("InputFeatures", **features_dict)

# Dynamically generate the Pydantic model for input validation
try:
    InputFeatures = create_input_features_class(model)
except Exception as e:
    raise RuntimeError(f"Failed to create InputFeatures model: {e}")

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict", response_model=List[float])
async def predict_post(datas: List[InputFeatures]):
    """
    Predict endpoint for batch predictions.
    """
    try:
        # Convert input data into a NumPy array
        input_data = np.asarray([list(data.dict().values()) for data in datas])
        
        # Make predictions
        predictions = model.predict(input_data).tolist()
        return predictions
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {e}")

if __name__ == "__main__":
    # Debug: Print feature names at startup
    if hasattr(model, "feature_names_in_"):
        print("Feature names:", model.feature_names_in_)
    else:
        print("Model does not provide feature names.")
    uvicorn.run(app, host="0.0.0.0", port=8080)
