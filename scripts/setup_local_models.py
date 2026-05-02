#!/usr/bin/env python3
import os
import urllib.request
import zipfile
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np

def setup_models():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # 1. Download Xception weights if not present
    xception_path = models_dir / "full_c23.p"
    if not xception_path.exists():
        print("Downloading FaceForensics++ models zip (this may take a while)...")
        zip_path = models_dir / "faceforensics_models.zip"
        
        try:
            urllib.request.urlretrieve(
                "http://kaldir.vc.in.tum.de/FaceForensics/models/faceforensics++_models.zip",
                zip_path
            )
            print("Extracting full_c23.p...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Find full_c23.p in the zip
                for file_info in zip_ref.infolist():
                    if file_info.filename.endswith("full_c23.p"):
                        # Extract just this file
                        file_info.filename = "full_c23.p"
                        zip_ref.extract(file_info, models_dir)
                        break
            
            # Clean up zip
            os.remove(zip_path)
            print(f"Xception weights saved to {xception_path}")
        except Exception as e:
            print(f"Failed to download Xception weights: {e}")
            print("Please manually download from http://kaldir.vc.in.tum.de/FaceForensics/models/faceforensics++_models.zip")
    else:
        print(f"Xception weights already exist at {xception_path}")

    # 2. Create Dummy Fusion Model
    fusion_path = models_dir / "fusion_lr.pkl"
    if not fusion_path.exists():
        print("Creating fallback dummy fusion_lr.pkl...")
        # Create a dummy logistic regression model
        lr = LogisticRegression()
        # X: [Ss, Ts], y: [0, 1]
        X = np.array([[0.1, 0.1], [0.9, 0.9]])
        y = np.array([0, 1])
        lr.fit(X, y)
        # Force the coefficients to roughly match configs/fusion_weights.yaml
        lr.coef_ = np.array([[0.65 * 10, 0.35 * 10]])
        lr.intercept_ = np.array([-5.0]) # Set intercept so 0.5 threshold works reasonably
        lr.classes_ = np.array([0, 1])
        
        joblib.dump(lr, fusion_path)
        print(f"Dummy fusion model saved to {fusion_path}")
    else:
        print(f"Fusion model already exists at {fusion_path}")

if __name__ == "__main__":
    setup_models()
