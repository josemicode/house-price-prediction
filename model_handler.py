import os
import joblib

class ModelHandler:
    def __init__(self):
        folder_path = '/models'
        folder_walk = os.walk(folder_path)
        self.model = any
        try:
            _, _, files = next(folder_walk)
            if files:
                first = files[0]
                model_path = os.path.join(folder_path, first)
                self.model = joblib.load(model_path)
            else:
                print(f"No files found in {folder_path}")
        except StopIteration:
            print("Inaccesible folder, might not exist")
    
    def predict(self, features):
        return self.model.predict([features])