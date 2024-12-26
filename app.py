# --------------------------------------------------------------------------------------------------------------------------
## ====================================== FastAPI Code =====================================================================
# from fastapi import FastAPI, Form, HTTPException
# from ultralytics import YOLO
# from PIL import Image
# import numpy as np

# # Initialize the FastAPI app
# app = FastAPI(debug=True)

# # Initialize the YOLO model
# model = YOLO('./models/model_yolo-v1.pt')

# @app.post('/Chest_Xray_Detection')
# async def Chest_Xray_Detection(image_url: str = Form(...)):
    
#     try:
#         if not image_url:
#             raise HTTPException(status_code=400, detail="Bad Request: 'image_url' is required.")
        
#         else:   
#             image = Image.open(image_url)
            
#             # Perform prediction
#             results = model.predict(image)
#             names_dict = results[0].names
#             probs = results[0].probs.data.tolist()

#             # Extract the top prediction
#             prediction = names_dict[np.argmax(probs)]
#             probability = round(np.max(probs) * 100, 3)
            
#             # Return the prediction as a JSON response
#             return {'Prediction': prediction, 'Probability': probability}
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail='Failed to Search Data')

# --------------------------------------------------------------------------------------------------------------------------
## ====================================== FastAPI Code =====================================================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Initialize the FastAPI app
app = FastAPI(debug=True)

# Initialize the YOLO model
model = YOLO('./models/model_yolo-v1.pt')

@app.post('/Chest_Xray_Detection')
async def Chest_Xray_Detection(file: UploadFile = File(...)):
    try:
        # Open the uploaded image
        image = Image.open(file.file)

        # Perform prediction
        results = model.predict(image)
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()

        # Extract the top prediction
        prediction = names_dict[np.argmax(probs)]
        probability = round(np.max(probs) * 100, 3)

        # Return the prediction as a JSON response
        return {'Prediction': prediction, 'Probability': probability}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process the image: {str(e)}")
