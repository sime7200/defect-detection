from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import PILToTensor, ToPILImage
from torchvision.utils import draw_segmentation_masks
import io
import os

# Inizializza API
app = FastAPI(title="Defect Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in produzione puoi mettere solo ["http://127.0.0.1:5500"] o simili
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Percorso del modello .pt già sul tuo PC
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yolo_model_path = os.path.join(BASE_DIR, "best_ingranaggioYOLO.pt")
mask2former_checkpoint_path = r"C:\Users\web\Desktop\defect-detection\checkpoint-4640"

# Loading YOLO
print(f"Loading YOLO model from: {yolo_model_path}")
model = YOLO(yolo_model_path)

# Loading Mask2Former
print("Loading Mask2Former model...")
processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
mask2former_model = Mask2FormerForUniversalSegmentation.from_pretrained(mask2former_checkpoint_path)

# Endpoint di prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  model_type: str = Query("yolo", enum=["yolo", "mask2former"]),
                  return_image: bool = False):
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    if model_type == "yolo":
        print("Eseguendo predizione con YOLO...")
        results = model.predict(image)

        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "bbox": box.xyxy[0].tolist(),
                    "confidence": float(box.conf[0]),
                    "class": int(box.cls[0])
                })

        if return_image:
            plotted = results[0].plot()
            pil_image = Image.fromarray(plotted[..., ::-1])
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")  # 🔑 manca questo!
                    
    elif model_type == "mask2former":
        print("Eseguendo predizione con Mask2Former...")
        if mask2former_model is None or processor is None:
            return JSONResponse(content={"error": "Mask2Former non è caricato"}, status_code=500)

        buf = predict_mask2former(image, mask2former_model, processor, threshold=0.2)
        return StreamingResponse(buf, media_type="image/png")
    

def predict_mask2former(image: Image.Image, model, processor, threshold: float = 0.5) -> bytes:
    """
    Esegue inferenza con Mask2Former e restituisce immagine con maschere disegnate come bytes PNG.
    """
    # Preprocessa immagine
    inputs = processor(images=image, return_tensors="pt").to(model.device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # Logits
    pred_masks = outputs.masks_queries_logits[0]    # [num_queries, H, W]
    pred_logits = outputs.class_queries_logits[0]   # [num_queries, num_classes+1]

    # Filtra background
    scores = pred_logits.softmax(-1)
    labels = scores.argmax(-1)
    keep = (labels != model.config.num_labels) & (scores.max(-1).values > threshold)

    filtered_masks = pred_masks[keep]
    if filtered_masks.shape[0] == 0:
        # Nessuna maschera = ritorno immagine originale
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        return buf

    # Resize maschere a dimensioni originali
    target_height, target_width = image.height, image.width
    filtered_masks = F.interpolate(
        filtered_masks.unsqueeze(0),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    # Binarizza
    masks_bin = (filtered_masks > 0.5).cpu()

    # Disegno su immagine
    image_tensor = PILToTensor()(image)
    overlay = draw_segmentation_masks(image_tensor, masks=masks_bin, alpha=0.5, colors="red")

    # Converti in PNG
    overlay_pil = ToPILImage()(overlay)
    buf = io.BytesIO()
    overlay_pil.save(buf, format="PNG")
    buf.seek(0)
    return buf



# Servire HTML (frontend)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def frontend():
    return FileResponse("app/static/index.html")
