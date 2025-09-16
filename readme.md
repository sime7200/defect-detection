# 🛠️ Defect Detection Project

This project shows an end-to-end workflow for detecting defects (cracks, folds, imperfections) on mechanical parts (gears) using **YOLOv8** (object detection) and **Mask2Former** (instance segmentation).
It integrates **MLOps** capabilities for experiment tracking, model registration, and serving via API and frontend.

# 📦 Technologies used

- [**YOLOv8**](https://github.com/ultralytics/ultralytics) → Defect detection with bounding boxes
- [**Mask2Former**](https://github.com/facebookresearch/Mask2Former) → Instance segmentation with masks
- [**MLflow**](https://github.com/mlflow/mlflow) →  Tracking, metrics, model registry
- [**FastAPI**](https://github.com/fastapi/fastapi) → Serving API  
- **Frontend HTML/JS** → Simple interface for image uploads
- [**Docker**](https://github.com/docker) → Scalable deployment  


---

# 🖼️ Demo
## YOLOv8 – Bounding Box
![YOLO predictions](screenshots_mlFlow\pred_YOLO.png)

## Mask2Former – Maschere segmentazione
![Mask2former predictions](screenshots_mlFlow\pred_mask2former.png)

## MLflow Tracking
The training shown here has been reduced to 2 epochs for demonstration purposes only. The final model used in production was trained for 30 epochs.
![title](screenshots_mlFlow\general-results.png)

---

# 🔍 How does it work?

- Upload an image via the web UI
- Choose which model to use (**YOLO** or **Mask2Former**)
- Get back predictions:
  - YOLO → bounding boxes (/predict?model=yolo)
  - Mask2Former → segmentation masks (/predict?model=mask2former)

---

# 📊 Experiments

- YOLOv8 fine-tuning with 30 epochs → model registered on MLflow. (Metrics on MLflow are just about 2 epochs)
- Mask2Former fine-tuning → logged on MLflow with IoU and loss metrics
- Model versioning in MLflow Model Registry (e.g., Production, Staging)
![MLflow dashboard](screenshots_mlFlow\dashboard.png)
---

# 📚 Conclusions

This project is linked to my thesis project, which aimed to fine-tune open-source models (taken from Hugging Face such as DETR or Mask2Former) to replace industrial software in the company where I did my internship.
Here, I use some final models to combine Computer Vision and MLOps in a realistic case:
- CV model development and fine-tuning (YOLOv8 and Mask2Former reported)
- Tracking experiments with MLflow
- API and frontend for real testing
- Scalable structure for possible cloud deployment