import mlflow
import mlflow.pytorch
from ultralytics import YOLO 
import os

# Config MLflow
mlflow.set_tracking_uri("sqlite:///mlruns.db")  # DB locale
mlflow.set_experiment("yolo-defect-detection")

def train_yolo():
    model = YOLO("yolov8n.pt")  # modello base
    with mlflow.start_run():
        # Log parametri
        mlflow.log_param("base_model", "yolov8n.pt")
        mlflow.log_param("epochs", 2)
        mlflow.log_param("img_size", 640)

        # Addestramento
        results = model.train(
            data="ml/data.yaml",
            epochs=2,
            imgsz=640,
            project="runs/train",
            name="exp"
        )

        # Log metriche principali
        metrics = results.results_dict
        for k, v in metrics.items():
            # Normalizzo i nomi delle metriche per MLflow. Avevo delle parentesi tonde e dava errore
            safe_key = (
                k.replace("(", "_")
                 .replace(")", "_")
                 .replace(" ", "_")
                 .replace("/", "_")
            )
            mlflow.log_metric(safe_key, v)


        # Salva il modello su MLflow registry
        best_model_path = model.ckpt_path  # checkpoint migliore
        mlflow.pytorch.log_model(
            pytorch_model=model.model,
            artifact_path="model"
        )

if __name__ == "__main__":
    train_yolo()
