import mlflow
import mlflow.pytorch
from ultralytics import YOLO

mlflow.set_tracking_uri("sqlite:///mlruns.db") 
mlflow.set_experiment("yolo-defect-detection")

def register_model(weight_path, version_name, stage=None):
    model = YOLO(weight_path)

    with mlflow.start_run() as run:
        mlflow.log_param("weights_path", weight_path)
        mlflow.pytorch.log_model(
            pytorch_model=model.model,
            artifact_path="model",
            registered_model_name="DefectDetectionModel"
        )

        print(f"Run ID: {run.info.run_id} - registered as {version_name}")

        # (Staging/Production)
        if stage:
            client = mlflow.tracking.MlflowClient()
            latest_versions = client.get_latest_versions("DefectDetectionModel")
            for v in latest_versions:
                if v.run_id == run.info.run_id:
                    client.transition_model_version_stage(
                        name="DefectDetectionModel",
                        version=v.version,
                        stage=stage
                    )
                    print(f"Model {version_name} set to stage: {stage}")

def register_existing_model(weight_path, stage=None):
    model = YOLO(weight_path)

    with mlflow.start_run() as run:
        mlflow.log_param("source", weight_path)
        mlflow.pytorch.log_model(
            pytorch_model=model.model,
            artifact_path="model",
            registered_model_name="DefectDetectionModel"
        )

        print(f"Model registered from {weight_path} (run_id={run.info.run_id})")

        if stage:
            client = mlflow.tracking.MlflowClient()
            latest_versions = client.get_latest_versions("DefectDetectionModel")
            for v in latest_versions:
                if v.run_id == run.info.run_id:
                    client.transition_model_version_stage(
                        name="DefectDetectionModel",
                        version=v.version,
                        stage=stage
                    )
                    print(f"Model moved to stage: {stage}")


if __name__ == "__main__":
    # Modello dimostrativo
    register_model("runs/train/exp3/weights/best.pt", "v1", stage="Staging")

    # Modello finale (30 epoche)
    register_existing_model(r"C:\Users\web\Desktop\projectTesi\app\best_ingranaggioYOLO.pt", stage="Production")
