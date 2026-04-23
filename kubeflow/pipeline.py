from kfp import dsl


@dsl.component(base_image="python:3.11-slim")
def data_ingestion() -> str:
    return "Data ingestion completed"


@dsl.component(base_image="python:3.11-slim")
def data_validation(ingestion_status: str) -> str:
    return f"{ingestion_status} -> validation completed"


@dsl.component(base_image="python:3.11-slim")
def data_preprocessing(validation_status: str) -> str:
    return f"{validation_status} -> preprocessing completed"


@dsl.component(base_image="python:3.11-slim")
def feature_engineering(preprocessing_status: str) -> str:
    return f"{preprocessing_status} -> feature engineering completed"


@dsl.component(base_image="python:3.11-slim")
def model_training(feature_status: str) -> str:
    return f"{feature_status} -> model training completed"


@dsl.component(base_image="python:3.11-slim")
def model_evaluation(training_status: str) -> float:
    _ = training_status
    return 0.93


@dsl.component(base_image="python:3.11-slim")
def conditional_deployment(auc_score: float) -> str:
    return f"Deployment approved with AUC={auc_score}"


@dsl.pipeline(
    name="fraud-detection-pipeline",
    description="Kubeflow pipeline for fraud detection lifecycle",
)
def fraud_detection_pipeline(deploy_auc_threshold: float = 0.92):
    ingestion = data_ingestion().set_retry(2)
    validation = data_validation(ingestion_status=ingestion.output).set_retry(2)
    preprocessing = data_preprocessing(validation_status=validation.output).set_retry(2)
    features = feature_engineering(preprocessing_status=preprocessing.output).set_retry(2)
    training = model_training(feature_status=features.output).set_retry(1)
    evaluation = model_evaluation(training_status=training.output).set_retry(1)

    with dsl.If(evaluation.output >= deploy_auc_threshold):
        conditional_deployment(auc_score=evaluation.output).set_retry(1)


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=fraud_detection_pipeline,
        package_path="artifacts/fraud_detection_pipeline.yaml",
    )
