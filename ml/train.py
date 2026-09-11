import mlflow
import mlflow.pyfunc
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("sentiment-analysis")

class SentimentModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.clf = pipeline("sentiment-analysis")

    def predict(self, context, model_input):
        texts = model_input["text"].tolist()
        results = self.clf(texts, truncation=True, max_length=512)
        return [r["label"] for r in results]

def evaluate(df, model):
    preds = model.predict(None, df)
    y_true = df["label"].map({0: "NEGATIVE", 1: "POSITIVE"})
    return {
        "accuracy": accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds, pos_label="POSITIVE"),
    }

if __name__ == "__main__":
    test_df = pd.read_csv("ml/data/raw/imdb_test.csv").sample(200, random_state=42)

    with mlflow.start_run():
        model = SentimentModel()
        model.load_context(None)

        metrics = evaluate(test_df, model)
        mlflow.log_metrics(metrics)
        mlflow.log_param("model_source", "distilbert-base-uncased-finetuned-sst-2-english")
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model,
            registered_model_name="sentiment-classifier",
        )
        print(metrics)
