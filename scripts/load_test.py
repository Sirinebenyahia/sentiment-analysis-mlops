from locust import HttpUser, task, between


class SentimentUser(HttpUser):
    wait_time = between(0.1, 0.3)

    @task
    def predict(self):
        self.client.post("/predict", json={"text": "This is a great day for load testing!"})
