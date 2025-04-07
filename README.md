# Intrusion Detection Inference Module

This Python module allows you to run inference using a trained Spatiotemporal CNN model that combines traffic data, system metrics, and Kubernetes log embeddings into RGB image representations.

---

## How to Use

### Install dependencies

pip install -r requirements.txt

### Initialize the intrusion detector with path to artifacts
detector = IntrusionDetector('/home/sascha/kubernetes-intrusion-detection-main/KubeFocus/artifacts')

### Example inputs (replace with live data)
traffic_input = [0.3, 0.7, 0.1, 0.0]

metrics_input = 377 features  # You must use real metrics data here

log_text = "Error: failed to start container..."

pod_name = "node_k8s-worker-1"

### Predict
pred = detector.predict(traffic_input, metrics_input, log_text, pod_name)








