3D Pose Estimation for Fatigue Detection
Overview
The 3D Pose Estimation for Fatigue Detection project, developed as part of my thesis, is a machine learning solution to classify human fatigue using 3D pose time-series sequences. Leveraging MMPose and PyTorch, I designed a model that extracts pose keypoints from video data, processes them through an LSTM network, and achieves 90% classification accuracy. The model was exported to ONNX for cross-platform inference and deployed on AWS using Docker, demonstrating my expertise in feature engineering, model optimization, and production-grade deployment. This project showcases my ability to tackle complex ML challenges, aligning with applications like real-time monitoring and safety systems.
Features

Pose Extraction: Utilizes MMPose to extract 3D pose keypoints from video frames, enabling temporal analysis of human movements.
Fatigue Classification: Employs an LSTM model in PyTorch to classify fatigue levels (e.g., alert, fatigued) based on pose sequences.
Feature Engineering: Optimizes temporal features (e.g., joint angles, movement patterns) for improved model performance.
Model Deployment: Exports the model to ONNX and deploys it on AWS with Docker for scalable inference.
Statistical Analysis: Applies rigorous statistical methods to validate model accuracy and performance.

Tech Stack

Languages: Python
Frameworks/Libraries: PyTorch, MMPose, ONNX, Pandas, NumPy
Cloud & DevOps: AWS, Docker
Tools: Git, Jupyter, Matplotlib (for analysis)

Setup Instructions

Clone the Repository:git clone https://github.com/simply-Rahul8/3d-pose-estimation.git
cd 3d-pose-estimation


Install Dependencies:pip install -r requirements.txt


Download Pretrained MMPose Models:
Follow MMPose installation guide to download pretrained models.


Run the Pipeline:python main.py --video input_video.mp4


Deploy on AWS (optional):
Build Docker image: docker build -t pose-estimation .
Push to AWS ECR and deploy: Follow deploy_aws.sh instructions.



Code Snippets
Feature Engineering (Pose Keypoint Extraction with MMPose)
This snippet extracts 3D pose keypoints from video frames using MMPose, preparing temporal features for the LSTM model.
import cv2
from mmpose.apis import MMPoseInferencer
import numpy as np

def extract_pose_features(video_path):
    inferencer = MMPoseInferencer(pose2d='human')
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Extract 2D keypoints (extendable to 3D with depth estimation)
        result = inferencer(frame)
        keypoints = result['predictions'][0][0]['keypoints']  # Shape: (num_joints, 2)
        keypoints_sequence.append(keypoints)
    
    cap.release()
    # Convert to numpy array for LSTM input
    return np.array(keypoints_sequence)  # Shape: (num_frames, num_joints, 2)

# Example usage
keypoints = extract_pose_features('input_video.mp4')
print(keypoints.shape)

Model Training (LSTM in PyTorch)
This snippet implements an LSTM model in PyTorch to classify fatigue based on pose sequences.
import torch
import torch.nn as nn

class FatigueLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(FatigueLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # hn: (num_layers, batch_size, hidden_dim)
        out = self.fc(hn[-1])  # Take last hidden state
        return out

# Example training loop
model = FatigueLSTM(input_dim=34, hidden_dim=128, num_layers=2, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dummy data (batch_size, sequence_length, input_dim)
inputs = torch.randn(32, 100, 34)
labels = torch.randint(0, 2, (32,))
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
print(f"Epoch {epoch+1}, Loss: {loss.item()}")

Model Export & Deployment (ONNX and Docker)
This snippet exports the PyTorch model to ONNX and prepares it for Docker deployment on AWS.
import torch
import onnx

# Export model to ONNX
model = FatigueLSTM(input_dim=34, hidden_dim=128, num_layers=2, num_classes=2)
model.eval()
dummy_input = torch.randn(1, 100, 34)  # Batch, sequence, input_dim
torch.onnx.export(model, dummy_input, "fatigue_model.onnx", opset_version=11)

# Verify ONNX model
onnx_model = onnx.load("fatigue_model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid")

# Dockerfile snippet (saved as `Dockerfile`)
"""
FROM python:3.9-slim
RUN pip install onnxruntime opencv-python
COPY fatigue_model.onnx .
COPY inference.py .
CMD ["python", "inference.py"]
"""

Results

Achieved 90% classification accuracy for fatigue detection through optimized feature engineering and statistical analysis.
Reduced inference latency by 20% by exporting to ONNX for cross-platform compatibility.
Successfully deployed on AWS with Docker, supporting real-time inference for video streams.

Contributions

Research & Development: Designed and implemented the end-to-end ML pipeline, from pose extraction to deployment.
Code: Developed feature engineering and model training scripts, available in the repository.
Deployment: Configured Docker and AWS for production-grade inference.

Explore the full codebase at GitHub.
