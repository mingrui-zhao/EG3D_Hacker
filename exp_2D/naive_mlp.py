import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load an image using PIL and convert to numpy array
img_path = "./exp_2D/data/mong_tea.png"  # Replace with your image path
img = Image.open(img_path).convert('RGB')
img = np.array(img)
height, width, _ = img.shape

# Normalize image values and coordinates to [0, 1]
img_normalized = img / 255.0
x_coords = np.linspace(0, 1, width)
y_coords = np.linspace(0, 1, height)

# Create training data
x, y = np.meshgrid(x_coords, y_coords)
coords = np.stack([x.flatten(), y.flatten()], axis=-1)
pixels = img_normalized.reshape(-1, 3)

# Convert to PyTorch tensors
coords_tensor = torch.tensor(coords, dtype=torch.float32)
pixels_tensor = torch.tensor(pixels, dtype=torch.float32)

# Define MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.layers(x)

# Initialize model, loss, and optimizer
model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 5000
for epoch in range(num_epochs):
    output = model(coords_tensor)
    loss = criterion(output, pixels_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Generate image from the trained model
with torch.no_grad():
    predicted = model(coords_tensor).numpy()
predicted = np.clip(predicted, 0, 1)
predicted_img = (predicted * 255).astype(np.uint8).reshape(height, width, 3)

# Plot original and predicted images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Predicted Image")
plt.imshow(predicted_img)
plt.axis('off')

plt.show()
