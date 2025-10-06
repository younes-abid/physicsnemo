import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from typing import List, Tuple, Dict

class CircleDataset(Dataset):
    """Dataset for circle detection training."""
    
    def __init__(self, images: List[np.ndarray], annotations: List[Dict], transform=None):
        self.images = images
        self.annotations = annotations
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        annotation = self.annotations[idx]
        
        # Convert image to tensor
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)  # Add channel dimension
        else:
            image = image.transpose(2, 0, 1)  # HWC to CHW
        
        image_tensor = torch.FloatTensor(image) / 255.0
        
        # Create target: heatmap of circle centers
        target = self._create_target_heatmap(image.shape[1:], annotation['circles'])
        
        return image_tensor, target
    
    def _create_target_heatmap(self, image_shape: Tuple[int, int], circles: List[Dict]) -> torch.Tensor:
        """Create Gaussian heatmap for circle centers."""
        heatmap = torch.zeros(image_shape)
        
        for circle in circles:
            center = circle['center']
            radius = circle['radius']
            
            # Create Gaussian blob at center
            y, x = np.ogrid[:image_shape[0], :image_shape[1]]
            y_center, x_center = center[1], center[0]
            
            # Gaussian sigma proportional to radius
            sigma = max(radius * 0.3, 2.0)
            
            gaussian = np.exp(-((x - x_center)**2 + (y - y_center)**2) / (2 * sigma**2))
            heatmap = torch.max(heatmap, torch.FloatTensor(gaussian))
        
        return heatmap

class CircleDetectionModel(nn.Module):
    """CNN for circle detection."""
    
    def __init__(self, input_channels=1):
        super(CircleDetectionModel, self).__init__()
        
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()  # Output heatmap between 0-1
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AICircleDetector:
    """AI-based circle detector."""
    
    def __init__(self, model_path: str = None):
        self.model = CircleDetectionModel()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
    
    def train(self, train_images: List[np.ndarray], train_annotations: List[Dict],
              val_images: List[np.ndarray] = None, val_annotations: List[Dict] = None,
              epochs: int = 50, batch_size: int = 4):
        """Train the circle detection model."""
        
        train_dataset = CircleDataset(train_images, train_annotations)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_images and val_annotations:
            val_dataset = CircleDataset(val_images, val_annotations)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target.unsqueeze(1))
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            if val_images and val_annotations:
                val_loss = self._validate(val_loader, criterion)
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, '
                      f'Val Loss: {val_loss:.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}')
    
    def _validate(self, val_loader, criterion):
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += criterion(output, target.unsqueeze(1)).item()
        
        return val_loss / len(val_loader)
    
    def detect_circles(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """Detect circles using the trained model."""
        self.model.eval()
        
        # Preprocess image
        if len(image.shape) == 2:
            image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0) / 255.0
        else:
            image_tensor = torch.FloatTensor(image.transpose(2, 0, 1)).unsqueeze(0) / 255.0
        
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            heatmap = self.model(image_tensor)
            heatmap = heatmap.squeeze().cpu().numpy()
        
        # Find local maxima in heatmap
        circles = self._find_circles_from_heatmap(heatmap, confidence_threshold)
        
        return circles
    
    def _find_circles_from_heatmap(self, heatmap: np.ndarray, threshold: float) -> List[Dict]:
        """Extract circles from heatmap using connected components."""
        from skimage.feature import peak_local_max
        
        # Find local maxima
        coordinates = peak_local_max(heatmap, min_distance=10, threshold_abs=threshold)
        
        circles = []
        for coord in coordinates:
            y, x = coord
            confidence = heatmap[y, x]
            
            # Estimate radius (you might need to predict this separately)
            # For now, use a fixed ratio or implement radius prediction
            radius = self._estimate_radius(heatmap, (x, y))
            
            circles.append({
                'center': (int(x), int(y)),
                'radius': int(radius),
                'confidence': confidence,
                'method': 'ai'
            })
        
        return circles
    
    def _estimate_radius(self, heatmap: np.ndarray, center: Tuple[int, int]) -> int:
        """Estimate circle radius from heatmap (simplified)."""
        # This is a simplified approach - you might want to predict radius directly
        x, y = center
        height, width = heatmap.shape
        
        # Sample distances from center and find where heatmap drops
        max_radius = min(x, y, width - x, height - y)
        
        for r in range(5, max_radius, 2):
            # Check if heatmap value drops significantly at this radius
            if r + 5 < max_radius:
                inner_val = heatmap[y, x + r]
                outer_val = heatmap[y, x + r + 5]
                if outer_val < inner_val * 0.5:  # Significant drop
                    return r
        
        return min(20, max_radius)  # Default radius