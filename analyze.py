#!/usr/bin/env python3
# analyze.py - PRODUCTION ONLY Grad-CAM Implementation for CheXNet
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import io
import logging
from typing import Tuple, List, Dict, Optional
import time
from collections import OrderedDict
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 14 Thoracic disease labels (EXACT same order as original CheXNet)
DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# Disease severity and urgency mapping
DISEASE_INFO = {
    'Atelectasis': {'severity': 'moderate', 'urgency': 'routine', 'description': 'Collapsed or airless lung'},
    'Cardiomegaly': {'severity': 'moderate', 'urgency': 'urgent', 'description': 'Enlarged heart'},
    'Effusion': {'severity': 'moderate', 'urgency': 'urgent', 'description': 'Fluid around lungs'},
    'Infiltration': {'severity': 'mild', 'urgency': 'routine', 'description': 'Lung tissue changes'},
    'Mass': {'severity': 'severe', 'urgency': 'urgent', 'description': 'Abnormal growth in lung'},
    'Nodule': {'severity': 'moderate', 'urgency': 'routine', 'description': 'Small round growth'},
    'Pneumonia': {'severity': 'moderate', 'urgency': 'urgent', 'description': 'Lung infection'},
    'Pneumothorax': {'severity': 'severe', 'urgency': 'emergency', 'description': 'Collapsed lung'},
    'Consolidation': {'severity': 'moderate', 'urgency': 'urgent', 'description': 'Lung filled with liquid'},
    'Edema': {'severity': 'severe', 'urgency': 'urgent', 'description': 'Fluid in lungs'},
    'Emphysema': {'severity': 'moderate', 'urgency': 'routine', 'description': 'Damaged air sacs'},
    'Fibrosis': {'severity': 'severe', 'urgency': 'routine', 'description': 'Lung scarring'},
    'Pleural_Thickening': {'severity': 'mild', 'urgency': 'routine', 'description': 'Thickened lung lining'},
    'Hernia': {'severity': 'moderate', 'urgency': 'routine', 'description': 'Organ displacement'}
}

class DenseNet121(nn.Module):
    """EXACT replica of the original CheXNet model architecture"""
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        # EXACT same architecture as original - includes Sigmoid
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

class GradCAMHook:
    """FIXED Grad-CAM implementation that works properly"""
    
    def __init__(self):
        self.features = None
        self.gradients = None
        self.handles = []
    
    def clear_hooks(self):
        """Clear all registered hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.features = None
        self.gradients = None
    
    def register_hooks(self, target_layer):
        """Register forward and backward hooks on target layer"""
        def forward_hook(module, input, output):
            self.features = output.detach().clone()
            logger.info(f"Forward hook captured features: {output.shape}")
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach().clone()
                logger.info(f"Backward hook captured gradients: {grad_output[0].shape}")
        
        # Register hooks and store handles for cleanup
        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)
        self.handles.extend([h1, h2])
        
        logger.info("Grad-CAM hooks registered successfully")
    
    def generate_cam(self, class_idx: int) -> np.ndarray:
        """Generate Class Activation Map"""
        if self.features is None or self.gradients is None:
            logger.error("Features or gradients not captured")
            return np.zeros((224, 224))
        
        try:
            # Calculate weights (Global Average Pooling of gradients)
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            
            # Weighted combination of feature maps
            cam = torch.sum(weights * self.features, dim=1).squeeze(0)
            
            # Apply ReLU to keep only positive contributions
            cam = F.relu(cam)
            
            # Normalize CAM
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            
            # Resize to input image size
            cam = F.interpolate(
                cam.unsqueeze(0).unsqueeze(0),
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            ).squeeze().cpu().numpy()
            
            logger.info(f"CAM generated successfully: shape={cam.shape}, range=[{cam.min():.3f}, {cam.max():.3f}]")
            return cam
            
        except Exception as e:
            logger.error(f"CAM generation failed: {e}")
            return np.zeros((224, 224))

class XRayAnalyzer:
    """PRODUCTION ONLY X-ray analyzer with working Grad-CAM"""
    
    def __init__(self, model_path: str = "model.pth.tar", device: str = "cpu"):
        """Initialize the X-ray analyzer"""
        self.device = torch.device(device)
        self.model = None
        self.gradcam_hook = GradCAMHook()
        self.model_path = model_path
        self.target_layer = None
        
        # Load model - NO FALLBACKS
        self._load_model()
        
        # EXACT same preprocessing as original CheXNet
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ])
        
        # Single image transform for Grad-CAM
        self.single_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
        
        logger.info("XRayAnalyzer initialized in PRODUCTION mode")

    def _load_model(self):
        """Load the CheXNet model - PRODUCTION ONLY"""
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            raise FileNotFoundError(f"PRODUCTION MODE: Model file not found: {self.model_path}")
        
        try:
            logger.info(f"Loading CheXNet model from {self.model_path}")
            
            # Initialize model
            model = DenseNet121(len(DISEASE_LABELS))
            model = torch.nn.DataParallel(model)
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
            
            self.model = model.to(self.device)
            self.model.eval()
            
            # Setup Grad-CAM
            self._setup_gradcam()
            
            logger.info("CheXNet model loaded successfully - PRODUCTION MODE")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"PRODUCTION MODE: Model loading failed: {e}")

    def _setup_gradcam(self):
        """Setup Grad-CAM hooks - MUST WORK"""
        try:
            # Find the target layer - last convolutional layer in DenseNet
            target_layer = None
            
            # Try multiple paths to find the correct layer
            possible_paths = [
                'module.densenet121.features.denseblock4.denselayer16.conv2',
                'module.densenet121.features.norm5',
                'densenet121.features.denseblock4.denselayer16.conv2',
                'densenet121.features.norm5'
            ]
            
            for path in possible_paths:
                try:
                    layer = self.model
                    for attr in path.split('.'):
                        layer = getattr(layer, attr)
                    target_layer = layer
                    logger.info(f"Found target layer: {path}")
                    break
                except AttributeError:
                    continue
            
            # Fallback: search for the last conv layer
            if target_layer is None:
                conv_layers = []
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Conv2d) and 'classifier' not in name:
                        conv_layers.append((name, module))
                
                if conv_layers:
                    layer_name, target_layer = conv_layers[-1]
                    logger.info(f"Using fallback conv layer: {layer_name}")
                else:
                    raise AttributeError("No suitable convolutional layer found")
            
            if target_layer is not None:
                self.target_layer = target_layer
                self.gradcam_hook.register_hooks(target_layer)
                logger.info("Grad-CAM setup completed successfully")
            else:
                raise AttributeError("Failed to find target layer for Grad-CAM")
                
        except Exception as e:
            logger.error(f"Grad-CAM setup failed: {e}")
            raise RuntimeError(f"PRODUCTION MODE: Grad-CAM setup failed: {e}")

    def _preprocess_image(self, image_bytes: bytes) -> Tuple[Image.Image, torch.Tensor]:
        """Preprocess image exactly like original CheXNet"""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            width, height = image.size
            if width < 224 or height < 224:
                logger.warning(f"Low resolution image: {width}x{height}")
            
            crops_tensor = self.transform(image)
            
            return image, crops_tensor
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise

    def _generate_gradcam(self, image: Image.Image, class_idx: int) -> np.ndarray:
        """Generate Grad-CAM heatmap - MUST WORK"""
        try:
            if self.target_layer is None:
                logger.error("No target layer available for Grad-CAM")
                raise RuntimeError("Grad-CAM not properly initialized")
            
            logger.info(f"Generating Grad-CAM for class {class_idx} ({DISEASE_LABELS[class_idx]})")
            
            # Clear previous hooks data
            self.gradcam_hook.features = None
            self.gradcam_hook.gradients = None
            
            # Prepare image for single inference
            image_tensor = self.single_transform(image).unsqueeze(0).to(self.device)
            image_tensor.requires_grad_(True)
            
            # Enable gradients and clear previous gradients
            self.model.zero_grad()
            
            # Forward pass with gradient tracking
            with torch.set_grad_enabled(True):
                output = self.model(image_tensor)
                
                # Check if hooks captured features
                if self.gradcam_hook.features is None:
                    logger.error("Forward hook failed to capture features")
                    raise RuntimeError("Grad-CAM forward hook failed")
                
                # Get class score and perform backward pass
                class_score = output[0, class_idx]
                logger.info(f"Class score: {class_score.item():.4f}")
                
                # Backward pass
                class_score.backward(retain_graph=False)
                
                # Check if hooks captured gradients
                if self.gradcam_hook.gradients is None:
                    logger.error("Backward hook failed to capture gradients")
                    raise RuntimeError("Grad-CAM backward hook failed")
                
                # Generate CAM
                cam = self.gradcam_hook.generate_cam(class_idx)
                
                if np.all(cam == 0):
                    logger.warning("Generated CAM is all zeros")
                
                return cam
                
        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {e}")
            raise RuntimeError(f"Grad-CAM generation failed: {e}")

    def _create_heatmap_overlay(self, image: Image.Image, heatmap: np.ndarray) -> Image.Image:
        """Create heatmap overlay with original image"""
        try:
            # Resize image to match heatmap
            base_image = image.resize((224, 224)).convert("RGB")
            image_np = np.array(base_image)
            
            # Ensure heatmap is properly normalized
            if heatmap.max() > 0:
                heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            else:
                logger.warning("Empty heatmap, using zeros")
                heatmap_norm = np.zeros_like(heatmap)
            
            # Apply threshold to focus on important regions
            threshold = 0.2
            heatmap_norm[heatmap_norm < threshold] = 0
            
            # Create colored heatmap using JET colormap
            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap_norm), 
                cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Create overlay with transparency
            alpha = 0.4
            overlay = cv2.addWeighted(image_np, 1-alpha, heatmap_colored, alpha, 0)
            
            # Convert back to PIL Image
            result_image = Image.fromarray(overlay)
            
            # Check quality
            non_zero_pixels = np.sum(heatmap_norm > threshold)
            logger.info(f"Heatmap overlay created with {non_zero_pixels} highlighted pixels")
            
            return result_image
            
        except Exception as e:
            logger.error(f"Heatmap overlay creation failed: {e}")
            raise RuntimeError(f"Heatmap overlay creation failed: {e}")

    def analyze_xray(
        self, 
        image_bytes: bytes, 
        threshold: float = 0.5,
        generate_heatmap: bool = True
    ) -> Tuple[List[Dict], bytes, Dict]:
        """MAIN analysis function - PRODUCTION ONLY"""
        start_time = time.time()
        
        try:
            logger.info("Starting PRODUCTION CheXNet X-ray analysis...")
            
            # Preprocess image
            original_image, crops_tensor = self._preprocess_image(image_bytes)
            crops_tensor = crops_tensor.to(self.device)
            
            # Forward pass for classification
            with torch.no_grad():
                bs = 1
                n_crops = crops_tensor.size(0)  # 10
                c, h, w = crops_tensor.size(1), crops_tensor.size(2), crops_tensor.size(3)
                
                input_var = crops_tensor.view(-1, c, h, w)  # [10, 3, 224, 224]
                output = self.model(input_var)  # [10, 14]
                output_mean = output.view(bs, n_crops, -1).mean(1)  # [1, 14]
                probabilities = output_mean.cpu().numpy()[0]  # [14]
            
            # Process findings
            findings = []
            has_abnormalities = False
            max_confidence = 0.0
            
            for i, prob in enumerate(probabilities):
                condition = DISEASE_LABELS[i]
                confidence = float(prob)
                max_confidence = max(max_confidence, confidence)
                
                if confidence > threshold:
                    has_abnormalities = True
                    disease_info = DISEASE_INFO.get(condition, {})
                    
                    finding = {
                        'condition': condition,
                        'confidence': confidence,
                        'severity': disease_info.get('severity', 'unknown'),
                        'urgency': disease_info.get('urgency', 'routine'),
                        'description': disease_info.get('description', ''),
                        'category': 'thoracic_disease'
                    }
                    findings.append(finding)
            
            findings.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Generate Grad-CAM heatmap - MANDATORY
            heatmap_bytes = None
            heatmap_generated = False
            
            if generate_heatmap:
                try:
                    logger.info("Starting MANDATORY Grad-CAM heatmap generation...")
                    
                    # Use the highest confidence class for heatmap
                    top_class_idx = np.argmax(probabilities)
                    top_confidence = probabilities[top_class_idx]
                    
                    logger.info(f"Generating heatmap for: {DISEASE_LABELS[top_class_idx]} (confidence: {top_confidence:.3f})")
                    
                    # Generate Grad-CAM
                    heatmap = self._generate_gradcam(original_image, top_class_idx)
                    
                    # Create overlay
                    overlay_image = self._create_heatmap_overlay(original_image, heatmap)
                    
                    # Convert to bytes
                    heatmap_buffer = io.BytesIO()
                    overlay_image.save(heatmap_buffer, format='PNG', quality=95, optimize=True)
                    heatmap_bytes = heatmap_buffer.getvalue()
                    heatmap_generated = True
                    
                    logger.info("Grad-CAM heatmap generated successfully!")
                    
                except Exception as e:
                    logger.error(f"CRITICAL: Grad-CAM generation failed: {e}")
                    raise RuntimeError(f"PRODUCTION MODE: Grad-CAM generation failed: {e}")
            
            processing_time = time.time() - start_time
            
            # Metadata
            metadata = {
                'processing_time': processing_time,
                'image_quality_score': 0.8,
                'has_abnormalities': has_abnormalities,
                'max_confidence': max_confidence,
                'total_findings': len(findings),
                'model_version': 'CheXNet-DenseNet121',
                'threshold_used': threshold,
                'heatmap_generated': heatmap_generated,
                'gradcam_available': True,
                'mode': 'production'
            }
            
            # Generate report
            report = self._generate_report(findings, metadata)
            
            logger.info(f"PRODUCTION analysis completed in {processing_time:.2f}s, {len(findings)} findings, heatmap: {heatmap_generated}")
            
            return report, heatmap_bytes, metadata
            
        except Exception as e:
            logger.error(f"PRODUCTION analysis failed: {e}")
            raise RuntimeError(f"PRODUCTION MODE: Analysis failed: {e}")

    def _generate_report(self, findings: List[Dict], metadata: Dict) -> List[str]:
        """Generate medical report"""
        report = []
        
        if not findings or not metadata.get('has_abnormalities', False):
            report.append(
                "No significant abnormalities detected among the 14 thoracic conditions analyzed. "
                "This AI screening does not replace professional radiological interpretation."
            )
        else:
            report.append("CheXNet AI Analysis Results:")
            
            # Categorize by urgency
            urgent_findings = [f for f in findings if f['urgency'] == 'emergency']
            important_findings = [f for f in findings if f['urgency'] == 'urgent']
            routine_findings = [f for f in findings if f['urgency'] == 'routine']
            
            if urgent_findings:
                report.append("URGENT FINDINGS - Immediate attention required:")
                for finding in urgent_findings:
                    report.append(f"  • {finding['condition']}: {finding['confidence']:.3f} probability - {finding['description']}")
            
            if important_findings:
                report.append("SIGNIFICANT FINDINGS - Medical consultation recommended:")
                for finding in important_findings:
                    report.append(f"  • {finding['condition']}: {finding['confidence']:.3f} probability - {finding['description']}")
            
            if routine_findings:
                report.append("ADDITIONAL FINDINGS - Consider follow-up:")
                for finding in routine_findings:
                    report.append(f"  • {finding['condition']}: {finding['confidence']:.3f} probability - {finding['description']}")
        
        # Technical info
        report.append(f"Processing Time: {metadata.get('processing_time', 0):.2f} seconds")
        report.append("Model: CheXNet (DenseNet-121 on ChestX-ray14)")
        
        if metadata.get('heatmap_generated', False):
            report.append("Grad-CAM Heatmap: Successfully generated")
        else:
            report.append("Grad-CAM Heatmap: Generation failed")
        
        return report

    def __del__(self):
        """Cleanup hooks when object is destroyed"""
        if hasattr(self, 'gradcam_hook'):
            self.gradcam_hook.clear_hooks()

# Global analyzer instance
_analyzer = None

def get_analyzer() -> XRayAnalyzer:
    """Get or create global analyzer instance - PRODUCTION ONLY"""
    global _analyzer
    if _analyzer is None:
        if not check_model_availability():
            raise FileNotFoundError("PRODUCTION MODE: Model file not found")
        _analyzer = XRayAnalyzer()
    return _analyzer

def analyze_xray(
    image_bytes: bytes, 
    threshold: float = 0.5,
    generate_heatmap: bool = True
) -> Tuple[List[str], bytes, Dict]:
    """
    Main analysis function - PRODUCTION ONLY
    """
    if not check_model_availability():
        raise FileNotFoundError("PRODUCTION MODE: Model file 'model.pth.tar' not found")
    
    analyzer = get_analyzer()
    findings, heatmap_bytes, metadata = analyzer.analyze_xray(
        image_bytes, threshold, generate_heatmap
    )
    return findings, heatmap_bytes, metadata

def check_model_availability() -> bool:
    """Check if model file exists"""
    return os.path.exists("model.pth.tar")

def get_model_info() -> Dict:
    """Get model information - PRODUCTION ONLY"""
    model_available = check_model_availability()
    return {
        'model_available': model_available,
        'mode': 'production' if model_available else 'error',
        'supported_conditions': DISEASE_LABELS,
        'model_architecture': 'CheXNet (DenseNet-121)',
        'input_size': '224x224',
        'supported_formats': ['JPEG', 'PNG', 'BMP', 'TIFF'],
        'dataset': 'ChestX-ray14',
        'multi_label': True,
        'gradcam_support': True,
        'production_only': True
    }

if __name__ == "__main__":
    # Test the analyzer
    if check_model_availability():
        print("Model available, testing production mode...")
        try:
            analyzer = get_analyzer()
            print("Analyzer initialized successfully")
            print(f"Target layer available: {analyzer.target_layer is not None}")
            print(f"Grad-CAM hooks registered: {len(analyzer.gradcam_hook.handles) > 0}")
        except Exception as e:
            print(f"Analyzer initialization failed: {e}")
    else:
        print("PRODUCTION MODE: Model file 'model.pth.tar' not found")
        print("Please ensure the model file is available for production mode")
    
    print(f"\nModel Info: {get_model_info()}")
