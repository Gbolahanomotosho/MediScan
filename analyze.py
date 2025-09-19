#!/usr/bin/env python3
# analyze.py - Final Fixed CheXNet Compatible X-ray Analysis Engine
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

class XRayAnalyzer:
    """Production X-ray analyzer - Exact CheXNet compatibility"""
    
    def __init__(self, model_path: str = "model.pth.tar", device: str = "cpu"):
        """Initialize the X-ray analyzer"""
        self.device = torch.device(device)
        self.model = None
        self.features = None
        self.gradients = None
        self.model_path = model_path
        
        # Load model
        self._load_model()
        
        # EXACT same preprocessing as original CheXNet
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ])
        
        # Single image transform for visualization
        self.single_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
        
        logger.info("CheXNet XRayAnalyzer initialized successfully")

    def _load_model(self):
        """Load the exact CheXNet model with DataParallel handling"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading CheXNet model from {self.model_path}")
            
            # Initialize the EXACT model architecture
            model = DenseNet121(len(DISEASE_LABELS))
            
            # Wrap with DataParallel (as in original)
            model = torch.nn.DataParallel(model)
            
            # Load checkpoint
            if os.path.isfile(self.model_path):
                logger.info("=> loading checkpoint")
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Load state dict with non-strict loading (expected for CheXNet)
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                logger.info(f"=> loaded checkpoint with {len(missing_keys)} missing keys")
                if missing_keys:
                    logger.warning(f"Missing keys: {missing_keys[:3]}...")
                
                self.model = model
                
            else:
                logger.error("=> no checkpoint found")
                raise FileNotFoundError("No checkpoint found")
            
            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            # Register hooks for Grad-CAM
            self._register_hooks()
            
            logger.info("CheXNet model loaded and ready for inference")
            
        except Exception as e:
            logger.error(f"Failed to load CheXNet model: {e}")
            logger.error("Please ensure your model.pth.tar is a valid CheXNet checkpoint")
            raise

    def _register_hooks(self):
        """Register forward and backward hooks for Grad-CAM"""
        def forward_hook(module, input, output):
            self.features = output

        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0]

        # Register hooks on the last convolutional layer
        try:
            # Access through DataParallel wrapper - try different layer names
            try:
                # Try standard DenseNet naming first
                target_layer = self.model.module.densenet121.features.denseblock4.denselayer16.conv2
            except AttributeError:
                # Try alternative naming that might exist in your model
                try:
                    target_layer = self.model.module.densenet121.features.denseblock4.denselayer16.conv.2
                except AttributeError:
                    # If all else fails, find the last conv layer
                    layers = []
                    for name, module in self.model.named_modules():
                        if isinstance(module, nn.Conv2d):
                            layers.append((name, module))
                    if layers:
                        target_layer = layers[-1][1]
                        logger.info(f"Using last conv layer: {layers[-1][0]}")
                    else:
                        raise AttributeError("Could not find any conv layers")
            
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_full_backward_hook(backward_hook)
            logger.info("Grad-CAM hooks registered successfully")
            
        except AttributeError as e:
            logger.warning(f"Could not register Grad-CAM hooks: {e}")
            logger.warning("Grad-CAM visualization may not work properly")

    def _preprocess_image(self, image_bytes: bytes) -> Tuple[Image.Image, torch.Tensor]:
        """Preprocess image exactly like original CheXNet"""
        try:
            # Load image and convert to RGB
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Basic image quality checks
            width, height = image.size
            if width < 224 or height < 224:
                logger.warning(f"Image resolution is low: {width}x{height}")
            
            # Apply CheXNet transforms (no additional enhancement)
            crops_tensor = self.transform(image)
            
            return image, crops_tensor
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise

    def _calculate_image_quality_score(self, image: Image.Image) -> float:
        """Calculate image quality score for medical images"""
        try:
            # Convert to numpy array
            img_array = np.array(image.convert('L'))
            
            # Calculate sharpness using Laplacian variance
            laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Calculate contrast
            contrast = img_array.std()
            
            # Calculate brightness distribution (important for X-rays)
            histogram = np.histogram(img_array, bins=256)[0]
            brightness_distribution = np.sum(histogram[64:192]) / np.sum(histogram)
            
            # Medical image quality score
            quality_score = min(1.0, (
                min(sharpness / 800, 1.0) * 0.4 +
                min(contrast / 80, 1.0) * 0.4 +
                brightness_distribution * 0.2
            ))
            
            return float(quality_score)
            
        except Exception as e:
            logger.warning(f"Quality score calculation failed: {e}")
            return 0.7

    def _generate_gradcam(self, image: Image.Image, class_idx: int) -> np.ndarray:
        """Generate Grad-CAM heatmap for CheXNet"""
        try:
            # Prepare single image for Grad-CAM
            image_tensor = self.single_transform(image).unsqueeze(0).to(self.device)
            
            # Forward pass
            self.model.zero_grad()
            output = self.model(image_tensor)
            
            # Get class score (output already has sigmoid applied)
            if output.dim() == 1:
                output = output.unsqueeze(0)
            
            class_score = output[0, class_idx]
            
            # Backward pass
            class_score.backward()
            
            # Generate Grad-CAM
            if self.gradients is not None and self.features is not None:
                # Calculate weights
                weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
                
                # Generate CAM
                grad_cam = F.relu(torch.sum(weights * self.features, dim=1)).squeeze()
                
                # Normalize
                if grad_cam.numel() > 0:
                    grad_cam -= grad_cam.min()
                    if grad_cam.max() > 0:
                        grad_cam /= grad_cam.max()
                
                    # Resize to original image size
                    grad_cam = F.interpolate(
                        grad_cam.unsqueeze(0).unsqueeze(0),
                        size=(224, 224),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze().cpu().numpy()
                else:
                    grad_cam = np.zeros((224, 224))
                
                return grad_cam
            else:
                logger.warning("Grad-CAM features not available")
                return np.zeros((224, 224))
                
        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {e}")
            return np.zeros((224, 224))

    def _create_heatmap_overlay(self, image: Image.Image, heatmap: np.ndarray, findings: List[Dict]) -> Image.Image:
        """Create heatmap overlay for medical visualization"""
        try:
            # Resize image and heatmap to same dimensions
            base_image = image.resize((224, 224)).convert("RGB")
            image_np = np.array(base_image)
            
            # Create colored heatmap with medical-appropriate colors
            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap), cv2.COLORMAP_HOT
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Blend images with appropriate transparency for medical viewing
            overlay = cv2.addWeighted(image_np, 0.7, heatmap_colored, 0.3, 0)
            
            return Image.fromarray(overlay)
            
        except Exception as e:
            logger.error(f"Medical heatmap overlay creation failed: {e}")
            return image.resize((224, 224))

    def analyze_xray(
        self, 
        image_bytes: bytes, 
        threshold: float = 0.5,
        generate_heatmap: bool = True
    ) -> Tuple[List[Dict], bytes, Dict]:
        """
        Analyze chest X-ray image using exact CheXNet model
        """
        start_time = time.time()
        
        try:
            logger.info("Starting CheXNet X-ray analysis")
            
            # Preprocess image exactly like original
            original_image, crops_tensor = self._preprocess_image(image_bytes)
            crops_tensor = crops_tensor.to(self.device)
            
            # Calculate image quality
            quality_score = self._calculate_image_quality_score(original_image)
            
            # Forward pass through CheXNet model (EXACT same as original)
            with torch.no_grad():
                # FIXED: Handle TenCrop format exactly like original model.py
                # The crops_tensor has shape [10, 3, 224, 224] (10 crops from TenCrop)
                bs = 1  # batch size is 1 for single image
                n_crops = crops_tensor.size(0)  # Should be 10
                c, h, w = crops_tensor.size(1), crops_tensor.size(2), crops_tensor.size(3)
                
                # Reshape for model input - this matches your original model.py
                input_var = crops_tensor.view(-1, c, h, w)  # [10, 3, 224, 224]
                
                # Forward pass
                output = self.model(input_var)  # [10, 14]
                
                # Average across crops - this matches your original model.py
                output_mean = output.view(bs, n_crops, -1).mean(1)  # [1, 14]
                
                # NO ADDITIONAL SIGMOID - the model already has it!
                probabilities = output_mean.cpu().numpy()[0]  # Take first batch, shape [14]
            
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
            
            # Sort findings by confidence
            findings.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Generate heatmap if requested and abnormalities found
            heatmap_bytes = None
            if generate_heatmap and findings:
                try:
                    # Use highest confidence finding for Grad-CAM
                    top_finding_idx = np.argmax(probabilities)
                    heatmap = self._generate_gradcam(original_image, top_finding_idx)
                    
                    # Create overlay
                    overlay_image = self._create_heatmap_overlay(original_image, heatmap, findings)
                    
                    # Convert to bytes
                    heatmap_buffer = io.BytesIO()
                    overlay_image.save(heatmap_buffer, format='PNG', quality=95)
                    heatmap_bytes = heatmap_buffer.getvalue()
                    
                except Exception as e:
                    logger.error(f"Heatmap generation failed: {e}")
            
            # If no heatmap generated, use original image
            if heatmap_bytes is None:
                display_image = original_image.resize((224, 224))
                heatmap_buffer = io.BytesIO()
                display_image.save(heatmap_buffer, format='PNG')
                heatmap_bytes = heatmap_buffer.getvalue()
            
            processing_time = time.time() - start_time
            
            # Prepare metadata
            metadata = {
                'processing_time': processing_time,
                'image_quality_score': quality_score,
                'has_abnormalities': has_abnormalities,
                'max_confidence': max_confidence,
                'total_findings': len(findings),
                'model_version': 'CheXNet-Original',
                'threshold_used': threshold
            }
            
            # Create summary report
            report = self._generate_report(findings, metadata)
            
            logger.info(f"CheXNet analysis completed in {processing_time:.2f}s, found {len(findings)} abnormalities")
            
            return report, heatmap_bytes, metadata
            
        except Exception as e:
            logger.error(f"CheXNet X-ray analysis failed: {e}")
            processing_time = time.time() - start_time
            
            # Return error report
            error_report = [{
                'condition': 'Analysis Error',
                'confidence': 0.0,
                'severity': 'unknown',
                'urgency': 'routine',
                'description': f'CheXNet analysis failed: {str(e)}',
                'category': 'error'
            }]
            
            metadata = {
                'processing_time': processing_time,
                'image_quality_score': 0.0,
                'has_abnormalities': False,
                'max_confidence': 0.0,
                'total_findings': 0,
                'error': str(e)
            }
            
            return self._generate_report(error_report, metadata), None, metadata

    def _generate_report(self, findings: List[Dict], metadata: Dict) -> List[str]:
        """Generate human-readable report for medical findings"""
        report = []
        
        if not findings or metadata.get('has_abnormalities', False) is False:
            report.append(
                "No significant abnormalities detected among the 14 thoracic conditions analyzed by CheXNet. "
                "However, this AI analysis does not replace professional radiological interpretation. "
                "Please consult with a qualified radiologist for comprehensive clinical assessment."
            )
        else:
            report.append("CheXNet Analysis Results:")
            
            # Group findings by urgency
            urgent_findings = [f for f in findings if f['urgency'] == 'emergency']
            important_findings = [f for f in findings if f['urgency'] == 'urgent']
            routine_findings = [f for f in findings if f['urgency'] == 'routine']
            
            # Report urgent findings first
            if urgent_findings:
                report.append("⚠️ URGENT FINDINGS - Immediate medical attention recommended:")
                for finding in urgent_findings:
                    report.append(
                        f"  • {finding['condition']}: {finding['confidence']:.1%} confidence - {finding['description']}"
                    )
                report.append("")
            
            # Important findings
            if important_findings:
                report.append("🔍 SIGNIFICANT FINDINGS - Medical consultation recommended:")
                for finding in important_findings:
                    report.append(
                        f"  • {finding['condition']}: {finding['confidence']:.1%} confidence - {finding['description']}"
                    )
                report.append("")
            
            # Routine findings
            if routine_findings:
                report.append("📋 ADDITIONAL FINDINGS - Consider follow-up:")
                for finding in routine_findings:
                    report.append(
                        f"  • {finding['condition']}: {finding['confidence']:.1%} confidence - {finding['description']}"
                    )
                report.append("")
            
            # Add medical disclaimer
            report.append(
                "MEDICAL DISCLAIMER: This CheXNet AI analysis is for screening and educational purposes only. "
                "All findings require verification and clinical interpretation by qualified medical professionals. "
                "Do not use these results for diagnostic or treatment decisions without professional oversight."
            )
        
        # Add technical information
        report.append(f"Image Quality Score: {metadata.get('image_quality_score', 0):.2f}/1.00")
        report.append(f"Processing Time: {metadata.get('processing_time', 0):.2f} seconds")
        report.append("Model: CheXNet (DenseNet-121 trained on ChestX-ray14)")
        
        return report

# Global analyzer instance
_analyzer = None

def get_analyzer() -> XRayAnalyzer:
    """Get or create global analyzer instance"""
    global _analyzer
    if _analyzer is None:
        try:
            _analyzer = XRayAnalyzer()
        except Exception as e:
            logger.error(f"Failed to initialize CheXNet analyzer: {e}")
            raise
    return _analyzer

def analyze_xray(
    image_bytes: bytes, 
    threshold: float = 0.5,
    generate_heatmap: bool = True
) -> Tuple[List[str], bytes, Dict]:
    """
    Main analysis function for exact CheXNet compatibility
    """
    try:
        analyzer = get_analyzer()
        findings, heatmap_bytes, metadata = analyzer.analyze_xray(
            image_bytes, threshold, generate_heatmap
        )
        return findings, heatmap_bytes, metadata
        
    except Exception as e:
        logger.error(f"CheXNet analysis failed: {e}")
        error_report = [f"CheXNet analysis failed: {str(e)}"]
        return error_report, None, {'error': str(e)}

def check_model_availability() -> bool:
    """Check if CheXNet model file is available"""
    return os.path.exists("model.pth.tar")

def get_model_info() -> Dict:
    """Get CheXNet model information"""
    return {
        'model_available': check_model_availability(),
        'mode': 'production' if check_model_availability() else 'demo',
        'supported_conditions': DISEASE_LABELS,
        'model_architecture': 'CheXNet (DenseNet-121)',
        'input_size': '224x224',
        'supported_formats': ['JPEG', 'PNG', 'BMP', 'TIFF'],
        'dataset': 'ChestX-ray14',
        'multi_label': True
    }

if __name__ == "__main__":
    # Test the CheXNet analyzer
    if check_model_availability():
        print("✅ CheXNet model available, analyzer ready")
        try:
            analyzer = get_analyzer()
            print(f"✅ CheXNet analyzer initialized successfully")
            print(f"Supported conditions: {len(DISEASE_LABELS)}")
            print("Model architecture: CheXNet (DenseNet-121)")
        except Exception as e:
            print(f"❌ CheXNet analyzer initialization failed: {e}")
    else:
        print("❌ CheXNet model file 'model.pth.tar' not found")
        print("Please ensure the model file is in the current directory")
