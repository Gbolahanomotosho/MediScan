#!/usr/bin/env python3
# analyze.py - Advanced X-ray Analysis Engine
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import io
import logging
from typing import Tuple, List, Dict, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 14 Thoracic disease labels (from ChestX-ray14 dataset)
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

class XRayAnalyzer:
    def __init__(self, model_path: str = "model.pth.tar", device: str = "cpu"):
        """Initialize the X-ray analyzer"""
        self.device = torch.device(device)
        self.model = None
        self.features = None
        self.gradients = None
        self.model_path = model_path
        
        # Load model
        self._load_model()
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])(crop) for crop in crops
            ]))
        ])
        
        # Single image transform for visualization
        self.single_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        logger.info("XRayAnalyzer initialized successfully")

    def _load_model(self):
        """Load the pre-trained model"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Initialize DenseNet-121 model
            self.model = models.densenet121(pretrained=False)
            self.model.classifier = torch.nn.Linear(1024, len(DISEASE_LABELS))
            
            # Load weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Register hooks for Grad-CAM
            self._register_hooks()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _register_hooks(self):
        """Register forward and backward hooks for Grad-CAM"""
        def forward_hook(module, input, output):
            self.features = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Register hooks on the last convolutional layer
        target_layer = self.model.features.denseblock4.denselayer16.conv2
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def _preprocess_image(self, image_bytes: bytes) -> Tuple[Image.Image, torch.Tensor]:
        """Preprocess image for analysis"""
        try:
            # Load image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Basic image quality checks
            width, height = image.size
            if width < 224 or height < 224:
                logger.warning(f"Image resolution is low: {width}x{height}")
            
            # Enhance image quality
            image = self._enhance_image(image)
            
            # Apply transforms
            crops_tensor = self.transform(image)
            
            return image, crops_tensor
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise

    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Enhance image quality for better analysis"""
        try:
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Apply slight denoising
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image

    def _calculate_image_quality_score(self, image: Image.Image) -> float:
        """Calculate image quality score"""
        try:
            # Convert to numpy array
            img_array = np.array(image.convert('L'))
            
            # Calculate sharpness using Laplacian variance
            laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Calculate contrast
            contrast = img_array.std()
            
            # Calculate brightness distribution
            histogram = np.histogram(img_array, bins=256)[0]
            brightness_distribution = np.sum(histogram[64:192]) / np.sum(histogram)
            
            # Combine metrics (normalize to 0-1 scale)
            quality_score = min(1.0, (
                min(sharpness / 500, 1.0) * 0.4 +
                min(contrast / 100, 1.0) * 0.4 +
                brightness_distribution * 0.2
            ))
            
            return float(quality_score)
            
        except Exception as e:
            logger.warning(f"Quality score calculation failed: {e}")
            return 0.5

    def _generate_gradcam(self, image: Image.Image, class_idx: int) -> np.ndarray:
        """Generate Grad-CAM heatmap"""
        try:
            # Prepare single image for Grad-CAM
            image_tensor = self.single_transform(image).unsqueeze(0).to(self.device)
            
            # Forward pass
            self.model.zero_grad()
            output = self.model(image_tensor)
            
            # Backward pass
            class_score = output[0, class_idx]
            class_score.backward()
            
            # Generate Grad-CAM
            if self.gradients is not None and self.features is not None:
                weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
                grad_cam = F.relu(torch.sum(weights * self.features, dim=1)).squeeze()
                
                # Normalize
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
                
                return grad_cam
            else:
                logger.warning("Grad-CAM features not available")
                return np.zeros((224, 224))
                
        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {e}")
            return np.zeros((224, 224))

    def _create_heatmap_overlay(self, image: Image.Image, heatmap: np.ndarray, findings: List[Dict]) -> Image.Image:
        """Create heatmap overlay with disease labels"""
        try:
            # Resize image and heatmap to same dimensions
            base_image = image.resize((224, 224)).convert("RGB")
            image_np = np.array(base_image)
            
            # Create colored heatmap
            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap), cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Blend images
            overlay = cv2.addWeighted(image_np, 0.6, heatmap_colored, 0.4, 0)
            
            # Add annotations for findings
            overlay_with_labels = self._add_finding_labels(overlay, findings, heatmap)
            
            return Image.fromarray(overlay_with_labels)
            
        except Exception as e:
            logger.error(f"Heatmap overlay creation failed: {e}")
            return image.resize((224, 224))

    def _add_finding_labels(self, image: np.ndarray, findings: List[Dict], heatmap: np.ndarray) -> np.ndarray:
        """Add disease labels to the heatmap image"""
        try:
            # Find regions of high activation
            threshold = 0.7
            high_activation = heatmap > threshold
            
            if np.any(high_activation):
                # Find contours of high activation regions
                high_activation_uint8 = (high_activation * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    high_activation_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Add labels for significant findings
                for i, finding in enumerate(findings[:3]):  # Limit to top 3 findings
                    if finding['confidence'] > 0.5 and len(contours) > i:
                        # Get centroid of contour
                        M = cv2.moments(contours[i])
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Add text label
                            label = f"{finding['condition']}: {finding['confidence']:.1%}"
                            cv2.putText(
                                image, label, (cx - 50, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                            )
                            
                            # Add colored circle
                            color = (0, 255, 0) if finding['confidence'] < 0.7 else (0, 165, 255)
                            cv2.circle(image, (cx, cy), 8, color, 2)
            
            return image
            
        except Exception as e:
            logger.error(f"Label addition failed: {e}")
            return image

    def analyze_xray(
        self, 
        image_bytes: bytes, 
        threshold: float = 0.5,
        generate_heatmap: bool = True
    ) -> Tuple[List[Dict], bytes, Dict]:
        """
        Analyze chest X-ray image for thoracic diseases
        
        Args:
            image_bytes: Raw image bytes
            threshold: Confidence threshold for positive findings
            generate_heatmap: Whether to generate Grad-CAM heatmap
            
        Returns:
            Tuple of (findings_list, heatmap_bytes, metadata)
        """
        start_time = time.time()
        
        try:
            logger.info("Starting X-ray analysis")
            
            # Preprocess image
            original_image, crops_tensor = self._preprocess_image(image_bytes)
            crops_tensor = crops_tensor.to(self.device)
            
            # Calculate image quality
            quality_score = self._calculate_image_quality_score(original_image)
            
            # Forward pass through model
            with torch.no_grad():
                outputs = self.model(crops_tensor.view(-1, 3, 224, 224))
                probabilities = torch.sigmoid(outputs).mean(0).cpu().numpy()
            
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
                    
                    # Create overlay with labels
                    overlay_image = self._create_heatmap_overlay(original_image, heatmap, findings)
                    
                    # Convert to bytes
                    heatmap_buffer = io.BytesIO()
                    overlay_image.save(heatmap_buffer, format='PNG', quality=95)
                    heatmap_bytes = heatmap_buffer.getvalue()
                    
                except Exception as e:
                    logger.error(f"Heatmap generation failed: {e}")
                    # Create a simple overlay without Grad-CAM
                    simple_overlay = original_image.resize((224, 224))
                    heatmap_buffer = io.BytesIO()
                    simple_overlay.save(heatmap_buffer, format='PNG')
                    heatmap_bytes = heatmap_buffer.getvalue()
            
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
                'model_version': '1.0',
                'threshold_used': threshold
            }
            
            # Create summary report
            report = self._generate_report(findings, metadata)
            
            logger.info(f"Analysis completed in {processing_time:.2f}s, found {len(findings)} abnormalities")
            
            return report, heatmap_bytes, metadata
            
        except Exception as e:
            logger.error(f"X-ray analysis failed: {e}")
            processing_time = time.time() - start_time
            
            # Return error report
            error_report = [{
                'condition': 'Analysis Error',
                'confidence': 0.0,
                'severity': 'unknown',
                'urgency': 'routine',
                'description': f'Analysis failed: {str(e)}',
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
            
            return error_report, None, metadata

    def _generate_report(self, findings: List[Dict], metadata: Dict) -> List[str]:
        """Generate human-readable report"""
        report = []
        
        if not findings or metadata.get('has_abnormalities', False) is False:
            report.append(
                "No major abnormalities detected among the 14 thoracic conditions analyzed. "
                "However, this AI analysis does not replace professional medical evaluation. "
                "Please consult with a qualified radiologist for comprehensive assessment."
            )
        else:
            report.append("AI Analysis Results:")
            
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
            
            # Add disclaimer
            report.append(
                "IMPORTANT: This AI analysis is for screening purposes only and should not replace "
                "professional medical diagnosis. All findings require verification by qualified medical professionals."
            )
        
        # Add technical information
        report.append(f"Image Quality Score: {metadata.get('image_quality_score', 0):.2f}/1.00")
        report.append(f"Processing Time: {metadata.get('processing_time', 0):.2f} seconds")
        
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
            logger.error(f"Failed to initialize analyzer: {e}")
            raise
    return _analyzer

def analyze_xray(
    image_bytes: bytes, 
    threshold: float = 0.5,
    generate_heatmap: bool = True
) -> Tuple[List[str], bytes, Dict]:
    """
    Main analysis function for compatibility
    
    Args:
        image_bytes: Raw image data
        threshold: Confidence threshold for findings
        generate_heatmap: Whether to generate visualization
        
    Returns:
        Tuple of (report_strings, heatmap_bytes, metadata)
    """
    try:
        analyzer = get_analyzer()
        findings, heatmap_bytes, metadata = analyzer.analyze_xray(
            image_bytes, threshold, generate_heatmap
        )
        return findings, heatmap_bytes, metadata
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        error_report = [f"Analysis failed: {str(e)}"]
        return error_report, None, {'error': str(e)}

# Utility functions for model management
def check_model_availability() -> bool:
    """Check if model file is available"""
    return os.path.exists("model.pth.tar")

def get_model_info() -> Dict:
    """Get model information"""
    return {
        'model_available': check_model_availability(),
        'supported_conditions': DISEASE_LABELS,
        'model_architecture': 'DenseNet-121',
        'input_size': '224x224',
        'supported_formats': ['JPEG', 'PNG', 'BMP', 'TIFF']
    }

if __name__ == "__main__":
    # Test the analyzer
    if check_model_availability():
        print("✅ Model available, analyzer ready")
        analyzer = get_analyzer()
        print(f"✅ Analyzer initialized successfully")
        print(f"Supported conditions: {len(DISEASE_LABELS)}")
    else:
        print("❌ Model file 'model.pth.tar' not found")
        print("Please ensure the model file is in the current directory")
