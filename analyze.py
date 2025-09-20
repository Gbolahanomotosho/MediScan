#!/usr/bin/env python3
# analyze.py - COMPLETELY FIXED Grad-CAM Implementation for CheXNet
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
    """COMPLETELY REWRITTEN Grad-CAM implementation that actually works"""
    
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
            self.features = output.detach()
            logger.debug(f"✅ Forward hook captured features: {output.shape}")
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()
                logger.debug(f"✅ Backward hook captured gradients: {grad_output[0].shape}")
        
        # Register hooks and store handles for cleanup
        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)
        self.handles.extend([h1, h2])
        
        logger.info("✅ Grad-CAM hooks registered successfully")
    
    def generate_cam(self, class_idx: int) -> np.ndarray:
        """Generate Class Activation Map"""
        if self.features is None or self.gradients is None:
            logger.error("❌ Features or gradients not captured")
            return np.zeros((224, 224))
        
        # Calculate weights (Global Average Pooling of gradients)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        logger.debug(f"Weights shape: {weights.shape}")
        
        # Weighted combination of feature maps
        cam = torch.sum(weights * self.features, dim=1).squeeze(0)
        logger.debug(f"CAM shape before ReLU: {cam.shape}")
        
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
        
        logger.info(f"✅ CAM generated: shape={cam.shape}, range=[{cam.min():.3f}, {cam.max():.3f}]")
        return cam

class XRayAnalyzer:
    """COMPLETELY FIXED X-ray analyzer with working Grad-CAM"""
    
    def __init__(self, model_path: str = "model.pth.tar", device: str = "cpu"):
        """Initialize the X-ray analyzer"""
        self.device = torch.device(device)
        self.model = None
        self.gradcam_hook = GradCAMHook()
        self.model_path = model_path
        self.target_layer = None
        
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
        
        # Single image transform for Grad-CAM
        self.single_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
        
        logger.info("✅ XRayAnalyzer initialized successfully")

    def _load_model(self):
        """Load the exact CheXNet model"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"❌ Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
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
            
            # Find and register target layer for Grad-CAM
            self._setup_gradcam()
            
            logger.info("✅ CheXNet model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def _setup_gradcam(self):
        """COMPLETELY REWRITTEN Grad-CAM setup"""
        try:
            # Find the target layer - last convolutional layer in DenseNet
            # Path: model.module.densenet121.features.denseblock4.denselayer16.conv2
            
            target_layer = None
            
            # Method 1: Direct path to last conv layer in DenseNet121
            try:
                target_layer = self.model.module.densenet121.features.denseblock4.denselayer16.conv2
                logger.info("✅ Found target layer: denseblock4.denselayer16.conv2")
            except AttributeError:
                logger.warning("Standard path failed, searching for alternatives...")
                
                # Method 2: Search all conv layers
                conv_layers = []
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        conv_layers.append((name, module))
                
                if conv_layers:
                    # Use the deepest conv layer before classifier
                    layer_name, target_layer = conv_layers[-1]
                    logger.info(f"Using conv layer: {layer_name}")
                else:
                    # Method 3: Try features module directly
                    try:
                        features = self.model.module.densenet121.features
                        # Get the last layer in features
                        for name, layer in features.named_modules():
                            if isinstance(layer, nn.Conv2d):
                                target_layer = layer
                        logger.info("Using last features conv layer")
                    except:
                        raise AttributeError("Could not find any suitable layer")
            
            if target_layer is not None:
                self.target_layer = target_layer
                self.gradcam_hook.register_hooks(target_layer)
                logger.info("✅ Grad-CAM setup completed successfully")
            else:
                raise AttributeError("No suitable target layer found")
                
        except Exception as e:
            logger.error(f"❌ Grad-CAM setup failed: {e}")
            self.target_layer = None

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
        """COMPLETELY REWRITTEN Grad-CAM generation"""
        try:
            if self.target_layer is None:
                logger.error("❌ No target layer available")
                return np.zeros((224, 224))
            
            logger.info(f"Generating Grad-CAM for class {class_idx} ({DISEASE_LABELS[class_idx]})")
            
            # Clear previous hooks data
            self.gradcam_hook.features = None
            self.gradcam_hook.gradients = None
            
            # Prepare image for single inference
            image_tensor = self.single_transform(image).unsqueeze(0).to(self.device)
            image_tensor.requires_grad_(True)
            
            # Clear model gradients
            self.model.zero_grad()
            
            # Forward pass
            logger.debug("Performing forward pass...")
            with torch.set_grad_enabled(True):
                output = self.model(image_tensor)
                
                # Check if hooks captured features
                if self.gradcam_hook.features is None:
                    logger.error("❌ Forward hook failed to capture features")
                    return np.zeros((224, 224))
                
                # Get class score and perform backward pass
                if output.dim() > 1:
                    class_score = output[0, class_idx]
                else:
                    class_score = output[class_idx]
                
                logger.debug(f"Class score: {class_score.item():.4f}")
                
                # Backward pass
                logger.debug("Performing backward pass...")
                class_score.backward(retain_graph=False)
                
                # Check if hooks captured gradients
                if self.gradcam_hook.gradients is None:
                    logger.error("❌ Backward hook failed to capture gradients")
                    return np.zeros((224, 224))
                
                # Generate CAM
                logger.debug("Generating CAM...")
                cam = self.gradcam_hook.generate_cam(class_idx)
                
                return cam
                
        except Exception as e:
            logger.error(f"❌ Grad-CAM generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return np.zeros((224, 224))

    def _create_heatmap_overlay(self, image: Image.Image, heatmap: np.ndarray) -> Image.Image:
        """IMPROVED heatmap overlay creation"""
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
            
            # Create colored heatmap - using JET colormap for medical imaging
            heatmap_colored = cv2.applyColorMap(
                np.uint8(255 * heatmap_norm), 
                cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Create overlay with transparency
            alpha = 0.4  # Transparency factor
            overlay = cv2.addWeighted(image_np, 1-alpha, heatmap_colored, alpha, 0)
            
            # Convert back to PIL Image
            result_image = Image.fromarray(overlay)
            
            # Check if we actually have meaningful heatmap data
            non_zero_pixels = np.sum(heatmap_norm > threshold)
            logger.info(f"✅ Heatmap overlay created with {non_zero_pixels} highlighted pixels")
            
            if non_zero_pixels == 0:
                logger.warning("⚠️ No significant heatmap regions found")
            
            return result_image
            
        except Exception as e:
            logger.error(f"❌ Heatmap overlay creation failed: {e}")
            return image.resize((224, 224))

    def analyze_xray(
        self, 
        image_bytes: bytes, 
        threshold: float = 0.5,
        generate_heatmap: bool = True
    ) -> Tuple[List[Dict], bytes, Dict]:
        """MAIN analysis function with FIXED heatmap generation"""
        start_time = time.time()
        
        try:
            logger.info("🔬 Starting CheXNet X-ray analysis...")
            
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
            
            # Generate heatmap
            heatmap_bytes = None
            heatmap_generated = False
            
            if generate_heatmap and self.target_layer is not None:
                try:
                    logger.info("🎯 Starting Grad-CAM heatmap generation...")
                    
                    # Use the highest confidence class for heatmap
                    top_class_idx = np.argmax(probabilities)
                    top_confidence = probabilities[top_class_idx]
                    
                    logger.info(f"Generating heatmap for: {DISEASE_LABELS[top_class_idx]} (confidence: {top_confidence:.3f})")
                    
                    # Generate Grad-CAM
                    heatmap = self._generate_gradcam(original_image, top_class_idx)
                    
                    # Check if heatmap is meaningful
                    if np.any(heatmap > 0):
                        # Create overlay
                        overlay_image = self._create_heatmap_overlay(original_image, heatmap)
                        
                        # Convert to bytes
                        heatmap_buffer = io.BytesIO()
                        overlay_image.save(heatmap_buffer, format='PNG', quality=95, optimize=True)
                        heatmap_bytes = heatmap_buffer.getvalue()
                        heatmap_generated = True
                        
                        logger.info("✅ Grad-CAM heatmap generated successfully!")
                    else:
                        logger.warning("⚠️ Generated heatmap is empty")
                        
                except Exception as e:
                    logger.error(f"❌ Heatmap generation error: {e}")
            
            # Fallback to original image if heatmap failed
            if heatmap_bytes is None:
                logger.info("📷 Using original image (no heatmap generated)")
                display_image = original_image.resize((224, 224))
                heatmap_buffer = io.BytesIO()
                display_image.save(heatmap_buffer, format='PNG')
                heatmap_bytes = heatmap_buffer.getvalue()
            
            processing_time = time.time() - start_time
            
            # Metadata
            metadata = {
                'processing_time': processing_time,
                'image_quality_score': 0.8,  # Placeholder
                'has_abnormalities': has_abnormalities,
                'max_confidence': max_confidence,
                'total_findings': len(findings),
                'model_version': 'CheXNet-DenseNet121',
                'threshold_used': threshold,
                'heatmap_generated': heatmap_generated,
                'gradcam_available': self.target_layer is not None
            }
            
            # Generate report
            report = self._generate_report(findings, metadata)
            
            logger.info(f"✅ Analysis completed in {processing_time:.2f}s, {len(findings)} findings, heatmap: {heatmap_generated}")
            
            return report, heatmap_bytes, metadata
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Error report
            processing_time = time.time() - start_time
            error_report = [f"Analysis failed: {str(e)}"]
            metadata = {
                'processing_time': processing_time,
                'error': str(e),
                'heatmap_generated': False
            }
            return error_report, None, metadata

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
                report.append("🚨 URGENT FINDINGS - Immediate attention required:")
                for finding in urgent_findings:
                    report.append(f"  • {finding['condition']}: {finding['confidence']:.3f} probability - {finding['description']}")
            
            if important_findings:
                report.append("⚠️ SIGNIFICANT FINDINGS - Medical consultation recommended:")
                for finding in important_findings:
                    report.append(f"  • {finding['condition']}: {finding['confidence']:.3f} probability - {finding['description']}")
            
            if routine_findings:
                report.append("📋 ADDITIONAL FINDINGS - Consider follow-up:")
                for finding in routine_findings:
                    report.append(f"  • {finding['condition']}: {finding['confidence']:.3f} probability - {finding['description']}")
        
        # Technical info
        report.append(f"Processing Time: {metadata.get('processing_time', 0):.2f} seconds")
        report.append("Model: CheXNet (DenseNet-121 on ChestX-ray14)")
        
        if metadata.get('heatmap_generated', False):
            report.append("✅ Grad-CAM Heatmap: Successfully generated")
        else:
            report.append("❌ Grad-CAM Heatmap: Generation failed")
        
        return report

    def __del__(self):
        """Cleanup hooks when object is destroyed"""
        if hasattr(self, 'gradcam_hook'):
            self.gradcam_hook.clear_hooks()

# Demo mode functions (unchanged)
def _generate_demo_heatmap(original_image: Image.Image) -> bytes:
    """Generate synthetic heatmap for demo mode"""
    try:
        img = original_image.resize((224, 224)).convert('RGB')
        img_array = np.array(img)
        
        # Create synthetic heatmap
        height, width = 224, 224
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Add hotspots
        y1, x1 = 120, 140
        for i in range(40):
            for j in range(35):
                if i + y1 < height and j + x1 < width:
                    distance = np.sqrt((i - 20)**2 + (j - 17)**2)
                    if distance < 20:
                        heatmap[i + y1, j + x1] = max(0, 0.8 - distance/25)
        
        y2, x2 = 60, 80
        for i in range(30):
            for j in range(30):
                if i + y2 < height and j + x2 < width:
                    distance = np.sqrt((i - 15)**2 + (j - 15)**2)
                    if distance < 15:
                        heatmap[i + y2, j + x2] = max(0, 0.6 - distance/20)
        
        # Add noise and smooth
        noise = np.random.normal(0, 0.05, (height, width))
        heatmap = np.clip(heatmap + noise, 0, 1)
        heatmap = cv2.GaussianBlur(heatmap, (9, 9), 0)
        
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Create colored overlay
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
        
        overlay_image = Image.fromarray(overlay)
        heatmap_buffer = io.BytesIO()
        overlay_image.save(heatmap_buffer, format='PNG', quality=95)
        
        logger.info("✅ Demo heatmap generated")
        return heatmap_buffer.getvalue()
        
    except Exception as e:
        logger.error(f"❌ Demo heatmap failed: {e}")
        return None

def _demo_analysis(image_bytes: bytes, threshold: float = 0.5, generate_heatmap: bool = True) -> Tuple[List[str], bytes, Dict]:
    """Demo mode analysis"""
    start_time = time.time()
    logger.info("🎭 Running DEMO mode analysis")
    
    try:
        original_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Demo conditions
        demo_conditions = [
            ('Consolidation', 0.683, 'urgent', 'Lung filled with liquid'),
            ('Infiltration', 0.524, 'routine', 'Lung tissue changes'),
            ('Pneumonia', 0.445, 'urgent', 'Lung infection'),
        ]
        
        findings = []
        has_abnormalities = False
        max_confidence = 0.0
        
        for condition, confidence, urgency, description in demo_conditions:
            if confidence > threshold:
                has_abnormalities = True
                max_confidence = max(max_confidence, confidence)
                
                findings.append({
                    'condition': condition,
                    'confidence': confidence,
                    'severity': 'moderate',
                    'urgency': urgency,
                    'description': description,
                    'category': 'thoracic_disease'
                })
        
        # Generate demo heatmap
        heatmap_bytes = None
        if generate_heatmap:
            heatmap_bytes = _generate_demo_heatmap(original_image)
        
        if heatmap_bytes is None:
            display_image = original_image.resize((224, 224))
            heatmap_buffer = io.BytesIO()
            display_image.save(heatmap_buffer, format='PNG')
            heatmap_bytes = heatmap_buffer.getvalue()
        
        processing_time = time.time() - start_time
        
        metadata = {
            'processing_time': processing_time,
            'image_quality_score': 0.75,
            'has_abnormalities': has_abnormalities,
            'max_confidence': max_confidence,
            'total_findings': len(findings),
            'model_version': 'Demo-Mode',
            'threshold_used': threshold,
            'mode': 'demo',
            'heatmap_generated': heatmap_bytes is not None
        }
        
        report = _generate_demo_report(findings, metadata)
        
        logger.info(f"✅ Demo analysis completed in {processing_time:.2f}s")
        return report, heatmap_bytes, metadata
        
    except Exception as e:
        logger.error(f"❌ Demo analysis failed: {e}")
        processing_time = time.time() - start_time
        
        return [f"Demo analysis failed: {str(e)}"], None, {
            'processing_time': processing_time,
            'error': str(e),
            'mode': 'demo'
        }

def _generate_demo_report(findings: List[Dict], metadata: Dict) -> List[str]:
    """Generate demo report"""
    report = []
    
    report.append("[DEMO MODE] This is a demonstration with sample results.")
    report.append("")
    
    if not findings:
        report.append("No significant abnormalities in this demo analysis.")
    else:
        report.append("Demo Analysis Results:")
        
        urgent = [f for f in findings if f['urgency'] == 'urgent']
        routine = [f for f in findings if f['urgency'] == 'routine']
        
        if urgent:
            report.append("🔍 SIGNIFICANT FINDINGS:")
            for finding in urgent:
                report.append(f"  • {finding['condition']}: {finding['confidence']:.3f} probability - {finding['description']}")
        
        if routine:
            report.append("📋 ADDITIONAL FINDINGS:")
            for finding in routine:
                report.append(f"  • {finding['condition']}: {finding['confidence']:.3f} probability - {finding['description']}")
    
    report.append("")
    report.append("DEMO DISCLAIMER: Simulated results for demonstration only.")
    report.append(f"Processing Time: {metadata.get('processing_time', 0):.2f} seconds")
    report.append("Model: Demo Mode (No trained model)")
    
    if metadata.get('heatmap_generated', False):
        report.append("✅ Demo Heatmap: Generated")
    else:
        report.append("❌ Demo Heatmap: Failed")
    
    return report

# Global analyzer instance
_analyzer = None

def get_analyzer() -> XRayAnalyzer:
    """Get or create global analyzer instance"""
    global _analyzer
    if _analyzer is None:
        try:
            if not check_model_availability():
                logger.warning("No model available, analyzer cannot be initialized")
                raise FileNotFoundError("Model file not found - running in demo mode")
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
    Main analysis function with automatic demo/production mode selection
    """
    model_available = check_model_availability()
    
    if not model_available:
        logger.info("🎭 DEMO MODE - No trained model available")
        return _demo_analysis(image_bytes, threshold, generate_heatmap)
    
    try:
        logger.info("🔬 PRODUCTION MODE - Using trained model")
        analyzer = get_analyzer()
        findings, heatmap_bytes, metadata = analyzer.analyze_xray(
            image_bytes, threshold, generate_heatmap
        )
        return findings, heatmap_bytes, metadata
        
    except Exception as e:
        logger.error(f"Production analysis failed, falling back to demo: {e}")
        return _demo_analysis(image_bytes, threshold, generate_heatmap)

def check_model_availability() -> bool:
    """Check if model file exists"""
    return os.path.exists("model.pth.tar")

def get_model_info() -> Dict:
    """Get model information"""
    return {
        'model_available': check_model_availability(),
        'mode': 'production' if check_model_availability() else 'demo',
        'supported_conditions': DISEASE_LABELS,
        'model_architecture': 'CheXNet (DenseNet-121)',
        'input_size': '224x224',
        'supported_formats': ['JPEG', 'PNG', 'BMP', 'TIFF'],
        'dataset': 'ChestX-ray14',
        'multi_label': True,
        'gradcam_support': True
    }

if __name__ == "__main__":
    # Test the analyzer
    if check_model_availability():
        print("✅ Model available, testing production mode...")
        try:
            analyzer = get_analyzer()
            print("✅ Analyzer initialized successfully")
            print(f"Target layer available: {analyzer.target_layer is not None}")
            print(f"Grad-CAM hooks registered: {len(analyzer.gradcam_hook.handles) > 0}")
        except Exception as e:
            print(f"❌ Analyzer initialization failed: {e}")
    else:
        print("❌ Model file 'model.pth.tar' not found")
        print("🎭 Running in demo mode with synthetic results")
        
        # Test demo mode
        try:
            # Create a small test image
            test_image = Image.new('RGB', (256, 256), color='gray')
            img_buffer = io.BytesIO()
            test_image.save(img_buffer, format='PNG')
            test_bytes = img_buffer.getvalue()
            
            findings, heatmap_bytes, metadata = analyze_xray(test_bytes)
            print(f"✅ Demo analysis completed: {len(findings)} findings")
            print(f"Heatmap generated: {heatmap_bytes is not None}")
            
        except Exception as e:
            print(f"❌ Demo analysis failed: {e}")
    
    print(f"\nModel Info: {get_model_info()}")
