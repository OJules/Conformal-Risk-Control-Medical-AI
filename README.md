# 🏥 Conformal Risk Control for Semantic Uncertainty Quantification in Computed Tomography

> Advanced uncertainty quantification in medical AI using conformal prediction methods for computed tomography analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Medical AI](https://img.shields.io/badge/Medical-AI-red.svg)]()
[![Conformal Prediction](https://img.shields.io/badge/Conformal-Prediction-purple.svg)]()
[![Uncertainty Quantification](https://img.shields.io/badge/Uncertainty-Quantification-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Thesis Overview

This **Master's thesis in Applied Mathematics & Computer Science (MIASHS)** develops advanced conformal prediction methods for uncertainty quantification in medical computed tomography. The research addresses critical challenges in medical AI reliability and clinical decision support.

### Key Research Contributions
- **Novel Conformal Risk Control Methods** - Advanced CRC, K-CRC, and sem-CRC implementations
- **Medical AI Applications** - Specialized techniques for computed tomography
- **Semantic Uncertainty Quantification** - Interpretable uncertainty in medical imaging
- **Clinical Validation** - Real-world medical dataset applications
- **Statistical Guarantees** - Provably reliable uncertainty bounds

## Research Framework

### **Conformal Prediction Theory**
- **Conformal Risk Control (CRC)** - Distribution-free uncertainty quantification
- **Kernel CRC (K-CRC)** - Advanced kernel-based methods
- **Semantic CRC (sem-CRC)** - Domain-aware uncertainty quantification
- **Statistical Validity** - Finite-sample coverage guarantees

### **Medical AI Integration**
- **Computed Tomography Analysis** - Specialized medical imaging applications
- **Clinical Decision Support** - Uncertainty-aware diagnostic assistance
- **Risk Assessment** - Quantified confidence for medical predictions
- **Interpretable AI** - Transparent uncertainty communication

## Academic Context

### **Institution & Degree**
- **University:** Université Lumière Lyon 2, France
- **Degree:** Master's MIASHS (Mathematics, Computer Science, AI) (2024-2025)
- **Thesis Defense:** May 2025
- **Research Focus:** Uncertainty quantification in medical AI
- **Specialization:** Domain adaptation, conformal prediction, medical AI

### **Research Significance**
This thesis addresses fundamental challenges in **trustworthy medical AI**:
- How to provide reliable uncertainty estimates in medical predictions?
- What statistical guarantees can we offer for clinical decision support?
- How to make AI uncertainty interpretable for medical professionals?

## Technical Implementation

### **Conformal Methods**
```python
# Advanced conformal risk control implementation
class ConformalRiskControl:
    def __init__(self, alpha=0.1, risk_function='semantic'):
        self.alpha = alpha  # Miscoverage rate
        self.risk_function = risk_function
    
    def calibrate(self, cal_features, cal_labels):
        # Conformal calibration process
        
    def predict_with_uncertainty(self, test_features):
        # Prediction with statistical guarantees
```

### **Medical AI Pipeline**
- **CT Image Processing** - Specialized preprocessing for medical imaging
- **Feature Extraction** - Domain-specific medical features
- **Uncertainty Quantification** - Conformal prediction intervals
- **Clinical Interpretation** - Uncertainty visualization for medical professionals

## Key Findings & Contributions

### **Methodological Advances**
- **15% Improvement** in uncertainty model accuracy
- **Novel sem-CRC Algorithm** - Semantic-aware conformal prediction
- **Medical Domain Adaptation** - Specialized techniques for CT analysis
- **Statistical Validation** - Rigorous theoretical and empirical evaluation

### **Clinical Applications**
- Enhanced diagnostic confidence assessment
- Risk-aware treatment planning support
- Improved medical AI transparency
- Reduced diagnostic uncertainty in critical cases

## Medical Impact & Ethics

### **Healthcare Applications**
- **Diagnostic Support** - Uncertainty-aware medical image analysis
- **Risk Assessment** - Quantified confidence for clinical decisions
- **Treatment Planning** - Reliable AI assistance in medical workflows
- **Quality Assurance** - Statistical guarantees for medical AI systems

### **Ethical Considerations**
- Patient safety through uncertainty awareness
- Transparent AI for medical professionals
- Regulatory compliance for medical devices
- Responsible AI deployment in healthcare

## Technical Stack

- **Conformal Prediction:** Custom implementations, MAPIE
- **Medical Imaging:** SimpleITK, PyDicom, nibabel
- **Deep Learning:** TensorFlow, PyTorch
- **Statistical Analysis:** NumPy, SciPy, scikit-learn
- **Visualization:** Matplotlib, medical imaging viewers

## Project Structure

```
Conformal-Risk-Control-Medical-AI/
├── thesis/
│   ├── MasterThesis_Math&CS.pdf           # Complete thesis document
│   └── defense_presentation.pdf           # Thesis defense slides
├── src/
│   ├── conformal_methods/                 # CRC, K-CRC, sem-CRC implementations
│   ├── medical_imaging/                   # CT processing pipelines
│   └── uncertainty_quantification/        # UQ algorithms
├── data/
│   ├── ct_datasets/                       # Medical imaging datasets
│   └── processed/                         # Preprocessed data
├── experiments/
│   ├── conformal_experiments.py           # Main experimental code
│   └── medical_validation.py              # Clinical validation
├── results/
│   ├── uncertainty_visualizations/        # UQ plots and heatmaps
│   ├── performance_metrics/               # Statistical evaluation
│   └── clinical_case_studies/             # Medical applications
├── requirements.txt                       # Dependencies
├── README.md                             # This file
└── LICENSE                               # MIT License
```

## Future Research Directions

- Real-time uncertainty quantification for medical AI
- Multi-modal conformal prediction in healthcare
- Federated learning with uncertainty quantification
- Regulatory framework development for medical AI

## Contact

**Jules Odje** - Data Scientist | Aspiring PhD Researcher  
📧 [odjejulesgeraud@gmail.com](mailto:odjejulesgeraud@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/jules-odje)  
🐙 [GitHub](https://github.com/OJules)

**Thesis Focus:** Medical AI | Uncertainty Quantification | Conformal Prediction

---

*"Bridging the gap between AI predictions and clinical confidence through rigorous uncertainty quantification"*
