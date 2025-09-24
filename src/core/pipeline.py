import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from typing import Dict, Any, Tuple

class MedicalImagePipeline:
    """Pipeline complet basé sur votre code de thèse."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def create_medical_image(self):
        """Votre fonction originale - génère l'image médicale."""
        image = np.zeros((128, 128))
        center = np.array([64, 64])
        y, x = np.ogrid[:128, :128]

        # Corps principal
        body_dist = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        image[body_dist < 50] = 0.4  # Tissu

        # Organe principal
        organ_dist = np.sqrt((x - center[1] + 10)**2 + (y - center[0])**2)
        image[organ_dist < 25] = 0.7  # Organe

        # Lésion critique
        lesion_dist = np.sqrt((x - center[1] + 5)**2 + (y - center[0] - 8)**2)
        image[lesion_dist < 4] = 1.0  # Pathologie

        return image

    def forward_propagation(self, clean_image, noise_level=0.15, blur_sigma=1.5):
        """Votre fonction originale - simulation de dégradation."""
        print("FORWARD PROPAGATION: Image → Observations")

        # 1. Sous-échantillonnage fréquentiel
        fft_img = np.fft.fft2(clean_image)
        center = np.array(fft_img.shape) // 2
        y, x = np.ogrid[:fft_img.shape[0], :fft_img.shape[1]]

        max_freq = min(fft_img.shape) // 2 * 0.3
        freq_mask = ((x - center[1])**2 + (y - center[0])**2) <= max_freq**2

        fft_degraded = fft_img * freq_mask
        undersampled = np.real(np.fft.ifft2(fft_degraded))

        # 2. Bruit quantique
        quantum_noise = np.random.normal(0, noise_level, clean_image.shape)
        noisy = undersampled + quantum_noise

        # 3. Flou
        blurred = ndimage.gaussian_filter(noisy, sigma=blur_sigma)

        # 4. Normalisation finale
        observations = np.clip(blurred, 0, 1)

        # Métriques
        mse_forward = np.mean((observations - clean_image)**2)
        psnr_forward = 20 * np.log10(1.0 / np.sqrt(mse_forward))

        print(f"  Dégradation - MSE: {mse_forward:.6f}, PSNR: {psnr_forward:.2f} dB")

        return observations, {'mse': mse_forward, 'psnr': psnr_forward}

    def backward_propagation(self, observations, ground_truth=None):
        """Votre fonction originale - reconstruction avec contrôle de risque."""
        print("BACKWARD PROPAGATION: Observations → Reconstruction")

        # 1. Reconstruction de base
        median_filtered = ndimage.median_filter(observations, size=3)
        reconstructed = ndimage.gaussian_filter(median_filtered, sigma=0.8)

        if reconstructed.max() > reconstructed.min():
            reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min())

        # 2. Segmentation
        segmentation = np.zeros_like(reconstructed, dtype=int)
        segmentation[reconstructed <= 0.2] = 0  # Background
        segmentation[(reconstructed > 0.2) & (reconstructed <= 0.5)] = 1  # Tissu
        segmentation[(reconstructed > 0.5) & (reconstructed <= 0.8)] = 2  # Organe
        segmentation[reconstructed > 0.8] = 3  # Pathologie

        # 3. Quantification d'incertitude
        if ground_truth is not None:
            errors = np.abs(reconstructed - ground_truth)
            uncertainty_map = np.zeros_like(reconstructed)

            clinical_factors = {0: 1.0, 1: 1.2, 2: 1.8, 3: 3.0}
            lambdas = {}

            for region_id in np.unique(segmentation):
                mask = (segmentation == region_id)
                if np.sum(mask) > 0:
                    regional_error = np.mean(errors[mask])
                    factor = clinical_factors.get(region_id, 1.0)
                    lambda_val = regional_error * factor

                    lambdas[region_id] = lambda_val
                    uncertainty_map[mask] = lambda_val
        else:
            local_variance = ndimage.generic_filter(reconstructed, np.var, size=5)
            uncertainty_map = local_variance
            lambdas = {0: 0.05, 1: 0.08, 2: 0.12, 3: 0.20}

        # 4. Intervalles de confiance
        confidence_levels = {0: 0.80, 1: 0.85, 2: 0.90, 3: 0.95}
        z_scores = {0.80: 1.28, 0.85: 1.44, 0.90: 1.645, 0.95: 1.96}

        lower_bound = reconstructed.copy()
        upper_bound = reconstructed.copy()

        for region_id in np.unique(segmentation):
            mask = (segmentation == region_id)
            if np.sum(mask) > 0:
                confidence = confidence_levels.get(region_id, 0.90)
                z_score = z_scores.get(confidence, 1.645)

                margin = z_score * uncertainty_map[mask]
                lower_bound[mask] = reconstructed[mask] - margin
                upper_bound[mask] = reconstructed[mask] + margin

        lower_bound = np.clip(lower_bound, 0, 1)
        upper_bound = np.clip(upper_bound, 0, 1)

        # 5. Évaluation du risque
        risk_thresholds = {0: 0.15, 1: 0.10, 2: 0.05, 3: 0.02}
        region_names = {0: 'Background', 1: 'Tissu', 2: 'Organe', 3: 'Pathologie'}

        risk_assessment = {}
        for region_id in np.unique(segmentation):
            mask = (segmentation == region_id)
            if np.sum(mask) > 0:
                avg_uncertainty = np.mean(uncertainty_map[mask])
                threshold = risk_thresholds.get(region_id, 0.05)

                if avg_uncertainty > threshold:
                    risk_level = "ÉLEVÉ"
                elif avg_uncertainty > threshold * 0.7:
                    risk_level = "MODÉRÉ"
                else:
                    risk_level = "ACCEPTABLE"

                risk_assessment[region_id] = {
                    'region_name': region_names.get(region_id, f'Région {region_id}'),
                    'risk_level': risk_level,
                    'uncertainty': avg_uncertainty,
                    'threshold': threshold
                }

        # Métriques
        if ground_truth is not None:
            mse_backward = np.mean((reconstructed - ground_truth)**2)
            psnr_backward = 20 * np.log10(1.0 / np.sqrt(mse_backward))
            print(f"  Reconstruction - MSE: {mse_backward:.6f}, PSNR: {psnr_backward:.2f} dB")
            metrics = {'mse': mse_backward, 'psnr': psnr_backward}
        else:
            metrics = {'mse': None, 'psnr': None}

        return {
            'reconstructed': reconstructed,
            'segmentation': segmentation,
            'uncertainty_map': uncertainty_map,
            'confidence_intervals': (lower_bound, upper_bound),
            'lambdas': lambdas,
            'risk_assessment': risk_assessment,
            'metrics': metrics
        }

    def run_complete_pipeline(self):
        """Exécute votre pipeline complet original."""
        print("="*60)
        print("PIPELINE COMPLET: FORWARD → BACKWARD AVEC CONTRÔLE DE RISQUE")
        print("="*60)

        # Génération
        print("\n1. GÉNÉRATION IMAGE MÉDICALE ORIGINALE")
        clean_image = self.create_medical_image()
        print(f"   Image générée: {clean_image.shape}, range: [{clean_image.min():.3f}, {clean_image.max():.3f}]")

        # Forward
        print("\n2. FORWARD PROPAGATION")
        observations, forward_metrics = self.forward_propagation(clean_image)

        # Backward
        print("\n3. BACKWARD PROPAGATION")
        backward_results = self.backward_propagation(observations, ground_truth=clean_image)

        # Analyse
        print("\n4. ANALYSE COMPARATIVE")
        forward_psnr = forward_metrics['psnr']
        backward_psnr = backward_results['metrics']['psnr']
        improvement = (forward_metrics['mse'] - backward_results['metrics']['mse']) / forward_metrics['mse'] * 100

        print(f"   Forward (dégradation):  PSNR = {forward_psnr:.2f} dB")
        print(f"   Backward (reconstruction): PSNR = {backward_psnr:.2f} dB")
        print(f"   Amélioration: {improvement:+.1f}%")

        # Risques
        print("\n5. ÉVALUATION DES RISQUES PAR RÉGION")
        high_risk_count = 0
        for region_id, risk_info in backward_results['risk_assessment'].items():
            print(f"     {risk_info['region_name']:12}: {risk_info['risk_level']:10} "
                  f"(incertitude: {risk_info['uncertainty']:.4f})")
            if risk_info['risk_level'] == 'ÉLEVÉ':
                high_risk_count += 1

        # Visualisation
        self.create_visualization(clean_image, observations, backward_results, forward_metrics, improvement)

        return {
            'clean_image': clean_image,
            'observations': observations,
            'results': backward_results,
            'forward_metrics': forward_metrics,
            'improvement': improvement
        }

    def create_visualization(self, clean_image, observations, results, forward_metrics, improvement):
        """Crée la visualisation comme dans votre code original."""
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))

        # Ligne 1
        axes[0,0].imshow(clean_image, cmap='gray', vmin=0, vmax=1)
        axes[0,0].set_title('1. Image Originale\n(Vérité terrain)')
        axes[0,0].axis('off')

        axes[0,1].imshow(observations, cmap='gray', vmin=0, vmax=1)
        axes[0,1].set_title(f'2. Forward: Observations\n(PSNR: {forward_metrics["psnr"]:.1f} dB)')
        axes[0,1].axis('off')

        axes[0,2].imshow(results['reconstructed'], cmap='gray', vmin=0, vmax=1)
        axes[0,2].set_title(f'3. Backward: Reconstruction\n(PSNR: {results["metrics"]["psnr"]:.1f} dB)')
        axes[0,2].axis('off')

        axes[0,3].imshow(results['segmentation'], cmap='tab10')
        axes[0,3].set_title('4. Segmentation\nAutomatique')
        axes[0,3].axis('off')

        axes[0,4].imshow(results['uncertainty_map'], cmap='hot')
        axes[0,4].set_title('5. Incertitude\nSémantique')
        axes[0,4].axis('off')

        # Ligne 2
        error_forward = np.abs(observations - clean_image)
        axes[1,0].imshow(error_forward, cmap='hot')
        axes[1,0].set_title(f'Erreur Forward\n(MSE: {forward_metrics["mse"]:.4f})')
        axes[1,0].axis('off')

        error_backward = np.abs(results['reconstructed'] - clean_image)
        axes[1,1].imshow(error_backward, cmap='hot')
        axes[1,1].set_title(f'Erreur Backward\n(MSE: {results["metrics"]["mse"]:.4f})')
        axes[1,1].axis('off')

        lower, upper = results['confidence_intervals']
        interval_width = upper - lower
        axes[1,2].imshow(interval_width, cmap='viridis')
        axes[1,2].set_title('Largeur Intervalles\n(Confiance adaptative)')
        axes[1,2].axis('off')

        # Carte de risque
        risk_map = np.zeros_like(results['uncertainty_map'])
        colors = {'ACCEPTABLE': 1, 'MODÉRÉ': 2, 'ÉLEVÉ': 3}
        for region_id, risk_info in results['risk_assessment'].items():
            mask = (results['segmentation'] == region_id)
            risk_map[mask] = colors.get(risk_info['risk_level'], 1)

        high_risk_count = sum(1 for risk in results['risk_assessment'].values() 
                             if risk['risk_level'] == 'ÉLEVÉ')
        axes[1,3].imshow(risk_map, cmap='RdYlGn_r', vmin=1, vmax=3)
        axes[1,3].set_title(f'Carte de Risque\n({high_risk_count} région(s) critique(s))')
        axes[1,3].axis('off')

        improvement_map = error_forward - error_backward
        axes[1,4].imshow(improvement_map, cmap='RdBu', vmin=-0.2, vmax=0.2)
        axes[1,4].set_title('Amélioration\n(Bleu = Mieux)')
        axes[1,4].axis('off')

        plt.suptitle(f'Pipeline Complet: Forward → Backward (Amélioration {improvement:+.1f}%)', fontsize=16)
        plt.tight_layout()
        plt.show()
