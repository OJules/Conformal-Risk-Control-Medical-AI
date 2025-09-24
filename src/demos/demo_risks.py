import numpy as np
import matplotlib.pyplot as plt

def demo_risk_evaluation():
    """Votre fonction originale - démonstration des risques."""
    print("="*60)
    print("DÉMONSTRATION : ÉVALUATION DES RISQUES ÉTAPE PAR ÉTAPE")
    print("="*60)

    np.random.seed(42)

    # Simulation d'une image reconstruite (128x128)
    reconstructed = np.random.rand(128, 128)

    # Simulation d'erreurs par région
    ground_truth = reconstructed + np.random.normal(0, 0.1, (128, 128))
    errors = np.abs(reconstructed - ground_truth)

    print("\n1. SEGMENTATION AUTOMATIQUE")
    print("-" * 30)

    # Segmentation basée sur l'intensité
    segmentation = np.zeros_like(reconstructed, dtype=int)
    segmentation[reconstructed <= 0.2] = 0  # Background
    segmentation[(reconstructed > 0.2) & (reconstructed <= 0.5)] = 1  # Tissu
    segmentation[(reconstructed > 0.5) & (reconstructed <= 0.8)] = 2  # Organe
    segmentation[reconstructed > 0.8] = 3  # Pathologie

    region_names = {0: 'Background', 1: 'Tissu', 2: 'Organe', 3: 'Pathologie'}

    for region_id in [0, 1, 2, 3]:
        count = np.sum(segmentation == region_id)
        percentage = (count / segmentation.size) * 100
        print(f"   {region_names[region_id]:12}: {count:5d} pixels ({percentage:5.1f}%)")

    print("\n2. CALCUL DES FACTEURS D'IMPORTANCE CLINIQUE")
    print("-" * 50)

    clinical_factors = {0: 1.0, 1: 1.2, 2: 1.8, 3: 3.0}

    print("   Région        | Facteur λ | Justification")
    print("   --------------|-----------|-----------------------------------")
    for region_id in [0, 1, 2, 3]:
        name = region_names[region_id]
        factor = clinical_factors[region_id]

        if region_id == 0:
            justification = "Impact minimal sur diagnostic"
        elif region_id == 1:
            justification = "Structure de référence"
        elif region_id == 2:
            justification = "Anatomie importante"
        else:
            justification = "CRITIQUE - Impact vital"

        print(f"   {name:12} | {factor:8.1f} | {justification}")

    print("\n3. QUANTIFICATION D'INCERTITUDE PAR RÉGION")
    print("-" * 45)

    uncertainty_map = np.zeros_like(reconstructed)
    lambdas = {}

    print("   Région        | Erreur Moy. | Facteur λ | Incertitude")
    print("   --------------|-------------|-----------|------------")

    for region_id in [0, 1, 2, 3]:
        mask = (segmentation == region_id)
        if np.sum(mask) > 0:
            regional_error = np.mean(errors[mask])
            factor = clinical_factors[region_id]
            lambda_val = regional_error * factor

            lambdas[region_id] = lambda_val
            uncertainty_map[mask] = lambda_val

            name = region_names[region_id]
            print(f"   {name:12} | {regional_error:10.4f} | {factor:8.1f} | {lambda_val:10.4f}")

    print("\n4. INTERVALLES DE CONFIANCE ADAPTATIFS")
    print("-" * 42)

    confidence_levels = {0: 0.80, 1: 0.85, 2: 0.90, 3: 0.95}
    z_scores = {0.80: 1.28, 0.85: 1.44, 0.90: 1.645, 0.95: 1.96}

    print("   Région        | Confiance | Z-Score | Marge d'Erreur")
    print("   --------------|-----------|---------|----------------")

    for region_id in [0, 1, 2, 3]:
        if region_id in lambdas:
            confidence = confidence_levels[region_id]
            z_score = z_scores[confidence]
            margin = z_score * lambdas[region_id]

            name = region_names[region_id]
            print(f"   {name:12} | {confidence:8.0%} | {z_score:6.3f} | ±{margin:10.4f}")

    print("\n5. CLASSIFICATION DU RISQUE CLINIQUE")
    print("-" * 38)

    risk_thresholds = {0: 0.15, 1: 0.10, 2: 0.05, 3: 0.02}

    print("   Région        | Incertitude | Seuil   | Classification")
    print("   --------------|-------------|---------|----------------")

    high_risk_regions = []

    for region_id in [0, 1, 2, 3]:
        if region_id in lambdas:
            avg_uncertainty = lambdas[region_id]
            threshold = risk_thresholds[region_id]

            if avg_uncertainty > threshold:
                risk_level = "🔴 ÉLEVÉ"
                high_risk_regions.append(region_names[region_id])
            elif avg_uncertainty > threshold * 0.7:
                risk_level = "🟡 MODÉRÉ"
            else:
                risk_level = "🟢 ACCEPTABLE"

            name = region_names[region_id]
            print(f"   {name:12} | {avg_uncertainty:10.4f} | {threshold:6.2f} | {risk_level}")

    print("\n6. RECOMMANDATIONS CLINIQUES")
    print("-" * 32)

    if len(high_risk_regions) == 0:
        print("   ✅ EXCELLENT : Toutes les régions sont dans les seuils acceptables")
        print("   → Validation automatique possible")
    elif len(high_risk_regions) == 1:
        print(f"   ⚠️  ATTENTION : 1 région nécessite contrôle humain")
        print(f"   → Région critique : {high_risk_regions[0]}")
        print("   → Recommandation : Double lecture par radiologue")
    else:
        print(f"   🚨 CRITIQUE : {len(high_risk_regions)} régions nécessitent contrôle humain")
        print(f"   → Régions critiques : {', '.join(high_risk_regions)}")
        print("   → Recommandation : Nouvelle acquisition ou reconstruction avancée")

    return {
        'segmentation': segmentation,
        'uncertainty_map': uncertainty_map,
        'lambdas': lambdas,
        'high_risk_regions': high_risk_regions
    }

def compare_forward_vs_backward():
    """Votre fonction originale - comparaison Forward vs Backward."""
    print("\n" + "="*60)
    print("COMPARAISON : FORWARD vs BACKWARD PROPAGATION")
    print("="*60)

    print("\n📉 FORWARD PROPAGATION - Simulation de Dégradation")
    print("-" * 55)
    print("   Objectif     : Simuler l'acquisition CT réelle avec limitations")
    print("   Direction    : Image Parfaite → Observations Dégradées")
    print("   Processus    :")
    print("     1. Sous-échantillonnage fréquentiel (30% des fréquences)")
    print("     2. Ajout de bruit quantique (réduction de dose)")
    print("     3. Application de flou (limitations détecteur)")
    print("     4. Normalisation physique [0,1]")
    print("   Métriques    : MSE_forward, PSNR_forward")
    print("   Utilisation  : Évaluer la robustesse des algorithmes")

    print("\n📈 BACKWARD PROPAGATION - Reconstruction + Risque")
    print("-" * 52)
    print("   Objectif     : Reconstruire ET quantifier l'incertitude")
    print("   Direction    : Observations Dégradées → Image + Carte de Risque")
    print("   Processus    :")
    print("     1. Débruitage (filtre médian + gaussien)")
    print("     2. Segmentation anatomique automatique")
    print("     3. Quantification d'incertitude sémantique")
    print("     4. Calcul d'intervalles de confiance adaptatifs")
    print("     5. Classification de risque clinique")
    print("   Métriques    : MSE_backward, PSNR_backward, Cartes de risque")
    print("   Innovation   : Contrôle de risque conforme sémantique (CRC)")

    print("\n🔄 MÉTRIQUE D'AMÉLIORATION")
    print("-" * 26)
    print("   Formule      : (MSE_forward - MSE_backward) / MSE_forward * 100")
    print("   Interprétation :")
    print("     > 15%      : 🎯 Très efficace")
    print("     5-15%      : ✅ Efficace")
    print("     0-5%       : 🔸 Limité")
    print("     < 0%       : ❌ Inefficace")

    print("\n🏥 AVANTAGES CLINIQUES DU SYSTÈME CRC")
    print("-" * 37)
    print("   • Sécurité patient      : Détection auto des régions critiques")
    print("   • Optimisation workflow : Priorisation intelligente")
    print("   • Traçabilité médicale  : Documentation de l'incertitude")
    print("   • Aide au diagnostic    : Quantification objective du risque")

def run_complete_demo():
    """Exécute votre démonstration complète."""
    risk_results = demo_risk_evaluation()
    compare_forward_vs_backward()

    print(f"\n" + "="*60)
    print("RÉSUMÉ FINAL")
    print("="*60)
    print("Le système combine:")
    print("✓ Simulation réaliste d'acquisition dégradée (Forward)")
    print("✓ Reconstruction avec contrôle de risque sémantique (Backward)")
    print("✓ Classification automatique par importance clinique")
    print("✓ Quantification d'incertitude adaptative")
    print("\nCette approche est particulièrement adaptée aux applications")
    print("médicales où les erreurs ont des conséquences variables selon")
    print("l'importance anatomique des structures concernées.")

    return risk_results

if __name__ == "__main__":
    run_complete_demo()
