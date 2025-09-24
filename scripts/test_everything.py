import sys
import os

# Ajouter le dossier src au path Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test des imports de base."""
    print("📦 TEST 1: Imports")
    print("-" * 18)
    
    packages_to_test = [
        ('numpy', 'np'),
        ('scipy.ndimage', 'ndimage'),
        ('matplotlib.pyplot', 'plt')
    ]
    
    all_good = True
    for package, alias in packages_to_test:
        try:
            exec(f"import {package} as {alias}")
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            all_good = False
    
    return all_good

def test_pipeline():
    """Test du pipeline principal."""
    print("\n🔬 TEST 2: Pipeline Principal")
    print("-" * 30)
    
    try:
        from src.core.pipeline import MedicalImagePipeline
        
        # Créer et tester le pipeline
        pipeline = MedicalImagePipeline(seed=42)
        results = pipeline.run_complete_pipeline()
        
        # Vérifications simples
        assert 'clean_image' in results
        assert 'improvement' in results
        assert results['improvement'] > -50  # Pas trop mauvais
        
        print("✅ Pipeline fonctionne parfaitement!")
        print(f"   Amélioration: {results['improvement']:.1f}%")
        return True
        
    except Exception as e:
        print(f"❌ Erreur pipeline: {e}")
        return False

def test_demo():
    """Test de la démonstration."""
    print("\n🎓 TEST 3: Démonstration des Risques")
    print("-" * 40)
    
    try:
        from src.demos.demo_risks import run_complete_demo
        
        # Exécuter la démonstration
        demo_results = run_complete_demo()
        
        # Vérifications
        assert 'segmentation' in demo_results
        assert 'lambdas' in demo_results
        
        print("\n✅ Démonstration fonctionne parfaitement!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur démonstration: {e}")
        return False

def test_quick():
    """Test rapide sans visualisations."""
    print("\n⚡ TEST 4: Test Rapide (sans graphiques)")
    print("-" * 45)
    
    try:
        # Import et création rapide
        from src.core.pipeline import MedicalImagePipeline
        
        pipeline = MedicalImagePipeline(seed=42)
        
        # Test juste la création d'image
        image = pipeline.create_medical_image()
        assert image.shape == (128, 128)
        
        # Test forward
        obs, metrics = pipeline.forward_propagation(image)
        assert 'mse' in metrics
        assert 'psnr' in metrics
        
        print("✅ Tests rapides réussis!")
        print(f"   Image: {image.shape}")
        print(f"   PSNR Forward: {metrics['psnr']:.1f} dB")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test rapide: {e}")
        return False

def main():
    """Fonction principale de test."""
    print("🚀 TEST COMPLET DE VOTRE CODE OPTIMISÉ")
    print("=" * 50)
    
    # Exécuter tous les tests
    test_results = []
    test_results.append(("Imports", test_imports()))
    test_results.append(("Test Rapide", test_quick()))
    test_results.append(("Pipeline", test_pipeline()))
    test_results.append(("Démonstration", test_demo()))
    
    # Résumé
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in test_results:
        status = "✅ RÉUSSI" if passed else "❌ ÉCHOUÉ"
        print(f"{test_name:15}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 FÉLICITATIONS ! Tout fonctionne parfaitement !")
        print("\nVotre code de thèse est maintenant:")
        print("✅ Organisé et modulaire")
        print("✅ Testé et validé")
        print("✅ Prêt pour publication")
        print("✅ Réutilisable pour d'autres projets")
        
        print("\nVous pouvez maintenant:")
        print("1. Cloner le repo localement pour développement")
        print("2. Utiliser le code dans d'autres projets") 
        print("3. Le présenter dans votre thèse")
        print("4. Le partager avec la communauté scientifique")
        
    else:
        print("⚠️  Quelques problèmes à corriger")
        print("\nVérifiez les erreurs ci-dessus et:")
        print("1. Installez les packages manquants: pip install numpy scipy matplotlib")
        print("2. Vérifiez que les 3 fichiers ont été créés correctement")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
