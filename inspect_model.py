from models.deepfake_detector import DeepfakeDetector


print("=" * 60)
print("ORIGINX AI - MODEL ARCHITECTURE")
print("=" * 60)

detector = DeepfakeDetector()

print()
print(detector.model)
print()
print("=" * 60)
print("MODEL MODULES")
print("=" * 60)

for name, module in detector.model.named_modules():
    print(name, "->", module.__class__.__name__)

print()
print("=" * 60)
print("DONE")
print("=" * 60)