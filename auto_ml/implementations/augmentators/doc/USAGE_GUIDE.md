# Guía de Uso: Augmentadores en el Proyecto

Esta guía te muestra cómo usar los augmentadores de datos en tu proyecto AutoML, con ejemplos prácticos y diferentes combinaciones para distintos escenarios.

---

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Uso Básico](#uso-básico)
3. [Combinaciones Comunes](#combinaciones-comunes)
4. [Augmentadores Composite](#augmentadores-composite)
5. [Integración con AutoML](#integración-con-automl)
6. [Ejemplos por Escenario](#ejemplos-por-escenario)
7. [Mejores Prácticas](#mejores-prácticas)

---

## Introducción

El sistema de augmentación permite expandir tu dataset aplicando transformaciones a las imágenes y máscaras. Todos los augmentadores implementan la interfaz `DataAugmentatorInterface` y tienen un método `augment()` que recibe un `DatasetInterface` y retorna un nuevo dataset augmentado.

### Importar Augmentadores

```python
from auto_ml.implementations import (
    # Geometric
    RotationAugmentator,
    HorizontalFlipAugmentator,
    VerticalFlipAugmentator,
    ScaleAugmentator,
    TranslationAugmentator,
    RandomCropAugmentator,
    
    # Photometric
    BrightnessAugmentator,
    ContrastAugmentator,
    GaussianNoiseAugmentator,
    GaussianBlurAugmentator,
    GammaAugmentator,
    
    # SEM-specific
    ElasticDeformationAugmentator,
    AdaptiveHistogramEqualizationAugmentator,
    ChargingArtifactAugmentator,
    ScanLineNoiseAugmentator,
    
    # Composite
    SequentialAugmentator,
    RandomApplyAugmentator,
    RandomChoiceAugmentator,
    OneOfAugmentator,
    MultiplyDatasetAugmentator,
    
    # Identity (sin cambios)
    IdentityAugmentator,
)
```

---

## Uso Básico

### Ejemplo 1: Augmentador Simple

```python
from auto_ml.implementations import (
    RotationAugmentator,
    load_dataset_from_directories,
)
from pathlib import Path

# Cargar dataset
input_dir = Path("data/input")
target_dir = Path("data/target")
dataset = load_dataset_from_directories(input_dir, target_dir)

# Crear augmentador
augmentator = RotationAugmentator(
    angle_range=(-30, 30),  # Rotar entre -30 y +30 grados
    random_seed=42,  # Para reproducibilidad
)

# Aplicar augmentación
augmented_dataset = augmentator.augment(dataset)

print(f"Dataset original: {len(dataset)} muestras")
print(f"Dataset augmentado: {len(augmented_dataset)} muestras")
```

### Ejemplo 2: Sin Augmentación (Identity)

```python
from auto_ml.implementations import IdentityAugmentator

# No aplica ninguna transformación
augmentator = IdentityAugmentator()
augmented_dataset = augmentator.augment(dataset)

# El dataset mantiene las mismas muestras
assert len(augmented_dataset) == len(dataset)
```

---

## Combinaciones Comunes

### Ejemplo 3: Pipeline Secuencial Simple

Aplica múltiples augmentaciones en orden:

```python
from auto_ml.implementations import (
    SequentialAugmentator,
    RotationAugmentator,
    BrightnessAugmentator,
)

# Crear pipeline: primero rota, luego ajusta brillo
pipeline = SequentialAugmentator([
    RotationAugmentator(angle_range=(-15, 15), random_seed=42),
    BrightnessAugmentator(brightness_range=(0.8, 1.2), random_seed=42),
])

augmented_dataset = pipeline.augment(dataset)
```

### Ejemplo 4: Augmentación Geométrica Completa

```python
from auto_ml.implementations import (
    SequentialAugmentator,
    RotationAugmentator,
    HorizontalFlipAugmentator,
    ScaleAugmentator,
)

# Combinar rotación, flip y escala
geometric_pipeline = SequentialAugmentator([
    RotationAugmentator(angle_range=(-20, 20), random_seed=42),
    HorizontalFlipAugmentator(),
    ScaleAugmentator(scale_range=(0.9, 1.1), random_seed=42),
])

augmented_dataset = geometric_pipeline.augment(dataset)
```

### Ejemplo 5: Augmentación Fotométrica

```python
from auto_ml.implementations import (
    SequentialAugmentator,
    BrightnessAugmentator,
    ContrastAugmentator,
    GaussianNoiseAugmentator,
)

# Pipeline para mejorar variabilidad fotométrica
photometric_pipeline = SequentialAugmentator([
    BrightnessAugmentator(brightness_range=(0.85, 1.15), random_seed=42),
    ContrastAugmentator(contrast_range=(0.8, 1.2), random_seed=42),
    GaussianNoiseAugmentator(std=0.01, random_seed=42),
])

augmented_dataset = photometric_pipeline.augment(dataset)
```

### Ejemplo 6: Augmentación SEM Completa

Para imágenes de microscopía electrónica de barrido:

```python
from auto_ml.implementations import (
    SequentialAugmentator,
    ElasticDeformationAugmentator,
    AdaptiveHistogramEqualizationAugmentator,
    ChargingArtifactAugmentator,
    ScanLineNoiseAugmentator,
)

# Pipeline específico para SEM
sem_pipeline = SequentialAugmentator([
    ElasticDeformationAugmentator(alpha=50.0, sigma=5.0, random_seed=42),
    AdaptiveHistogramEqualizationAugmentator(clip_limit=2.0, tile_grid_size=(8, 8)),
    ChargingArtifactAugmentator(num_spots=(1, 3), intensity_range=(0.7, 1.3), random_seed=42),
    ScanLineNoiseAugmentator(probability=0.2, intensity_range=(0.02, 0.05), random_seed=42),
])

augmented_dataset = sem_pipeline.augment(dataset)
```

---

## Augmentadores Composite

### Ejemplo 7: Aplicación Aleatoria (RandomApplyAugmentator)

Aplica una augmentación con cierta probabilidad:

```python
from auto_ml.implementations import (
    RandomApplyAugmentator,
    GaussianBlurAugmentator,
)

# Aplica blur solo al 50% de las muestras
random_blur = RandomApplyAugmentator(
    augmentator=GaussianBlurAugmentator(sigma_range=(0.5, 1.5), random_seed=42),
    probability=0.5,
    random_seed=42,
)

augmented_dataset = random_blur.augment(dataset)
```

### Ejemplo 8: Elegir Una Augmentación (OneOfAugmentator)

Aplica solo UNA de varias augmentaciones posibles:

```python
from auto_ml.implementations import (
    OneOfAugmentator,
    RotationAugmentator,
    HorizontalFlipAugmentator,
    VerticalFlipAugmentator,
)

# Cada muestra recibe solo UNA de estas transformaciones
one_of = OneOfAugmentator(
    augmentators=[
        RotationAugmentator(angle_range=(-30, 30), random_seed=42),
        HorizontalFlipAugmentator(),
        VerticalFlipAugmentator(),
    ],
    random_seed=42,
)

augmented_dataset = one_of.augment(dataset)
```

### Ejemplo 9: Selección Aleatoria (RandomChoiceAugmentator)

Aplica N augmentaciones aleatorias de una lista:

```python
from auto_ml.implementations import (
    RandomChoiceAugmentator,
    BrightnessAugmentator,
    ContrastAugmentator,
    GaussianNoiseAugmentator,
    GaussianBlurAugmentator,
)

# Aplica 2 augmentaciones aleatorias de la lista
random_choice = RandomChoiceAugmentator(
    augmentators=[
        BrightnessAugmentator(brightness_range=(0.8, 1.2), random_seed=42),
        ContrastAugmentator(contrast_range=(0.8, 1.2), random_seed=42),
        GaussianNoiseAugmentator(std=0.02, random_seed=42),
        GaussianBlurAugmentator(sigma_range=(0.5, 1.0), random_seed=42),
    ],
    num_choices=2,
    random_seed=42,
)

augmented_dataset = random_choice.augment(dataset)
```

### Ejemplo 10: Multiplicar Dataset

Crea múltiples versiones augmentadas del dataset:

```python
from auto_ml.implementations import (
    MultiplyDatasetAugmentator,
    RotationAugmentator,
)

# Crea 5 versiones augmentadas de cada muestra
multiplier = MultiplyDatasetAugmentator(
    augmentator=RotationAugmentator(angle_range=(-30, 30), random_seed=42),
    num_copies=5,
)

augmented_dataset = multiplier.augment(dataset)

# Si el dataset original tenía 100 muestras, ahora tiene 500
print(f"Dataset multiplicado: {len(augmented_dataset)} muestras")
```

---

## Integración con AutoML

### Ejemplo 11: Uso con DataAugmentatorNode

```python
from auto_ml.implementations import (
    DataAugmentatorNode,
    SequentialAugmentator,
    RotationAugmentator,
    BrightnessAugmentator,
    load_dataset_from_directories,
)
from auto_ml.automl import AutoML
from pathlib import Path

# Cargar dataset
dataset = load_dataset_from_directories(
    Path("data/input"),
    Path("data/target"),
)

# Crear augmentador
augmentator = SequentialAugmentator([
    RotationAugmentator(angle_range=(-15, 15), random_seed=42),
    BrightnessAugmentator(brightness_range=(0.9, 1.1), random_seed=42),
])

# Crear nodo con k-fold cross-validation
aug_node = DataAugmentatorNode(
    augmentator=augmentator,
    name="GeometricPhotometric",
    k_folds=5,
    random_seed=42,
)

# Usar en AutoML
augmentators = [aug_node]
# ... definir models y evaluator_node ...
# automl = AutoML()
# automl.run_experiment(dataset, augmentators, models, evaluator_node)
```

### Ejemplo 12: Múltiples Estrategias de Augmentación

```python
from auto_ml.implementations import (
    DataAugmentatorNode,
    IdentityAugmentator,
    SequentialAugmentator,
    RotationAugmentator,
    BrightnessAugmentator,
    ElasticDeformationAugmentator,
)

# Estrategia 1: Sin augmentación (baseline)
aug_node_identity = DataAugmentatorNode(
    augmentator=IdentityAugmentator(),
    name="Baseline_NoAug",
    k_folds=5,
    random_seed=42,
)

# Estrategia 2: Augmentación ligera
aug_node_light = DataAugmentatorNode(
    augmentator=SequentialAugmentator([
        RotationAugmentator(angle_range=(-10, 10), random_seed=42),
        BrightnessAugmentator(brightness_range=(0.95, 1.05), random_seed=42),
    ]),
    name="LightAugmentation",
    k_folds=5,
    random_seed=42,
)

# Estrategia 3: Augmentación agresiva
aug_node_aggressive = DataAugmentatorNode(
    augmentator=SequentialAugmentator([
        RotationAugmentator(angle_range=(-45, 45), random_seed=42),
        BrightnessAugmentator(brightness_range=(0.7, 1.3), random_seed=42),
        ElasticDeformationAugmentator(alpha=70.0, sigma=7.0, random_seed=42),
    ]),
    name="AggressiveAugmentation",
    k_folds=5,
    random_seed=42,
)

# Comparar las tres estrategias
augmentators = [aug_node_identity, aug_node_light, aug_node_aggressive]
```

---

## Ejemplos por Escenario

### Escenario 1: Dataset Pequeño (< 100 muestras)

Maximiza la variabilidad con augmentación agresiva:

```python
from auto_ml.implementations import (
    MultiplyDatasetAugmentator,
    SequentialAugmentator,
    RotationAugmentator,
    HorizontalFlipAugmentator,
    VerticalFlipAugmentator,
    BrightnessAugmentator,
    ContrastAugmentator,
    GaussianNoiseAugmentator,
    ElasticDeformationAugmentator,
)

# Pipeline agresivo para expandir dataset pequeño
small_dataset_pipeline = MultiplyDatasetAugmentator(
    augmentator=SequentialAugmentator([
        RotationAugmentator(angle_range=(-45, 45), random_seed=42),
        OneOfAugmentator([
            HorizontalFlipAugmentator(),
            VerticalFlipAugmentator(),
        ], random_seed=42),
        BrightnessAugmentator(brightness_range=(0.7, 1.3), random_seed=42),
        ContrastAugmentator(contrast_range=(0.7, 1.3), random_seed=42),
        GaussianNoiseAugmentator(std=0.03, random_seed=42),
        ElasticDeformationAugmentator(alpha=60.0, sigma=6.0, random_seed=42),
    ]),
    num_copies=10,  # Multiplica por 10
)

augmented_dataset = small_dataset_pipeline.augment(dataset)
```

### Escenario 2: Dataset Balanceado (Entrenamiento General)

Augmentación moderada para mejorar generalización:

```python
from auto_ml.implementations import (
    SequentialAugmentator,
    RandomApplyAugmentator,
    RotationAugmentator,
    HorizontalFlipAugmentator,
    BrightnessAugmentator,
    GaussianNoiseAugmentator,
)

# Pipeline balanceado
balanced_pipeline = SequentialAugmentator([
    RotationAugmentator(angle_range=(-20, 20), random_seed=42),
    HorizontalFlipAugmentator(),
    RandomApplyAugmentator(
        augmentator=BrightnessAugmentator(brightness_range=(0.85, 1.15), random_seed=42),
        probability=0.5,
        random_seed=42,
    ),
    RandomApplyAugmentator(
        augmentator=GaussianNoiseAugmentator(std=0.015, random_seed=42),
        probability=0.3,
        random_seed=42,
    ),
])

augmented_dataset = balanced_pipeline.augment(dataset)
```

### Escenario 3: Imágenes SEM de Rocas

Augmentación específica para microscopía:

```python
from auto_ml.implementations import (
    SequentialAugmentator,
    RandomApplyAugmentator,
    RotationAugmentator,
    ElasticDeformationAugmentator,
    AdaptiveHistogramEqualizationAugmentator,
    ChargingArtifactAugmentator,
    ScanLineNoiseAugmentator,
)

# Pipeline especializado para SEM
sem_rocks_pipeline = SequentialAugmentator([
    # Transformaciones geométricas sutiles
    RotationAugmentator(angle_range=(-15, 15), random_seed=42),
    
    # Deformación elástica (común en muestras naturales)
    ElasticDeformationAugmentator(alpha=40.0, sigma=5.0, random_seed=42),
    
    # Mejora de contraste local (siempre útil)
    AdaptiveHistogramEqualizationAugmentator(clip_limit=2.0, tile_grid_size=(8, 8)),
    
    # Artefactos de carga (aleatorio, 30% probabilidad)
    RandomApplyAugmentator(
        augmentator=ChargingArtifactAugmentator(
            num_spots=(1, 3),
            intensity_range=(0.7, 1.2),
            random_seed=42,
        ),
        probability=0.3,
        random_seed=42,
    ),
    
    # Ruido de línea de escaneo (aleatorio, 20% probabilidad)
    RandomApplyAugmentator(
        augmentator=ScanLineNoiseAugmentator(
            probability=0.3,
            intensity_range=(0.02, 0.05),
            random_seed=42,
        ),
        probability=0.2,
        random_seed=42,
    ),
])

augmented_dataset = sem_rocks_pipeline.augment(dataset)
```

### Escenario 4: Testing y Validación

Sin augmentación para evaluación justa:

```python
from auto_ml.implementations import IdentityAugmentator

# Sin cambios para test/validation
test_augmentator = IdentityAugmentator()
test_dataset = test_augmentator.augment(dataset)
```

---

## Mejores Prácticas

### 1. Combinar Augmentaciones Complementarias

Mezcla geométricas con fotométricas:

```python
# Buena combinación: geométrica + fotométrica
good_combo = SequentialAugmentator([
    RotationAugmentator(angle_range=(-20, 20), random_seed=42),  # Geométrica
    BrightnessAugmentator(brightness_range=(0.85, 1.15), random_seed=42),  # Fotométrica
])

# Evitar: múltiples augmentaciones del mismo tipo
# (puede ser redundante o excesivo)
redundant = SequentialAugmentator([
    BrightnessAugmentator(brightness_range=(0.8, 1.2), random_seed=42),
    ContrastAugmentator(contrast_range=(0.8, 1.2), random_seed=42),
    GammaAugmentator(gamma_range=(0.8, 1.2), random_seed=42),
    # Todas fotométricas, puede ser demasiado
])
```

### 2. Usar Composite Augmentators para Variabilidad

```python
# En lugar de aplicar siempre la misma augmentación:
always_same = RotationAugmentator(angle_range=(-20, 20), random_seed=42)

# Mejor: variar con OneOfAugmentator
varied = OneOfAugmentator([
    RotationAugmentator(angle_range=(-20, 20), random_seed=42),
    HorizontalFlipAugmentator(),
    ScaleAugmentator(scale_range=(0.9, 1.1), random_seed=42),
], random_seed=42)
```

---

## Recursos Adicionales

- **Documentación de API**: Ver `AUGMENTATORS.md` para detalles de cada función

---