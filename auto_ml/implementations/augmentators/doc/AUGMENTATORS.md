# Documentación de Augmentators

Este documento describe todas las clases y funciones de aumentación de datos disponibles en el módulo `augmentators`. Estas herramientas permiten expandir y diversificar datasets de imágenes y máscaras para mejorar el entrenamiento de modelos de machine learning.

---

## Tabla de Contenidos

1. [base.py - Funciones Base](#basepy---funciones-base)
2. [identity.py - Aumentación Identidad](#identitypy---aumentación-identidad)
3. [geometric.py - Aumentaciones Geométricas](#geometricpy---aumentaciones-geométricas)
4. [photometric.py - Aumentaciones Fotométricas](#photometricpy---aumentaciones-fotométricas)
5. [sem_specific.py - Aumentaciones Específicas para SEM](#sem_specificpy---aumentaciones-específicas-para-sem)
6. [composite.py - Aumentaciones Compuestas](#compositepy---aumentaciones-compuestas)

---

## base.py - Funciones Base

Este módulo contiene funciones utilitarias para aplicar transformaciones afines a imágenes y máscaras.

### `apply_affine_to_image`

Aplica una transformación afín a una imagen.

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `image` | `ImageArray` | Imagen de entrada (H, W) o (H, W, C) |
| `matrix` | `np.ndarray` | Matriz de transformación afín 2x3 o 3x3 |
| `order` | `int` | Orden de interpolación (0=nearest, 1=bilinear, 3=cubic). Default: 1 |
| `fill_value` | `float` | Valor para píxeles fuera de límites. Default: 0.0 |

**Retorna:** Imagen transformada con la misma forma que la entrada.

**Ejemplo:**
```python
import numpy as np
from auto_ml.implementations.augmentators.base import apply_affine_to_image, rotation_matrix

# Crear imagen de ejemplo
image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

# Crear matriz de rotación (30 grados, centro de la imagen)
center = (128, 128)
matrix = rotation_matrix(30, center)

# Aplicar transformación
rotated_image = apply_affine_to_image(image, matrix, order=1)
```

---

### `apply_affine_to_mask`

Aplica una transformación afín a una máscara usando interpolación de vecino más cercano.

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `mask` | `MaskArray` | Máscara de entrada (H, W) |
| `matrix` | `np.ndarray` | Matriz de transformación afín 2x3 o 3x3 |
| `fill_value` | `int` | Valor para píxeles fuera de límites. Default: 0 |

**Retorna:** Máscara transformada con la misma forma que la entrada.

**Ejemplo:**
```python
import numpy as np
from auto_ml.implementations.augmentators.base import apply_affine_to_mask, rotation_matrix

# Crear máscara de ejemplo
mask = np.zeros((256, 256), dtype=np.uint8)
mask[100:150, 100:150] = 1

# Crear matriz de rotación
center = (128, 128)
matrix = rotation_matrix(45, center)

# Aplicar transformación
rotated_mask = apply_affine_to_mask(mask, matrix)
```

---

### `rotation_matrix`

Crea una matriz de rotación 3x3.

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `angle_degrees` | `float` | Ángulo de rotación en grados (positivo = sentido antihorario) |
| `center` | `Tuple[float, float]` | Centro de rotación (x, y) |

**Retorna:** Matriz de transformación afín 3x3.

**Ejemplo:**
```python
from auto_ml.implementations.augmentators.base import rotation_matrix

# Matriz de rotación de 90 grados alrededor del centro (128, 128)
matrix = rotation_matrix(90, (128, 128))
print(matrix)
```

---

### `scale_matrix`

Crea una matriz de escala 3x3.

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `scale_x` | `float` | Factor de escala en el eje X |
| `scale_y` | `float` | Factor de escala en el eje Y |
| `center` | `Tuple[float, float]` | Centro de escalado (x, y) |

**Retorna:** Matriz de transformación afín 3x3.

**Ejemplo:**
```python
from auto_ml.implementations.augmentators.base import scale_matrix

# Matriz de escala (2x en X, 0.5x en Y) alrededor del centro
matrix = scale_matrix(2.0, 0.5, (128, 128))
```

---

### `translation_matrix`

Crea una matriz de traslación 3x3.

**Parámetros:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `tx` | `float` | Traslación en el eje X (positivo = derecha) |
| `ty` | `float` | Traslación en el eje Y (positivo = abajo) |

**Retorna:** Matriz de transformación afín 3x3.

**Ejemplo:**
```python
from auto_ml.implementations.augmentators.base import translation_matrix

# Matriz de traslación (50 píxeles a la derecha, 30 hacia abajo)
matrix = translation_matrix(50, 30)
```

---

## identity.py - Aumentación Identidad

### `IdentityAugmentator`

Aumentador de identidad que retorna el dataset sin cambios. Útil como línea base o cuando no se desea aumentación.

**Métodos:**
- `augment(dataset: DatasetInterface) -> DatasetInterface`: Retorna el dataset sin modificaciones.

**Ejemplo:**
```python
from auto_ml.implementations.augmentators import IdentityAugmentator
from auto_ml.interfaces import DatasetInterface
import numpy as np

# Crear dataset de ejemplo
images = [np.random.randint(0, 255, (256, 256), dtype=np.uint8)]
masks = [np.zeros((256, 256), dtype=np.uint8)]
dataset = DatasetInterface.from_pairs(list(zip(images, masks)))

# Aplicar aumentación identidad
augmentator = IdentityAugmentator()
result = augmentator.augment(dataset)
# result contiene los mismos datos que dataset
```

---

## geometric.py - Aumentaciones Geométricas

Estas aumentaciones afectan tanto la imagen como la máscara, manteniendo la alineación espacial.

### `RotationAugmentator`

Rota imágenes y máscaras por un ángulo especificado.

**Parámetros del constructor:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `angle_range` | `Tuple[float, float]` | Rango de ángulos en grados (min, max). Default: (-15.0, 15.0) |
| `random_seed` | `int \| None` | Semilla para reproducibilidad. Default: None |

**Ejemplo:**
```python
from auto_ml.implementations.augmentators import RotationAugmentator
from auto_ml.interfaces import DatasetInterface
import numpy as np

# Crear dataset
image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
mask = np.zeros((256, 256), dtype=np.uint8)
mask[100:150, 100:150] = 1
dataset = DatasetInterface.from_pairs([(image, mask)])

# Rotar entre -30 y 30 grados
augmentator = RotationAugmentator(
    angle_range=(-30.0, 30.0),
    random_seed=42,
)
rotated_dataset = augmentator.augment(dataset)
```

---

### `HorizontalFlipAugmentator`

Voltea imágenes y máscaras horizontalmente.

**Ejemplo:**
```python
from auto_ml.implementations.augmentators import HorizontalFlipAugmentator
from auto_ml.interfaces import DatasetInterface
import numpy as np

image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
mask = np.zeros((256, 256), dtype=np.uint8)
dataset = DatasetInterface.from_pairs([(image, mask)])

augmentator = HorizontalFlipAugmentator()
flipped_dataset = augmentator.augment(dataset)
```

---

### `VerticalFlipAugmentator`

Voltea imágenes y máscaras verticalmente.

**Ejemplo:**
```python
from auto_ml.implementations.augmentators import VerticalFlipAugmentator
from auto_ml.interfaces import DatasetInterface
import numpy as np

image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
mask = np.zeros((256, 256), dtype=np.uint8)
dataset = DatasetInterface.from_pairs([(image, mask)])

augmentator = VerticalFlipAugmentator()
flipped_dataset = augmentator.augment(dataset)
```

---

### `ScaleAugmentator`

Escala imágenes y máscaras (zoom in/out) manteniendo el tamaño original.

**Parámetros del constructor:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `scale_range` | `Tuple[float, float]` | Rango de factores de escala. <1.0 = zoom out, >1.0 = zoom in. Default: (0.8, 1.2) |
| `random_seed` | `int \| None` | Semilla para reproducibilidad. Default: None |

**Ejemplo:**
```python
from auto_ml.implementations.augmentators import ScaleAugmentator
from auto_ml.interfaces import DatasetInterface
import numpy as np

image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
mask = np.zeros((256, 256), dtype=np.uint8)
dataset = DatasetInterface.from_pairs([(image, mask)])

# Escalar entre 70% y 130%
augmentator = ScaleAugmentator(
    scale_range=(0.7, 1.3),
    random_seed=42,
)
scaled_dataset = augmentator.augment(dataset)
```

---

### `TranslationAugmentator`

Traslada (desplaza) imágenes y máscaras horizontal y/o verticalmente.

**Parámetros del constructor:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `translate_range` | `Tuple[float, float]` | Rango de traslación como fracción del tamaño. Ej: (-0.1, 0.1) = ±10%. Default: (-0.1, 0.1) |
| `random_seed` | `int \| None` | Semilla para reproducibilidad. Default: None |

**Ejemplo:**
```python
from auto_ml.implementations.augmentators import TranslationAugmentator
from auto_ml.interfaces import DatasetInterface
import numpy as np

image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
mask = np.zeros((256, 256), dtype=np.uint8)
dataset = DatasetInterface.from_pairs([(image, mask)])

# Trasladar hasta ±20% del tamaño de la imagen
augmentator = TranslationAugmentator(
    translate_range=(-0.2, 0.2),
    random_seed=42,
)
translated_dataset = augmentator.augment(dataset)
```

---

### `RandomCropAugmentator`

Recorta una región aleatoria de las imágenes y máscaras, redimensionando al tamaño original.

**Parámetros del constructor:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `crop_size_range` | `Tuple[float, float]` | Rango del tamaño de recorte como fracción. Ej: (0.8, 1.0) = 80-100%. Default: (0.8, 1.0) |
| `random_seed` | `int \| None` | Semilla para reproducibilidad. Default: None |

**Ejemplo:**
```python
from auto_ml.implementations.augmentators import RandomCropAugmentator
from auto_ml.interfaces import DatasetInterface
import numpy as np

image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
mask = np.zeros((256, 256), dtype=np.uint8)
dataset = DatasetInterface.from_pairs([(image, mask)])

# Recortar entre 70% y 90% de la imagen
augmentator = RandomCropAugmentator(
    crop_size_range=(0.7, 0.9),
    random_seed=42,
)
cropped_dataset = augmentator.augment(dataset)
```

---

## photometric.py - Aumentaciones Fotométricas

Estas aumentaciones solo afectan la imagen, dejando la máscara sin modificar.

### `BrightnessAugmentator`

Ajusta el brillo de las imágenes.

**Parámetros del constructor:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `brightness_range` | `tuple[float, float]` | Rango de multiplicadores. 1.0 = sin cambio, <1.0 = oscuro, >1.0 = brillante. Default: (0.8, 1.2) |
| `random_seed` | `int \| None` | Semilla para reproducibilidad. Default: None |

**Ejemplo:**
```python
from auto_ml.implementations.augmentators import BrightnessAugmentator
from auto_ml.interfaces import DatasetInterface
import numpy as np

image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
mask = np.zeros((256, 256), dtype=np.uint8)
dataset = DatasetInterface.from_pairs([(image, mask)])

# Variar brillo entre 60% y 140%
augmentator = BrightnessAugmentator(
    brightness_range=(0.6, 1.4),
    random_seed=42,
)
bright_dataset = augmentator.augment(dataset)
```

---

### `ContrastAugmentator`

Ajusta el contraste de las imágenes.

**Parámetros del constructor:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `contrast_range` | `tuple[float, float]` | Rango de multiplicadores. 1.0 = sin cambio, <1.0 = menos contraste, >1.0 = más contraste. Default: (0.8, 1.2) |
| `random_seed` | `int \| None` | Semilla para reproducibilidad. Default: None |

**Ejemplo:**
```python
from auto_ml.implementations.augmentators import ContrastAugmentator
from auto_ml.interfaces import DatasetInterface
import numpy as np

image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
mask = np.zeros((256, 256), dtype=np.uint8)
dataset = DatasetInterface.from_pairs([(image, mask)])

# Variar contraste entre 70% y 150%
augmentator = ContrastAugmentator(
    contrast_range=(0.7, 1.5),
    random_seed=42,
)
contrast_dataset = augmentator.augment(dataset)
```

---

### `GaussianNoiseAugmentator`

Añade ruido gaussiano a las imágenes.

**Parámetros del constructor:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `noise_std_range` | `tuple[float, float]` | Rango de desviación estándar del ruido. Default: (0.0, 10.0) |
| `random_seed` | `int \| None` | Semilla para reproducibilidad. Default: None |

**Ejemplo:**
```python
from auto_ml.implementations.augmentators import GaussianNoiseAugmentator
from auto_ml.interfaces import DatasetInterface
import numpy as np

image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
mask = np.zeros((256, 256), dtype=np.uint8)
dataset = DatasetInterface.from_pairs([(image, mask)])

# Añadir ruido con std entre 5 y 20
augmentator = GaussianNoiseAugmentator(
    noise_std_range=(5.0, 20.0),
    random_seed=42,
)
noisy_dataset = augmentator.augment(dataset)
```

---

### `GaussianBlurAugmentator`

Aplica desenfoque gaussiano a las imágenes.

**Parámetros del constructor:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `sigma_range` | `tuple[float, float]` | Rango de valores sigma. Mayor = más desenfoque. Default: (0.0, 2.0) |
| `random_seed` | `int \| None` | Semilla para reproducibilidad. Default: None |

**Ejemplo:**
```python
from auto_ml.implementations.augmentators import GaussianBlurAugmentator
from auto_ml.interfaces import DatasetInterface
import numpy as np

image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
mask = np.zeros((256, 256), dtype=np.uint8)
dataset = DatasetInterface.from_pairs([(image, mask)])

# Aplicar desenfoque con sigma entre 0.5 y 3.0
augmentator = GaussianBlurAugmentator(
    sigma_range=(0.5, 3.0),
    random_seed=42,
)
blurred_dataset = augmentator.augment(dataset)
```

---

### `GammaAugmentator`

Aplica corrección gamma a las imágenes.

**Parámetros del constructor:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `gamma_range` | `tuple[float, float]` | Rango de valores gamma. 1.0 = sin cambio, <1.0 = más brillante, >1.0 = más oscuro. Default: (0.8, 1.2) |
| `random_seed` | `int \| None` | Semilla para reproducibilidad. Default: None |

**Ejemplo:**
```python
from auto_ml.implementations.augmentators import GammaAugmentator
from auto_ml.interfaces import DatasetInterface
import numpy as np

image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
mask = np.zeros((256, 256), dtype=np.uint8)
dataset = DatasetInterface.from_pairs([(image, mask)])

# Aplicar corrección gamma entre 0.5 y 1.5
augmentator = GammaAugmentator(
    gamma_range=(0.5, 1.5),
    random_seed=42,
)
gamma_dataset = augmentator.augment(dataset)
```

---

## sem_specific.py - Aumentaciones Específicas para SEM

Aumentaciones diseñadas específicamente para imágenes de microscopía electrónica de barrido (SEM).

### `ElasticDeformationAugmentator`

Aplica deformación elástica para simular variaciones naturales en texturas de roca.

**Parámetros del constructor:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `alpha` | `float` | Fuerza de deformación (mayor = más distorsión). Default: 50.0 |
| `sigma` | `float` | Suavidad del campo de deformación (mayor = más suave). Default: 5.0 |
| `random_seed` | `int \| None` | Semilla para reproducibilidad. Default: None |

**Ejemplo:**
```python
from auto_ml.implementations.augmentators import ElasticDeformationAugmentator
from auto_ml.interfaces import DatasetInterface
import numpy as np

image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
mask = np.zeros((256, 256), dtype=np.uint8)
dataset = DatasetInterface.from_pairs([(image, mask)])

# Aplicar deformación elástica
augmentator = ElasticDeformationAugmentator(
    alpha=100.0,
    sigma=10.0,
    random_seed=42,
)
deformed_dataset = augmentator.augment(dataset)
```

---

### `AdaptiveHistogramEqualizationAugmentator`

Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) para mejorar el contraste local.

**Parámetros del constructor:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `clip_limit` | `float` | Umbral para limitar el contraste (mayor = más contraste). Default: 2.0 |
| `tile_grid_size` | `Tuple[int, int]` | Tamaño de la cuadrícula para ecualización local. Default: (8, 8) |

**Ejemplo:**
```python
from auto_ml.implementations.augmentators import AdaptiveHistogramEqualizationAugmentator
from auto_ml.interfaces import DatasetInterface
import numpy as np

image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
mask = np.zeros((256, 256), dtype=np.uint8)
dataset = DatasetInterface.from_pairs([(image, mask)])

# Aplicar CLAHE
augmentator = AdaptiveHistogramEqualizationAugmentator(
    clip_limit=3.0,
    tile_grid_size=(16, 16),
)
clahe_dataset = augmentator.augment(dataset)
```

---

### `ChargingArtifactAugmentator`

Simula artefactos de carga en imágenes SEM de muestras no conductivas.

**Parámetros del constructor:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `intensity_range` | `Tuple[float, float]` | Rango de multiplicadores de brillo para manchas de carga. Default: (0.7, 1.3) |
| `num_spots` | `Tuple[int, int]` | Rango de número de manchas a añadir. Default: (1, 3) |
| `spot_size_range` | `Tuple[int, int]` | Rango del radio de manchas en píxeles. Default: (30, 80) |
| `random_seed` | `int \| None` | Semilla para reproducibilidad. Default: None |

**Ejemplo:**
```python
from auto_ml.implementations.augmentators import ChargingArtifactAugmentator
from auto_ml.interfaces import DatasetInterface
import numpy as np

image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
mask = np.zeros((256, 256), dtype=np.uint8)
dataset = DatasetInterface.from_pairs([(image, mask)])

# Simular artefactos de carga
augmentator = ChargingArtifactAugmentator(
    intensity_range=(0.5, 1.5),
    num_spots=(2, 5),
    spot_size_range=(20, 60),
    random_seed=42,
)
charging_dataset = augmentator.augment(dataset)
```

---

### `ScanLineNoiseAugmentator`

Añade artefactos de líneas de escaneo típicos de imágenes SEM.

**Parámetros del constructor:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `probability` | `float` | Probabilidad de añadir ruido a cada línea. Default: 0.3 |
| `intensity_range` | `Tuple[float, float]` | Rango de intensidad del ruido (fracción del valor máximo). Default: (0.02, 0.08) |
| `direction` | `str` | Dirección de las líneas ("horizontal" o "vertical"). Default: "horizontal" |
| `random_seed` | `int \| None` | Semilla para reproducibilidad. Default: None |

**Ejemplo:**
```python
from auto_ml.implementations.augmentators import ScanLineNoiseAugmentator
from auto_ml.interfaces import DatasetInterface
import numpy as np

image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
mask = np.zeros((256, 256), dtype=np.uint8)
dataset = DatasetInterface.from_pairs([(image, mask)])

# Añadir ruido de líneas de escaneo
augmentator = ScanLineNoiseAugmentator(
    probability=0.5,
    intensity_range=(0.05, 0.15),
    direction="horizontal",
    random_seed=42,
)
scanline_dataset = augmentator.augment(dataset)
```

---

## composite.py - Aumentaciones Compuestas

Estas clases permiten combinar múltiples aumentaciones de diferentes formas.

### `SequentialAugmentator`

Aplica múltiples aumentaciones secuencialmente. Cada aumentación se aplica sobre el resultado de la anterior.

**Parámetros del constructor:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `augmentators` | `List[DataAugmentatorInterface]` | Lista de aumentadores a aplicar en orden |

**Ejemplo:**
```python
from auto_ml.implementations.augmentators import (
    SequentialAugmentator,
    RotationAugmentator,
    BrightnessAugmentator,
    GaussianNoiseAugmentator,
)
from auto_ml.interfaces import DatasetInterface
import numpy as np

image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
mask = np.zeros((256, 256), dtype=np.uint8)
dataset = DatasetInterface.from_pairs([(image, mask)])

# Aplicar rotación -> brillo -> ruido en secuencia
augmentator = SequentialAugmentator([
    RotationAugmentator(angle_range=(-15, 15)),
    BrightnessAugmentator(brightness_range=(0.9, 1.1)),
    GaussianNoiseAugmentator(noise_std_range=(0, 5)),
])
augmented = augmentator.augment(dataset)
```

---

### `RandomChoiceAugmentator`

Elige aleatoriamente una aumentación de la lista para aplicar a cada muestra.

**Parámetros del constructor:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `augmentators` | `List[DataAugmentatorInterface]` | Lista de aumentadores de donde elegir |
| `random_seed` | `int \| None` | Semilla para reproducibilidad. Default: None |

**Ejemplo:**
```python
from auto_ml.implementations.augmentators import (
    RandomChoiceAugmentator,
    HorizontalFlipAugmentator,
    VerticalFlipAugmentator,
    RotationAugmentator,
)
from auto_ml.interfaces import DatasetInterface
import numpy as np

image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
mask = np.zeros((256, 256), dtype=np.uint8)
dataset = DatasetInterface.from_pairs([(image, mask)])

# Elegir aleatoriamente una aumentación para cada muestra
augmentator = RandomChoiceAugmentator(
    augmentators=[
        HorizontalFlipAugmentator(),
        VerticalFlipAugmentator(),
        RotationAugmentator(angle_range=(-45, 45)),
    ],
    random_seed=42,
)
augmented = augmentator.augment(dataset)
```

---

### `RandomApplyAugmentator`

Aplica una aumentación con una probabilidad dada.

**Parámetros del constructor:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `augmentator` | `DataAugmentatorInterface` | El aumentador a aplicar |
| `probability` | `float` | Probabilidad de aplicar la aumentación (0.0-1.0). Default: 0.5 |
| `random_seed` | `int \| None` | Semilla para reproducibilidad. Default: None |

**Ejemplo:**
```python
from auto_ml.implementations.augmentators import (
    RandomApplyAugmentator,
    RotationAugmentator,
)
from auto_ml.interfaces import DatasetInterface
import numpy as np

image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
mask = np.zeros((256, 256), dtype=np.uint8)
dataset = DatasetInterface.from_pairs([(image, mask)])

# Aplicar rotación con 70% de probabilidad
augmentator = RandomApplyAugmentator(
    augmentator=RotationAugmentator(angle_range=(-30, 30)),
    probability=0.7,
    random_seed=42,
)
augmented = augmentator.augment(dataset)
```

---

### `OneOfAugmentator`

Similar a `RandomChoiceAugmentator` pero permite especificar pesos/probabilidades para cada aumentación.

**Parámetros del constructor:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `augmentators` | `List[DataAugmentatorInterface]` | Lista de aumentadores |
| `probabilities` | `List[float] \| None` | Probabilidad para cada aumentador. Si es None, usa distribución uniforme. Default: None |
| `random_seed` | `int \| None` | Semilla para reproducibilidad. Default: None |

**Ejemplo:**
```python
from auto_ml.implementations.augmentators import (
    OneOfAugmentator,
    HorizontalFlipAugmentator,
    VerticalFlipAugmentator,
    RotationAugmentator,
)
from auto_ml.interfaces import DatasetInterface
import numpy as np

image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
mask = np.zeros((256, 256), dtype=np.uint8)
dataset = DatasetInterface.from_pairs([(image, mask)])

# Elegir con probabilidades ponderadas (50% flip horizontal, 30% flip vertical, 20% rotación)
augmentator = OneOfAugmentator(
    augmentators=[
        HorizontalFlipAugmentator(),
        VerticalFlipAugmentator(),
        RotationAugmentator(angle_range=(-45, 45)),
    ],
    probabilities=[0.5, 0.3, 0.2],
    random_seed=42,
)
augmented = augmentator.augment(dataset)
```

---

### `MultiplyDatasetAugmentator`

Multiplica el tamaño del dataset aplicando diferentes aumentaciones. Crea N copias del dataset con diferentes aumentaciones.

**Parámetros del constructor:**
| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `augmentators` | `List[DataAugmentatorInterface]` | Lista de aumentadores a aplicar |
| `include_original` | `bool` | Si incluir las muestras originales. Default: True |

**Ejemplo:**
```python
from auto_ml.implementations.augmentators import (
    MultiplyDatasetAugmentator,
    HorizontalFlipAugmentator,
    VerticalFlipAugmentator,
    RotationAugmentator,
)
from auto_ml.interfaces import DatasetInterface
import numpy as np

# Dataset con 2 imágenes
images = [np.random.randint(0, 255, (256, 256), dtype=np.uint8) for _ in range(2)]
masks = [np.zeros((256, 256), dtype=np.uint8) for _ in range(2)]
dataset = DatasetInterface.from_pairs(list(zip(images, masks)))

# Multiplicar dataset: original + 3 versiones aumentadas = 8 muestras totales
augmentator = MultiplyDatasetAugmentator(
    augmentators=[
        HorizontalFlipAugmentator(),
        VerticalFlipAugmentator(),
        RotationAugmentator(angle_range=(-45, 45)),
    ],
    include_original=True,
)
expanded_dataset = augmentator.augment(dataset)
# expanded_dataset tendrá 2 * 4 = 8 muestras
```

---

## Ejemplo Completo: Pipeline de Aumentación

Aquí hay un ejemplo completo que combina varias aumentaciones para crear un pipeline de entrenamiento robusto:

```python
from auto_ml.implementations.augmentators import (
    SequentialAugmentator,
    RandomApplyAugmentator,
    OneOfAugmentator,
    RotationAugmentator,
    HorizontalFlipAugmentator,
    VerticalFlipAugmentator,
    ScaleAugmentator,
    BrightnessAugmentator,
    ContrastAugmentator,
    GaussianNoiseAugmentator,
    ElasticDeformationAugmentator,
)
from auto_ml.interfaces import DatasetInterface
import numpy as np

# Crear dataset de ejemplo
images = [np.random.randint(0, 255, (256, 256), dtype=np.uint8) for _ in range(10)]
masks = [np.zeros((256, 256), dtype=np.uint8) for _ in range(10)]
dataset = DatasetInterface.from_pairs(list(zip(images, masks)))

# Crear pipeline de aumentación complejo
pipeline = SequentialAugmentator([
    # Transformaciones geométricas (aplicar una de ellas)
    OneOfAugmentator([
        HorizontalFlipAugmentator(),
        VerticalFlipAugmentator(),
        RotationAugmentator(angle_range=(-30, 30)),
        ScaleAugmentator(scale_range=(0.8, 1.2)),
    ]),
    
    # Transformaciones fotométricas (aplicar con probabilidad)
    RandomApplyAugmentator(
        BrightnessAugmentator(brightness_range=(0.8, 1.2)),
        probability=0.5,
    ),
    RandomApplyAugmentator(
        ContrastAugmentator(contrast_range=(0.8, 1.2)),
        probability=0.5,
    ),
    
    # Ruido (aplicar con baja probabilidad)
    RandomApplyAugmentator(
        GaussianNoiseAugmentator(noise_std_range=(0, 10)),
        probability=0.3,
    ),
    
    # Deformación elástica para SEM (aplicar con baja probabilidad)
    RandomApplyAugmentator(
        ElasticDeformationAugmentator(alpha=50, sigma=5),
        probability=0.2,
    ),
])

# Aplicar pipeline
augmented_dataset = pipeline.augment(dataset)
```

---

## Resumen de Todas las Clases

| Módulo | Clase | Descripción |
|--------|-------|-------------|
| `identity` | `IdentityAugmentator` | Sin modificaciones (línea base) |
| `geometric` | `RotationAugmentator` | Rotación aleatoria |
| `geometric` | `HorizontalFlipAugmentator` | Volteo horizontal |
| `geometric` | `VerticalFlipAugmentator` | Volteo vertical |
| `geometric` | `ScaleAugmentator` | Zoom in/out |
| `geometric` | `TranslationAugmentator` | Desplazamiento |
| `geometric` | `RandomCropAugmentator` | Recorte aleatorio |
| `photometric` | `BrightnessAugmentator` | Ajuste de brillo |
| `photometric` | `ContrastAugmentator` | Ajuste de contraste |
| `photometric` | `GaussianNoiseAugmentator` | Ruido gaussiano |
| `photometric` | `GaussianBlurAugmentator` | Desenfoque gaussiano |
| `photometric` | `GammaAugmentator` | Corrección gamma |
| `sem_specific` | `ElasticDeformationAugmentator` | Deformación elástica |
| `sem_specific` | `AdaptiveHistogramEqualizationAugmentator` | CLAHE |
| `sem_specific` | `ChargingArtifactAugmentator` | Artefactos de carga SEM |
| `sem_specific` | `ScanLineNoiseAugmentator` | Ruido de líneas de escaneo |
| `composite` | `SequentialAugmentator` | Aplicar en secuencia |
| `composite` | `RandomChoiceAugmentator` | Elegir una aleatoriamente |
| `composite` | `RandomApplyAugmentator` | Aplicar con probabilidad |
| `composite` | `OneOfAugmentator` | Elegir con pesos |
| `composite` | `MultiplyDatasetAugmentator` | Expandir dataset |
