## State-of-the-Art: Data Augmentation in Scanning Electron Microscopy (SEM) Imagery - Comprehensive Table

| Article Title | Authors | Institution(s) | Year | Paper Focus | Base Model | Dataset(s) | Image Type | Application Domain | Image Resolution | Data Aug | Aug Techniques |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Enhancing Electron Microscopy Image Classification Using Data Augmentation** | Welsman J A, Weber G H, Amusat O O, Giannakou A, Ramakrishnan L | Lawrence Berkeley National Laboratory; Bournemouth University | N/A | Comparative Study | DenseNet169, MobileNetV2, ResNet101V2 | National Center for Electron Microscopy dataset (727 EM images) | Electron Microscopy (EM) | Image Classification (Metadata Generation) | Original: 1024x1024; Input: 224x224 | ✓ | Flipping, Random Masking, 90º Rotation, Random Rotation, Random Shifting, Random Zooming |
| **Enhancing scanning electron microscopy imaging quality of weakly conductive samples through unsupervised learning** | Gao X, Huang T, Tang P, Di J, Zhong L, Zhang W | Guangdong University of Technology, China | 2024 | Novel Model | CycleGAN | Simulated (Gaussian, Hybrid blur) and Real SEM (WO3, CuS, SiO2) | SEM (Weakly conductive samples) | Image Quality Enhancement / Deblurring | 256×256 px | ✓ | Edge Loss ($\text{L}_{\text{edge}}$) using Sobel operator; SSIM Loss ($\text{L}_{\text{cycle}}$) |
| **Physics-Based Synthetic Data Model for Automated Segmentation in Catalysis Microscopy** | Vuijk M, Ducci G, Sandoval L, Pietsch M, Reuter K, Lunkenbein T, Scheurer C | Fritz-Haber-Institut; Technische Universität München; Forschungszentrum Jülich | 2024 | Novel Model / Hybrid | U-NET | ESEM time-series (1,600 frames) of isopropanol oxidation on cobalt oxide catalyst | ESEM (time-series) | Semantic Segmentation (Evolving crack detection) | 512×512 px | ✓ | Physics-based crack trajectory generation (Distance Transform Map constraint: avoiding pores by 5 pixels); Geometric Aug. (Rotation, Translation, Scaling) |
| **Generation of highly realistic microstructural images of alloys from limited data with a style‑based generative adversarial network** | Lambard G, Yamazaki K, Demura M | National Institute for Materials Science (NIMS); JFE Steel Corporation | 2023 | Novel Model | StyleGAN2 with ADA | Private dataset of 3000 SEM images of ferrite-martensite DP steel sheets | SEM (Dual-Phase steel microstructures) | Synthetic Data Generation (Materials Science/FEM Simulation) | 512×512 px | ✓ (ADA mechanism) | Pixel blitting, geometrical transformations (X-flip, rotation, translation, scaling); Target heuristic $r_t$ = 0.5 |
| **Generation of highly realistic microstructural images of alloys from limited data with a style‑based generative adversarial network** | Ferreira I, Ochoa L, Koeshidayatullah A | King Fahd University of Petroleum and Minerals; Universidad Nacional de Colombia | N/A | Novel Model | StyleGAN2 with ADA | >10,000 thin section images (PPL and XPL) across four rock types | Petrographic Thin Section Images (PPL/XPL) | Synthetic Data Generation (Geosciences/Image Self-Labeling) | 512×512 px | ✓ | Image slicing; Truncation Trick (0.7 optimal) |

**Legend:**
*  ✓ = Yes/Used
*  ❌ = No/Not used
*  N/A = Not applicable/Not reported
*  Acc = Accuracy
*  Aug = Augmentation
*  FID = Fréchet Inception Distance
*  Seg = Segmentation
*  Trans = Transformer
*  ADA = Adaptive Discriminator Augmentation
*  kimg = Thousand images processed by discriminator

---