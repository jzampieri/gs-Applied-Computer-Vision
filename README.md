# Orbital Logis — GAIE: Módulo de Visão Computacional

> **Global Solution 2026 — FIAP | Applied Computer Vision (ACV) | ESW 4º Ano**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-EuroSAT-green)](https://github.com/phelber/EuroSAT)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Integrantes

| Nome | RM |
|------|-----|
| Júlio César Zampieri | RM98772 |
| João Gabriel Dias | RM99092 |
| Ricardo Matos | RM95906 |

---

## Contexto — Orbital Logis e a Síndrome de Kessler

O **Orbital Logis** é um sistema integrado de detecção e resposta autônoma a detritos espaciais. A **Síndrome de Kessler** descreve um cenário em que a densidade de detritos em órbita baixa terrestre (LEO) atinge um ponto crítico de cascata — ameaçando satélites de GPS, comunicação, monitoramento climático e a internet global via satélite.

Este repositório contém o módulo **GAIE (Ground-based AI Evaluation Engine)**, responsável pela classificação automática de imagens satelitais para apoio à tomada de decisão autônoma do sistema.

```
Satélite (imagem) → GAIE [este módulo] → API → RPA → Ação Autônoma
                          ↑
                    BDDI (pipeline de dados)
```

**Conexão com ODS 9**: Proteção da infraestrutura orbital que sustenta inovação, comunicação e conectividade global.

---

## Objetivo

Desenvolver e comparar **2 arquiteturas de CNN treinadas do zero** (sem transfer learning) para classificação de imagens satelitais, com acurácia mínima de referência de **88%** no conjunto de teste.

---

## Dataset — EuroSAT

| Propriedade | Valor |
|-------------|-------|
| Fonte | Satélite Sentinel-2 (ESA/Copernicus) |
| Total de imagens | 27.000 |
| Resolução | 64 × 64 pixels (RGB) |
| Classes | 10 (balanceado, ~2.700/classe) |
| Download | Automático via `torchvision` |

### Classes

| Classe (EN) | Classe (PT) | Relevância Orbital |
|-------------|-------------|-------------------|
| AnnualCrop | Lavoura Anual | Agronegócio depende de GPS e meteorologia satelital |
| Forest | Floresta | Monitoramento ambiental por satélite |
| HerbaceousVegetation | Veg. Herbácea | Indicador ecológico orbital |
| Highway | Rodovia | Infraestrutura logística dependente de GPS |
| Industrial | Área Industrial | Alta dependência de dados orbitais |
| Pasture | Pastagem | Rastreamento pecuário via satélite |
| PermanentCrop | Lavoura Perm. | Agricultura de precisão |
| Residential | Área Residencial | Internet via satélite |
| River | Rio | Monitoramento hídrico orbital |
| SeaLake | Mar / Lago | Navegação e pesca via GPS |

---

## Arquiteturas CNN (do Zero)

### Modelo 1 — OrbitalVision Baseline

```
INPUT (3, 64, 64)
├── Bloco 1: Conv(3→32) → BN → ReLU → MaxPool  [64→32]
├── Bloco 2: Conv(32→64) → BN → ReLU → MaxPool [32→16]
├── Bloco 3: Conv(64→128) → BN → ReLU → MaxPool[16→8]
├── Flatten
├── Dropout(0.4) → FC(8192→256) → ReLU
└── Dropout(0.3) → FC(256→10)
```
- **Parâmetros**: ~2.2M
- **Scheduler**: StepLR (step=10, gamma=0.5)
- **Épocas**: 30

### Modelo 2 — OrbitalVision Deep

```
INPUT (3, 64, 64)
├── Bloco 1: [Conv→BN→ReLU]×2 → MaxPool → Dropout2d(0.10) [64→32]
├── Bloco 2: [Conv→BN→ReLU]×2 → MaxPool → Dropout2d(0.20) [32→16]
├── Bloco 3: [Conv→BN→ReLU]×2 → MaxPool → Dropout2d(0.25) [16→8]
├── Bloco 4: [Conv→BN→ReLU]×2 → MaxPool → Dropout2d(0.25) [8→4]
├── GlobalAvgPool → (512)
├── FC(512→512) → BN → ReLU → Dropout(0.5)
├── FC(512→256) → ReLU → Dropout(0.3)
└── FC(256→10)
```
- **Parâmetros**: ~3.5M
- **Scheduler**: CosineAnnealingLR (T_max=40, eta_min=1e-6)
- **Épocas**: 40

---

## Como Executar

### 1. Pré-requisitos

```bash
# Python 3.10+
python --version
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Executar o notebook

```bash
jupyter notebook orbital_logis_acv.ipynb
```

> **GPU recomendada** — Sem GPU, o treinamento pode levar 1-2h (CPU).
> Para aceleração gratuita, use **Google Colab** ou **Kaggle Notebooks**.

### 4. Executar tudo de uma vez

No Jupyter, vá em: `Kernel → Restart & Run All`

O dataset EuroSAT (~92MB) será baixado automaticamente na primeira execução.

---

## Estrutura do Repositório

```
GS-Computer-Vision/
├── orbital_logis_acv.ipynb    # Notebook principal (treino, avaliação, demo)
├── requirements.txt            # Dependências Python
├── README.md                   # Este arquivo
├── models_saved/               # Pesos salvos após treinamento (gerado)
│   ├── orbital_baseline.pth
│   └── orbital_deep.pth
├── data/                       # Dataset EuroSAT (baixado automaticamente)
│   └── eurosat/
└── *.png                       # Gráficos gerados pelo notebook
```

---

## Resultados

> Os resultados exatos dependem do hardware e são gerados ao executar o notebook.

| Modelo | Acurácia (Teste) | F1-Score | Parâmetros |
|--------|-----------------|----------|------------|
| OrbitalVision Baseline | > 85% | - | ~2.2M |
| OrbitalVision Deep | > 88% | - | ~3.5M |

*Meta: ≥ 88% de acurácia no conjunto de teste.*

---

## Demonstração Funcional

O notebook inclui uma interface **Gradio** interativa para classificação de novas imagens:

```python
# Também disponível diretamente:
result = predict_image(pil_image)  # retorna Top-3 classes com probabilidades
```

---

## Referências

- Helber, P., et al. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification*. IEEE JSTARS.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). *Deep learning*. Nature.
- Simonyan & Zisserman (2015). *Very Deep Convolutional Networks*. ICLR.
- ESA Copernicus. Sentinel-2 Mission Guide. https://sentinel.esa.int
- Kessler, D. J., & Cour-Palais, B. G. (1978). *Collision frequency of artificial satellites*. JGR.

---

**Orbital Logis — Global Solution 2026 | FIAP | Período: 25/05 – 09/06/2026**
