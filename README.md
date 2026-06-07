# 🏪 Churn Hunter — Arca Continental

## Resumen

Este proyecto desarrolla un sistema de scoring de riesgo de churn para el canal tradicional de Arca Continental — las tienditas de abarrotes y misceláneas que representan el corazón de su operación comercial en México. A través de un pipeline de datos construido en Python y MongoDB, clasificamos a cada cliente en un semáforo de 4 niveles de riesgo (⚫ Inminente, 🔴 Alto, 🟡 Moderado, 🟢 Bajo) con base en su comportamiento de compra histórico, presencia de coolers y perfil comercial. El sistema permite al equipo de Arca actuar de forma proactiva antes de que una tiendita deje de comprar, atacando los **$12 millones MXN mensuales** que se pierden por churn.

Desarrollado por el equipo **Pink Code** para el hackathon de Digital Nest — Arca Continental 2026.

---

## Tabla de Contenidos

- [Sobre los Datos](#sobre-los-datos)
- [Análisis Exploratorio de Datos](#análisis-exploratorio-de-datos)
  - [Panorama General](#panorama-general)
  - [Churn por Territorio](#churn-por-territorio)
  - [Impacto de los Coolers](#impacto-de-los-coolers)
  - [Churn por Tamaño de Cliente](#churn-por-tamaño-de-cliente)
- [Ingeniería de Features](#ingeniería-de-features)
  - [Variables Seleccionadas](#variables-seleccionadas)
  - [Tratamiento de Nulos](#tratamiento-de-nulos)
  - [Outliers y Desbalance de Clases](#outliers-y-desbalance-de-clases)
- [Sistema de Scoring](#sistema-de-scoring)
  - [Definición de Churn](#definición-de-churn)
  - [Semáforo de Riesgo](#semáforo-de-riesgo)
  - [Justificación del Enfoque](#justificación-del-enfoque)
- [Insights de Negocio](#insights-de-negocio)
  - [Pregunta 1 — Variables más influyentes](#pregunta-1--variables-más-influyentes)
  - [Pregunta 2 — Impacto del Territorio](#pregunta-2--impacto-del-territorio)
  - [Pregunta 3 — Impacto de los Coolers](#pregunta-3--impacto-de-los-coolers)
  - [Recomendaciones Accionables](#recomendaciones-accionables)
- [Base de Datos en MongoDB](#base-de-datos-en-mongodb)
- [Instalación y Uso](#instalación-y-uso)
  - [Requisitos](#requisitos)
  - [Ejecución paso a paso](#ejecución-paso-a-paso)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Tecnologías](#tecnologías)
- [Referencias](#referencias)

---

## Sobre los Datos

El proyecto utiliza 4 tablas de datos anonimizados reales de Arca Continental, correspondientes al canal tradicional en México desde 2024.

| Archivo | Descripción | Filas |
|---|---|---|
| `sales_churn_train.csv` | Transacciones mensuales por cliente con etiqueta de churn | 5,030,534 |
| `sales_churn_test.csv` | Clientes a predecir (sin etiqueta) | 199,923 |
| `Clientes.csv` | Perfil comercial: territorio, subcanal, tamaño | 371,727 |
| `Coolers.csv` | Refrigeradores por cliente y mes | 4,636,676 |

**Filtros aplicados a los datos:** Solo canal tradicional · Solo México · Desde 2024 · Productos: TCC, Monster, Jugos del Valle · Solo ventas válidas.

> Los archivos de datos no están incluidos en este repositorio por razones de confidencialidad.

---

## Análisis Exploratorio de Datos

### Panorama General

De los 241,805 clientes únicos en el conjunto de entrenamiento:

- **198,403 clientes activos** (82.05%)
- **43,402 clientes churneados** (17.95%)
- Ratio de desbalance: **4.6:1**
- Valor promedio por cliente churneado: **$276 MXN/mes**

### Churn por Territorio

El territorio es una variable significativa. Los 25 territorios analizados muestran tasas de churn que van del 12% al 26%.

| Territorio | Clientes | % Churn | Impacto MXN/mes |
|---|---|---|---|
| Guadalajara | 31,541 | 22.2% | $1.9 MM |
| Monterrey | 25,220 | 19.5% | $1.4 MM |
| San Luis Potosí | 15,557 | 21.0% | $0.9 MM |
| Comarca Lagunera | 12,941 | 20.7% | $0.7 MM |
| Monclova / Reynosa | ~5,000 c/u | 25.9% | — |

### Impacto de los Coolers

La presencia de un refrigerador Arca en el punto de venta es el factor protector más importante:

| Condición | Clientes | % Churn |
|---|---|---|
| Sin cooler | 38,968 | **46.2%** |
| Con cooler | 202,837 | **12.5%** |

Los coolers reducen el churn en **33.7 puntos porcentuales**.

### Churn por Tamaño de Cliente

| Tamaño | Clientes | % Churn |
|---|---|---|
| Mini | 60,511 | **52.4%** |
| Pequeño | 63,547 | 12.8% |
| Mediano | 78,916 | 4.0% |
| Grande | 23,691 | 1.1% |
| Gigante | 15,140 | 0.8% |

Las tienditas **Mini** son el segmento más vulnerable y representan **$8.7 MM MXN/mes** en churn.

---

## Ingeniería de Features

### Variables Seleccionadas

Se construyeron 11 variables predictivas a partir de las 4 fuentes de datos:

| Variable | Fuente | Tipo | Justificación |
|---|---|---|---|
| `compras_ultimo_mes` | sales_churn_train | Comportamental | Variable más discriminante: churneados tienen 0.04 vs 82.42 en activos |
| `cambio_compras` | sales_churn_train | Comportamental | Caída promedio antes del churn: −10.7 transacciones |
| `compras_promedio` | sales_churn_train | Comportamental | Baseline histórico del cliente |
| `meses_sin_compra` | sales_churn_train | Comportamental | Churneados: 1.21 meses vs 0.04 en activos |
| `racha_sin_compra` | sales_churn_train | Comportamental | Meses consecutivos sin actividad al final del historial |
| `pct_inactivo` | sales_churn_train | Comportamental | Inactividad normalizada por antigüedad del cliente |
| `cajas_ultimo_mes` | sales_churn_train | Volumen | Complementa frecuencia con volumen de producto |
| `cajas_tendencia` | sales_churn_train | Volumen | Tendencia de volumen independiente de frecuencia |
| `num_coolers` | Coolers | Comercial | Sin cooler: 46.2% churn vs 12.5% con cooler |
| `territory_d` | Clientes | Perfil | Variación significativa entre territorios (12%–26%) |
| `rtm_customer_size_d` | Clientes | Perfil | Mini: 52.4% churn vs Gigante: 0.8% |

### Tratamiento de Nulos

| Campo | Nulos detectados | Tratamiento | Razón |
|---|---|---|---|
| `num_coolers` | 38,968 | Rellenar con `0` | Si no aparece en Coolers.csv, no tiene cooler asignado |
| `rtm_customer_size_d` | 0 | Sin acción necesaria | Sin nulos en la fuente |

### Outliers y Desbalance de Clases

**Outliers:** Se detectaron mediante el método IQR. El 4.5%–6.9% de los registros superan el límite de 1.5×IQR en variables de volumen.

> **Decisión: No se eliminaron.** Las bodegas y mayoristas legítimamente compran volúmenes mucho mayores que una tiendita Mini. El semáforo usa reglas de comportamiento, no distribuciones estadísticas, por lo que los outliers no distorsionan la clasificación.

**Desbalance (4.6:1):** Al usar un sistema basado en reglas en lugar de ML, el desbalance no afecta el scoring. Si en el futuro se implementa un modelo ML, se recomienda usar `class_weight='balanced'` o técnicas de oversampling como SMOTE.

---

## Sistema de Scoring

### Definición de Churn

> Un cliente se considera churneado cuando lleva **1 mes sin realizar ninguna compra**.
> — Definición oficial Arca Continental

### Semáforo de Riesgo

Las reglas se aplican en orden de prioridad:

| Nivel | Condición | Acción sugerida |
|---|---|---|
| ⚫ **Riesgo Inminente** | `target=1` confirmado, o sin compras 2+ meses consecutivos | Análisis post-mortem, intento de recuperación |
| 🔴 **Riesgo Alto** | No compró el último mes, o >20% de historia inactiva | Visita urgente del ejecutivo de cuenta |
| 🟡 **Riesgo Moderado** | Compra pero cayó ≥10 txn vs mes anterior, o 5–20% de historia inactiva | Llamada proactiva, oferta de retención |
| 🟢 **Riesgo Bajo** | Compra regularmente, sin señales de alarma | Mantener relación comercial |

### Justificación del Enfoque

Se eligió un **sistema basado en reglas de negocio** sobre un modelo ML por las siguientes razones:

1. **Interpretabilidad total** — el equipo comercial puede entender y confiar en la clasificación
2. **Los umbrales están respaldados por los datos** — se validaron comparando clientes churneados vs activos en el conjunto de entrenamiento
3. **No requiere reentrenamiento** — las reglas se pueden ajustar directamente si cambia el negocio
4. **Robusto al desbalance** — no depende de probabilidades calibradas sobre clases

---

## Insights de Negocio

### Pregunta 1 — Variables más influyentes

La variable más predictiva es `compras_ultimo_mes`. Los clientes churneados prácticamente no compraron en su último mes registrado (media de 0.04 transacciones), mientras que los activos promedian 82.42. La tendencia negativa de −10.7 transacciones es la segunda señal más fuerte.

### Pregunta 2 — Impacto del Territorio

Sí, el territorio influye significativamente. Guadalajara y Monterrey concentran el mayor impacto absoluto en pesos ($3.3 MM MXN/mes entre los dos). Monclova y Reynosa tienen la tasa proporcional más alta (25.9%) — casi 1 de cada 4 tienditas se va.

### Pregunta 3 — Impacto de los Coolers

Sí, los coolers son el factor protector más poderoso del dataset. Un cliente con cooler tiene **3.7 veces menos probabilidad** de hacer churn que uno sin cooler.

### Recomendaciones Accionables

1. **Priorizar clientes Mini sin cooler en Riesgo Alto** — representan $8.7 MM MXN/mes en churn. Recuperar el 20% = $1.75 MM MXN/mes adicionales.
2. **Intervención urgente en Guadalajara y Monterrey** — juntos representan $3.3 MM MXN/mes en clientes perdidos.
3. **Programa de instalación de coolers** — instalar coolers en los 38,968 clientes sin cooler podría rescatar ~13,000 clientes y recuperar ~$3.5 MM MXN/mes.

---

## Base de Datos en MongoDB

El pipeline guarda los resultados en la base de datos `arca_churn` con dos colecciones:

**`clientes_scoring`** — Un documento por cliente con su semáforo y todas las variables calculadas.

```json
{
  "customer_id": "d72f5c4b...",
  "nivel_riesgo": "Riesgo Alto",
  "razon": "No compró el último mes",
  "compras_ultimo_mes": 0,
  "cambio_compras": -15,
  "num_coolers": 0,
  "territory_d": "Monterrey",
  "rtm_customer_size_d": "Mini",
  "updated_at": "2026-06-07T00:00:00Z"
}
```

**`scoring_log`** — Registro de cada ejecución del pipeline con métricas de distribución.

---

## Instalación y Uso

### Requisitos

- Python 3.11+
- MongoDB Community Edition

```bash
pip install pandas pymongo tqdm matplotlib
```

Instalar y arrancar MongoDB en Mac:
```bash
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
```

### Ejecución paso a paso

**1. Clona el repositorio:**
```bash
git clone https://github.com/jenniferSA-quark/Pink-Code.git
cd Pink-Code
```

**2. Coloca los 5 archivos CSV en la carpeta del proyecto**

**3. Corre el pipeline principal:**
```bash
python3 churn_pipeline.py
```

**4. (Opcional) Corre los análisis complementarios:**
```bash
python3 eda_churn.py            # Gráficas del EDA
python3 pilar2_features.py      # Diagnóstico de features
python3 pilar4_insights.py      # Insights con impacto en MXN
```

**5. Visualiza los resultados en MongoDB Compass:**
```
mongodb://localhost:27017 → base de datos: arca_churn
```

---

## Estructura del Repositorio

```
Pink-Code/
│
├── churn_pipeline.py       # Pipeline principal
├── eda_churn.py            # Análisis exploratorio
├── pilar2_features.py      # Ingeniería de features
├── pilar4_insights.py      # Insights de negocio
└── README.md               # Este archivo
```

---

## Tecnologías

| Herramienta | Versión | Uso |
|---|---|---|
| Python | 3.11 | Lenguaje principal |
| Pandas | 2.x | Manipulación de datos |
| MongoDB | 8.3 | Base de datos de documentos |
| PyMongo | 4.x | Conexión Python ↔ MongoDB |
| Matplotlib | 3.x | Visualizaciones |
| tqdm | — | Barras de progreso |

---

## Referencias

- Arca Continental — Planteamiento del reto, Digital Nest 2026
- [Documentación oficial de MongoDB](https://www.mongodb.com/docs/)
- [Documentación de PyMongo](https://pymongo.readthedocs.io/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/)
- He, H., & Garcia, E. A. (2009). *Learning from Imbalanced Data*. IEEE Transactions on Knowledge and Data Engineering.

---

👥 **Equipo Pink Code** · Hackathon Digital Nest — Arca Continental · 2026
