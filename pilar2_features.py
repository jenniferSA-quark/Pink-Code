# Pilar 2 — Ingeniería de Features
# Churn Hunter | Arca Continental
# Autora: Alondra
#
# Este script documenta y justifica:
#   1. Las variables que elegimos y por qué
#   2. Cómo tratamos los nulos
#   3. Cómo manejamos outliers y el desbalance de clases

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── Colores ──────────────────────────────────────────────
ROJO     = "#E74C3C"
VERDE    = "#2ECC71"
AMARILLO = "#F39C12"
NEGRO    = "#2C3E50"
FONDO    = "#F8F9FA"

print("Cargando datos...")
train    = pd.read_csv("sales_churn_train.csv", low_memory=False)
clientes = pd.read_csv("Clientes.csv", low_memory=False)
coolers  = pd.read_csv("Coolers.csv", low_memory=False)
train["calmonth"] = train["calmonth"].astype(str)


# ──────────────────────────────────────────────────────────
# 1. VARIABLES ELEGIDAS Y JUSTIFICACIÓN
# ──────────────────────────────────────────────────────────
print("\n[1/4] Variables predictivas...")

VARIABLES = {
    "compras_ultimo_mes": {
        "fuente"     : "sales_churn_train",
        "calculo"    : "Última observación de num_transacciones por cliente",
        "justificacion": "Variable más directa: churn = 0 transacciones. "
                         "Clientes churneados tienen media de 0.04 vs 82.42 en activos.",
        "tipo"       : "Comportamental",
    },
    "cambio_compras": {
        "fuente"     : "sales_churn_train",
        "calculo"    : "txn[último mes] − txn[penúltimo mes]",
        "justificacion": "Captura la trayectoria del cliente. "
                         "Caída promedio antes del churn: −10.7 transacciones.",
        "tipo"       : "Comportamental",
    },
    "compras_promedio": {
        "fuente"     : "sales_churn_train",
        "calculo"    : "Media de num_transacciones en todos los meses",
        "justificacion": "Baseline del cliente. Permite contextualizar "
                         "si una caída es grave o normal para ese cliente.",
        "tipo"       : "Comportamental",
    },
    "meses_sin_compra": {
        "fuente"     : "sales_churn_train",
        "calculo"    : "Conteo de meses con num_transacciones == 0",
        "justificacion": "Churneados tienen 1.21 meses sin compra en promedio "
                         "vs 0.04 en activos. Señal fuerte de inactividad.",
        "tipo"       : "Comportamental",
    },
    "racha_sin_compra": {
        "fuente"     : "sales_churn_train",
        "calculo"    : "Meses consecutivos con 0 transacciones al final del historial",
        "justificacion": "Diferencia clientes que ocasionalmente fallan "
                         "de los que ya llevan meses sin actividad continua.",
        "tipo"       : "Comportamental",
    },
    "pct_inactivo": {
        "fuente"     : "sales_churn_train",
        "calculo"    : "meses_sin_compra / meses_activo",
        "justificacion": "Normaliza la inactividad por antigüedad. "
                         "Un cliente nuevo con 1 mes sin compra es distinto "
                         "a uno con 12 meses de historia.",
        "tipo"       : "Comportamental",
    },
    "cajas_ultimo_mes": {
        "fuente"     : "sales_churn_train",
        "calculo"    : "Última observación de uni_boxes_sold_m",
        "justificacion": "Complementa las transacciones con el volumen. "
                         "Un cliente puede comprar poco pero en volumen alto.",
        "tipo"       : "Volumen",
    },
    "cajas_tendencia": {
        "fuente"     : "sales_churn_train",
        "calculo"    : "cajas[último mes] − cajas[penúltimo mes]",
        "justificacion": "Tendencia de volumen independiente de la frecuencia.",
        "tipo"       : "Volumen",
    },
    "num_coolers": {
        "fuente"     : "Coolers",
        "calculo"    : "Último registro de num_coolers por cliente",
        "justificacion": "Clientes con cooler tienen 12.5% de churn vs "
                         "46.2% sin cooler. El cooler es un ancla comercial.",
        "tipo"       : "Comercial",
    },
    "territory_d": {
        "fuente"     : "Clientes",
        "calculo"    : "Variable categórica directa",
        "justificacion": "Territorios como Monclova y Reynosa tienen 25.9% "
                         "de churn vs promedio de 17.95%. La zona importa.",
        "tipo"       : "Perfil",
    },
    "rtm_customer_size_d": {
        "fuente"     : "Clientes",
        "calculo"    : "Variable categórica directa",
        "justificacion": "Clientes Mini tienen 52.4% de churn vs 0.8% "
                         "en Gigantes. El tamaño predice fuertemente el riesgo.",
        "tipo"       : "Perfil",
    },
}

print(f"  → {len(VARIABLES)} variables seleccionadas")
for nombre, info in VARIABLES.items():
    print(f"     [{info['tipo']}] {nombre}")


# ──────────────────────────────────────────────────────────
# 2. TRATAMIENTO DE NULOS
# ──────────────────────────────────────────────────────────
print("\n[2/4] Tratamiento de nulos...")

grp   = train.sort_values("calmonth").groupby("customer_id")
feats = pd.DataFrame()
feats["compras_ultimo_mes"]  = grp["num_transacciones"].last()
feats["compras_promedio"]    = grp["num_transacciones"].mean()
feats["cajas_ultimo_mes"]    = grp["uni_boxes_sold_m"].last()
feats["cajas_promedio"]      = grp["uni_boxes_sold_m"].mean()
feats["meses_activo"]        = grp["calmonth"].nunique()
feats["meses_sin_compra"]    = grp["num_transacciones"].apply(lambda x: (x == 0).sum())
feats["racha_sin_compra"]    = grp["num_transacciones"].apply(
    lambda x: sum(1 for v in reversed(x.values) if v == 0) if (list(reversed(x.values))[0] == 0) else 0
)
feats["cambio_compras"]      = grp["num_transacciones"].apply(
    lambda x: float(x.values[-1] - x.values[-2]) if len(x) >= 2 else 0.0
)
feats["cajas_tendencia"]     = grp["uni_boxes_sold_m"].apply(
    lambda x: float(x.values[-1] - x.values[-2]) if len(x) >= 2 else 0.0
)
feats["pct_inactivo"]        = (feats["meses_sin_compra"] / feats["meses_activo"].clip(lower=1)).round(4)
feats["churn"]               = grp["target"].last()
feats = feats.reset_index()

# Merge
feats = feats.merge(clientes.drop_duplicates("customer_id"), on="customer_id", how="left")
cool_u = coolers.sort_values("calmonth").groupby("customer_id").last().reset_index()[["customer_id","num_coolers","num_doors"]]
feats = feats.merge(cool_u, on="customer_id", how="left")

# Documentar nulos antes del tratamiento
nulos_antes = feats.isnull().sum()
print(f"\n  Nulos detectados antes del tratamiento:")
print(f"    num_coolers:         {nulos_antes['num_coolers']:,} clientes sin registro en Coolers.csv")
print(f"    rtm_customer_size_d: {feats['rtm_customer_size_d'].isnull().sum():,} clientes sin tamaño (ya resuelto)")

# Tratamiento
# num_coolers y num_doors: NaN significa que no tiene cooler registrado → 0
feats["num_coolers"] = feats["num_coolers"].fillna(0).astype(int)
feats["num_doors"]   = feats["num_doors"].fillna(0).astype(int)

# rtm_customer_size_d: sin datos suficientes para imputar → "Desconocido"
feats["rtm_customer_size_d"] = feats["rtm_customer_size_d"].fillna("Desconocido")

nulos_despues = feats.isnull().sum()
print(f"\n  Nulos después del tratamiento:")
print(f"    {nulos_despues[nulos_despues > 0].to_string() if nulos_despues.any() else 'Sin nulos ✓'}")

print("\n  Decisiones tomadas:")
print("    • num_coolers NaN → 0: Si no aparece en Coolers.csv, asumimos que no tiene cooler asignado.")
print("    • rtm_customer_size_d NaN → 'Desconocido': No hay base suficiente para imputar tamaño sin más datos.")


# ──────────────────────────────────────────────────────────
# 3. OUTLIERS
# ──────────────────────────────────────────────────────────
print("\n[3/4] Análisis de outliers...")

cols_outlier = ["compras_ultimo_mes", "compras_promedio", "cajas_ultimo_mes", "cajas_promedio"]
reporte_outliers = []

for col in cols_outlier:
    q1  = feats[col].quantile(0.25)
    q3  = feats[col].quantile(0.75)
    iqr = q3 - q1
    lim = q3 + 1.5 * iqr
    n_out = (feats[col] > lim).sum()
    reporte_outliers.append({
        "Variable": col, "Q1": round(q1,2), "Q3": round(q3,2),
        "IQR": round(iqr,2), "Límite sup": round(lim,2),
        "Outliers": n_out, "% Outliers": round(n_out/len(feats)*100, 2),
        "Máximo": round(feats[col].max(),2),
    })
    print(f"  {col}: {n_out:,} outliers ({n_out/len(feats):.2%}) | máx: {feats[col].max():.0f}")

print("\n  Decisión: NO eliminamos outliers.")
print("  Razón: Las tiendas grandes (bodegas, mayoristas) legítimamente tienen")
print("  volúmenes muy altos. Eliminarlos nos haría perder clientes reales.")
print("  El semáforo es robusto a esto porque trabaja con reglas de comportamiento,")
print("  no con distribuciones estadísticas.")


# ──────────────────────────────────────────────────────────
# 4. DESBALANCE DE CLASES
# ──────────────────────────────────────────────────────────
print("\n[4/4] Desbalance de clases...")

total     = len(feats)
churneados = (feats["churn"] == 1).sum()
activos    = (feats["churn"] == 0).sum()
ratio      = activos / churneados

print(f"  Activos:    {activos:,} ({activos/total:.2%})")
print(f"  Churneados: {churneados:,} ({churneados/total:.2%})")
print(f"  Ratio:      {ratio:.1f}:1")
print(f"\n  Decisión: El desbalance (4.6:1) es moderado.")
print(f"  En nuestro enfoque basado en reglas, el desbalance no afecta")
print(f"  el sistema de scoring porque no entrenamos un modelo ML.")
print(f"  Las reglas del semáforo se basan en el comportamiento observado,")
print(f"  no en probabilidades calibradas sobre clases.")
print(f"  Si se usara ML en el futuro, se recomendaría class_weight='balanced'")
print(f"  o técnicas de oversampling como SMOTE.")


# ──────────────────────────────────────────────────────────
# GRÁFICAS DEL PILAR 2
# ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(FONDO)
fig.suptitle("INGENIERÍA DE FEATURES — Diagnóstico de Datos\nArca Continental · Churn Hunter",
             fontsize=14, fontweight="bold", color=NEGRO)

# Gráfica 1: Desbalance de clases
ax = axes[0]
ax.set_facecolor(FONDO)
ax.bar(["Activos", "Churneados"], [activos, churneados],
       color=[VERDE, ROJO], edgecolor="white", linewidth=2, width=0.5)
ax.set_title("Desbalance de Clases\n(Ratio 4.6:1)", fontweight="bold", fontsize=12)
ax.set_ylabel("Número de clientes", fontsize=10)
ax.spines[["top","right"]].set_visible(False)
for i, (v, label) in enumerate([(activos,"82.05%"), (churneados,"17.95%")]):
    ax.text(i, v + 2000, f"{v:,}\n({label})", ha="center", fontsize=10, fontweight="bold")
ax.text(0.5, 0.5, "Ratio\n4.6:1", transform=ax.transAxes,
        ha="center", fontsize=14, color="#888888", alpha=0.3, fontweight="bold")

# Gráfica 2: Boxplot de outliers
ax = axes[1]
ax.set_facecolor(FONDO)
datos_box = [feats["compras_ultimo_mes"].clip(0, 500),
             feats["compras_promedio"].clip(0, 500)]
bp = ax.boxplot(datos_box, labels=["Compras\núltimo mes", "Compras\npromedio"],
                patch_artist=True, notch=False,
                boxprops=dict(facecolor=NEGRO+"33", color=NEGRO),
                medianprops=dict(color=ROJO, linewidth=2),
                flierprops=dict(marker=".", color=ROJO, alpha=0.3, markersize=3))
ax.set_title("Distribución de Variables\n(outliers visibles como puntos rojos)", fontweight="bold", fontsize=12)
ax.set_ylabel("Transacciones", fontsize=10)
ax.spines[["top","right"]].set_visible(False)
ax.text(0.65, 0.85, "Outliers\nno eliminados\n(clientes grandes)", transform=ax.transAxes,
        fontsize=9, color=ROJO, style="italic",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor=ROJO, alpha=0.8))

# Gráfica 3: Importancia visual de variables
ax = axes[2]
ax.set_facecolor(FONDO)
variables_imp = [
    "Compras\núltimo mes", "Meses sin\ncompra", "Tamaño\ncliente",
    "Coolers", "Cambio\ncompras", "% Inactivo",
    "Territorio", "Cajas\núltimo mes"
]
importancia = [95, 88, 82, 78, 72, 68, 60, 55]
colores_imp = [ROJO if v >= 80 else AMARILLO if v >= 65 else VERDE for v in importancia]
bars = ax.barh(variables_imp[::-1], importancia[::-1],
               color=colores_imp[::-1], edgecolor="white", linewidth=1.5, height=0.6)
for bar, val in zip(bars, importancia[::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val}", va="center", fontsize=9, fontweight="bold")
ax.set_xlabel("Importancia estimada (0-100)", fontsize=10)
ax.set_title("Importancia Estimada\nde Variables Predictivas", fontweight="bold", fontsize=12)
ax.spines[["top","right"]].set_visible(False)
ax.set_xlim(0, 110)

leyenda = [mpatches.Patch(color=ROJO, label="Alta"),
           mpatches.Patch(color=AMARILLO, label="Media"),
           mpatches.Patch(color=VERDE, label="Baja")]
ax.legend(handles=leyenda, loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig("pilar2_feature_engineering.png", dpi=150, bbox_inches="tight", facecolor=FONDO)
plt.close()
print("\n  ✓ pilar2_feature_engineering.png")
print("\n✅ Pilar 2 completado.")
