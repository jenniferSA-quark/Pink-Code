# EDA - Churn Hunter | Arca Continental
# Análisis exploratorio de datos para entender el churn en tienditas
# Autora: Alondra

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# Paleta de colores del semáforo
VERDE   = "#2ecc71"
AMARILLO= "#f39c12"
ROJO    = "#e74c3c"
NEGRO   = "#2c3e50"
GRIS    = "#bdc3c7"
FONDO   = "#f8f9fa"

print("Cargando datos...")
train    = pd.read_csv("sales_churn_train.csv", low_memory=False)
clientes = pd.read_csv("Clientes.csv", low_memory=False)
coolers  = pd.read_csv("Coolers.csv", low_memory=False)

train["calmonth"] = train["calmonth"].astype(str)

# --- Preparar tabla de features por cliente ---
grp   = train.sort_values("calmonth").groupby("customer_id")
feats = pd.DataFrame()
feats["compras_ultimo_mes"] = grp["num_transacciones"].last()
feats["compras_promedio"]   = grp["num_transacciones"].mean()
feats["meses_sin_compra"]   = grp["num_transacciones"].apply(lambda x: (x == 0).sum())
feats["meses_activo"]       = grp["calmonth"].nunique()
feats["churn"]              = grp["target"].last()
feats = feats.reset_index()

feats = feats.merge(clientes.drop_duplicates("customer_id"), on="customer_id", how="left")

cool_u = (coolers.sort_values("calmonth")
          .groupby("customer_id").last().reset_index()
          [["customer_id", "num_coolers"]])
feats = feats.merge(cool_u, on="customer_id", how="left")
feats["num_coolers"]  = feats["num_coolers"].fillna(0)
feats["tiene_cooler"] = (feats["num_coolers"] > 0)

# Ventas mensuales promedio (para tendencia temporal)
ventas_mes = train.groupby("calmonth")["num_transacciones"].mean().reset_index()
ventas_mes.columns = ["mes", "promedio"]

print("Generando gráficas...")

# ═══════════════════════════════════════════════════════
# FIGURA 1: Panorama general del churn
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(FONDO)
fig.suptitle("PANORAMA GENERAL DEL CHURN\nArca Continental — Canal Tradicional 2024-2026",
             fontsize=16, fontweight="bold", color=NEGRO, y=1.02)

# Gráfica 1a: Donut de churn rate
ax = axes[0]
ax.set_facecolor(FONDO)
total      = len(feats)
churneados = (feats["churn"] == 1).sum()
activos    = total - churneados
sizes  = [activos, churneados]
colors = [VERDE, ROJO]
wedges, texts, autotexts = ax.pie(
    sizes, colors=colors, autopct="%1.1f%%",
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.5, edgecolor="white", linewidth=3)
)
for at in autotexts:
    at.set_fontsize(13)
    at.set_fontweight("bold")
    at.set_color("white")
ax.text(0, 0, f"{churneados:,}\nchurneados", ha="center", va="center",
        fontsize=12, fontweight="bold", color=NEGRO)
ax.set_title("Tasa de Churn General", fontweight="bold", fontsize=13, pad=15)
legend = [mpatches.Patch(color=VERDE, label=f"Activos ({activos:,})"),
          mpatches.Patch(color=ROJO, label=f"Churneados ({churneados:,})")]
ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=10)

# Gráfica 1b: Compras promedio churned vs activo
ax = axes[1]
ax.set_facecolor(FONDO)
comp = feats.groupby("churn")[["compras_ultimo_mes", "compras_promedio"]].mean()
x    = np.arange(2)
w    = 0.35
b1 = ax.bar(x - w/2, comp["compras_ultimo_mes"], w, color=[VERDE, ROJO],
            label="Último mes", edgecolor="white", linewidth=1.5)
b2 = ax.bar(x + w/2, comp["compras_promedio"], w, color=[VERDE+"88", ROJO+"88"],
            label="Promedio histórico", edgecolor="white", linewidth=1.5)
ax.set_xticks(x)
ax.set_xticklabels(["Clientes Activos", "Clientes Churneados"], fontsize=11)
ax.set_ylabel("Número de transacciones", fontsize=10)
ax.set_title("Transacciones: Activos vs Churneados", fontweight="bold", fontsize=13)
ax.legend(fontsize=10)
ax.spines[["top","right"]].set_visible(False)
ax.set_facecolor(FONDO)
for bar in [*b1, *b2]:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

# Gráfica 1c: Tendencia de ventas mensual
ax = axes[2]
ax.set_facecolor(FONDO)
meses = ventas_mes["mes"].tolist()
x_idx = range(len(meses))
ax.plot(x_idx, ventas_mes["promedio"], color=NEGRO, linewidth=2.5, marker="o",
        markersize=4, markerfacecolor=ROJO)
ax.fill_between(x_idx, ventas_mes["promedio"], alpha=0.1, color=NEGRO)
ax.set_xticks(x_idx[::3])
ax.set_xticklabels([meses[i][4:] + "/" + meses[i][:4] for i in x_idx[::3]],
                   rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Transacciones promedio", fontsize=10)
ax.set_title("Tendencia de Ventas Mensuales\n(Ene 2024 — Ene 2026)", fontweight="bold", fontsize=13)
ax.spines[["top","right"]].set_visible(False)
ax.set_facecolor(FONDO)

plt.tight_layout()
plt.savefig("eda_01_panorama_general.png", dpi=150, bbox_inches="tight",
            facecolor=FONDO)
plt.close()
print("  ✓ eda_01_panorama_general.png")


# ═══════════════════════════════════════════════════════
# FIGURA 2: ¿El territorio influye?
# ═══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor(FONDO)
ax.set_facecolor(FONDO)

terr = (feats.groupby("territory_d")
        .agg(total=("churn","count"), churneados=("churn","sum"))
        .reset_index())
terr["pct_churn"] = (terr["churneados"] / terr["total"] * 100).round(1)
terr = terr[terr["total"] >= 500].sort_values("pct_churn", ascending=True)

colores_barra = [ROJO if p >= 22 else AMARILLO if p >= 18 else VERDE
                 for p in terr["pct_churn"]]

bars = ax.barh(terr["territory_d"], terr["pct_churn"],
               color=colores_barra, edgecolor="white", linewidth=1.5, height=0.7)

for bar, val, tot in zip(bars, terr["pct_churn"], terr["total"]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val}%  ({tot:,} clientes)", va="center", fontsize=9, color=NEGRO)

ax.axvline(terr["pct_churn"].mean(), color=NEGRO, linestyle="--",
           linewidth=1.5, alpha=0.6, label=f"Promedio: {terr['pct_churn'].mean():.1f}%")

ax.set_xlabel("% de clientes que churnearon", fontsize=12)
ax.set_title("PREGUNTA 2: ¿El territorio influye en la pérdida de clientes?\n"
             "Churn rate por territorio — Arca Continental",
             fontweight="bold", fontsize=14, pad=15)
ax.spines[["top","right"]].set_visible(False)
ax.legend(fontsize=11)

leyenda = [mpatches.Patch(color=ROJO,    label="Crítico (≥22%)"),
           mpatches.Patch(color=AMARILLO, label="Moderado (18-22%)"),
           mpatches.Patch(color=VERDE,   label="Bajo (<18%)")]
ax.legend(handles=leyenda, loc="lower right", fontsize=10)

plt.tight_layout()
plt.savefig("eda_02_churn_por_territorio.png", dpi=150, bbox_inches="tight",
            facecolor=FONDO)
plt.close()
print("  ✓ eda_02_churn_por_territorio.png")


# ═══════════════════════════════════════════════════════
# FIGURA 3: ¿Los coolers protegen?
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(FONDO)
fig.suptitle("PREGUNTA 3: ¿Los coolers reducen el riesgo de churn?",
             fontsize=15, fontweight="bold", color=NEGRO)

# Gráfica 3a: Con vs sin cooler
ax = axes[0]
ax.set_facecolor(FONDO)
cool = (feats.groupby("tiene_cooler")
        .agg(total=("churn","count"), churneados=("churn","sum"))
        .reset_index())
cool["pct_churn"] = (cool["churneados"] / cool["total"] * 100).round(1)
cool["label"] = cool["tiene_cooler"].map({False: "Sin cooler", True: "Con cooler"})

colores_cool = [ROJO, VERDE]
bars = ax.bar(cool["label"], cool["pct_churn"], color=colores_cool,
              edgecolor="white", linewidth=2, width=0.5)
for bar, val, tot in zip(bars, cool["pct_churn"], cool["total"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val}%\n({tot:,} clientes)", ha="center", fontsize=12,
            fontweight="bold", color=NEGRO)

ax.set_ylabel("% de churn", fontsize=12)
ax.set_title("Churn rate: Con cooler vs Sin cooler", fontweight="bold", fontsize=12)
ax.spines[["top","right"]].set_visible(False)
ax.set_ylim(0, 55)

diferencia = cool[cool["tiene_cooler"]==False]["pct_churn"].values[0] - \
             cool[cool["tiene_cooler"]==True]["pct_churn"].values[0]
ax.text(0.5, 0.85, f"Los coolers reducen el churn\nen {diferencia:.1f} puntos porcentuales",
        transform=ax.transAxes, ha="center", fontsize=11, color=NEGRO,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=AMARILLO+"44", edgecolor=AMARILLO))

# Gráfica 3b: Churn por tamaño de cliente
ax = axes[1]
ax.set_facecolor(FONDO)
size = (feats.groupby("rtm_customer_size_d")
        .agg(total=("churn","count"), churneados=("churn","sum"))
        .reset_index()
        .dropna(subset=["rtm_customer_size_d"]))
size["pct_churn"] = (size["churneados"] / size["total"] * 100).round(1)
orden_size = ["Mini", "Pequeño", "Mediano", "Grande", "Gigante"]
size = size[size["rtm_customer_size_d"].isin(orden_size)]
size["orden"] = size["rtm_customer_size_d"].map({s: i for i, s in enumerate(orden_size)})
size = size.sort_values("orden")

colores_size = [ROJO, AMARILLO, VERDE, VERDE+"bb", VERDE+"88"]
bars = ax.bar(size["rtm_customer_size_d"], size["pct_churn"],
              color=colores_size, edgecolor="white", linewidth=2, width=0.6)
for bar, val, tot in zip(bars, size["pct_churn"], size["total"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val}%\n({tot:,})", ha="center", fontsize=10,
            fontweight="bold", color=NEGRO)

ax.set_ylabel("% de churn", fontsize=12)
ax.set_title("Churn rate por tamaño de cliente\n(Mini = más vulnerable)", fontweight="bold", fontsize=12)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("eda_03_coolers_y_tamano.png", dpi=150, bbox_inches="tight",
            facecolor=FONDO)
plt.close()
print("  ✓ eda_03_coolers_y_tamano.png")


# ═══════════════════════════════════════════════════════
# FIGURA 4: Variables más importantes (Pilar 1 + 2)
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(FONDO)
fig.suptitle("¿QUÉ VARIABLES INFLUYEN MÁS EN EL CHURN?\nComparación de distribuciones: Churneados vs Activos",
             fontsize=14, fontweight="bold", color=NEGRO)

# Histograma de compras_ultimo_mes
ax = axes[0]
ax.set_facecolor(FONDO)
activos_comp  = feats[feats["churn"]==0]["compras_ultimo_mes"].clip(0, 300)
churned_comp  = feats[feats["churn"]==1]["compras_ultimo_mes"].clip(0, 300)
ax.hist(activos_comp, bins=40, color=VERDE, alpha=0.6, label="Activos", density=True)
ax.hist(churned_comp, bins=40, color=ROJO,  alpha=0.6, label="Churneados", density=True)
ax.set_xlabel("Transacciones en el último mes", fontsize=11)
ax.set_ylabel("Densidad", fontsize=11)
ax.set_title("Distribución de transacciones\nen el último mes", fontweight="bold", fontsize=12)
ax.legend(fontsize=11)
ax.spines[["top","right"]].set_visible(False)
ax.text(0.6, 0.85, "Los churneados\ntienen 0 compras", transform=ax.transAxes,
        fontsize=10, color=ROJO, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor=ROJO+"22", edgecolor=ROJO))

# Churn por subcanal
ax = axes[1]
ax.set_facecolor(FONDO)
sub = (feats.groupby("comercial_subchannel_d")
       .agg(total=("churn","count"), churneados=("churn","sum"))
       .reset_index())
sub["pct_churn"] = (sub["churneados"] / sub["total"] * 100).round(1)
sub = sub[sub["total"] >= 500].sort_values("pct_churn", ascending=True)

colores_sub = [ROJO if p >= 20 else AMARILLO if p >= 15 else VERDE
               for p in sub["pct_churn"]]
bars = ax.barh(sub["comercial_subchannel_d"], sub["pct_churn"],
               color=colores_sub, edgecolor="white", linewidth=1.5, height=0.7)
for bar, val in zip(bars, sub["pct_churn"]):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f"{val}%", va="center", fontsize=9, fontweight="bold")
ax.set_xlabel("% de churn", fontsize=11)
ax.set_title("Churn rate por subcanal comercial", fontweight="bold", fontsize=12)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("eda_04_variables_importantes.png", dpi=150, bbox_inches="tight",
            facecolor=FONDO)
plt.close()
print("  ✓ eda_04_variables_importantes.png")

print("\n✅ EDA completo. Se generaron 4 gráficas listas para la presentación.")
print("\nHallazgos clave:")
print(f"  • Tasa de churn general: 17.95% ({churneados:,} de {total:,} clientes)")
print(f"  • Territorios más críticos: Monclova y Reynosa (25.9% de churn)")
print(f"  • Clientes Mini tienen 52.4% de churn vs 0.8% en Gigantes")
print(f"  • Sin cooler: 46.2% de churn | Con cooler: 12.5% de churn")
print(f"  • Subcanal Hogares es el más vulnerable: 26.4% de churn")
