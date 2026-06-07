# Proyecto: Churn Hunters - Arca Continental
# Autor: Pink Code
# Descripción: Sistema para identificar clientes en riesgo de dejar de comprar
# Datos: ventas mensuales de tienditas del canal tradicional

import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
import sys

# Configuración (conecxion con Mongo DB)
MONGO = "mongodb://localhost:27017/"
DB = "arca_churn"
COLECCION = "clientes_scoring"
COLECCION_LOG = "scoring_log"

# Rutas de los archivos (deben estar en la misma carpeta)
TRAIN = "sales_churn_train.csv"
TEST = "sales_churn_test.csv"
SUBMISSION = "preds_submission.csv"
CLIENTES = "Clientes.csv"
COOLERS = "Coolers.csv"


# -------------------------------------------------------------------
# PASO 1: Cargar los datos
# -------------------------------------------------------------------
def cargar_datos():
    print("Cargando archivos...")

    train = pd.read_csv(TRAIN, low_memory=False)
    test = pd.read_csv(TEST, low_memory=False)
    clientes = pd.read_csv(CLIENTES, low_memory=False)
    coolers = pd.read_csv(COOLERS, low_memory=False)

    print(f"  train: {len(train):,} filas")
    print(f"  test: {len(test):,} filas")
    print(f"  clientes: {len(clientes):,} filas")
    print(f"  coolers: {len(coolers):,} filas")

    return train, test, clientes, coolers


# -------------------------------------------------------------------
# PASO 2: Calcular variables por cliente
# Básicamente resumir el historial de cada tiendita en números
# -------------------------------------------------------------------
def calcular_variables(ventas, clientes, coolers, nombre):
    print(f"\nCalculando variables para: {nombre}...")

    ventas = ventas.copy()
    ventas["calmonth"] = ventas["calmonth"].astype(str)
    ventas = ventas.sort_values(["customer_id", "calmonth"])

    g = ventas.groupby("customer_id")

    # Función para ver si el cliente está cayendo o subiendo
    def tendencia(x):
        vals = x.values
        if len(vals) >= 2:
            return float(vals[-1] - vals[-2])
        return 0.0

    # Cuántos meses seguidos sin comprar (para detectar churn inminente)
    def racha_sin_compra(x):
        count = 0
        for v in reversed(x.values):
            if v == 0:
                count += 1
            else:
                break
        return count

    # Armar tabla de variables
    tabla = pd.DataFrame()
    tabla["compras_ultimo_mes"]     = g["num_transacciones"].last()
    tabla["compras_penultimo_mes"]  = g["num_transacciones"].apply(
                                        lambda x: x.values[-2] if len(x) >= 2 else x.values[-1])
    tabla["compras_promedio"]       = g["num_transacciones"].mean().round(2)
    tabla["cambio_compras"]         = g["num_transacciones"].apply(tendencia)
    tabla["cajas_ultimo_mes"]       = g["uni_boxes_sold_m"].last().round(2)
    tabla["cajas_promedio"]         = g["uni_boxes_sold_m"].mean().round(2)
    tabla["cambio_cajas"]           = g["uni_boxes_sold_m"].apply(tendencia)
    tabla["meses_en_datos"]         = g["calmonth"].nunique()
    tabla["meses_sin_compra"]       = g["num_transacciones"].apply(lambda x: (x == 0).sum())
    tabla["racha_sin_compra"]       = g["num_transacciones"].apply(racha_sin_compra)
    tabla["ultimo_mes"]             = g["calmonth"].last()

    # Proporción de meses inactivos
    tabla["pct_inactivo"] = (
        tabla["meses_sin_compra"] / tabla["meses_en_datos"].clip(lower=1)
    ).round(4)

    # Target real (solo existe en train)
    if "target" in ventas.columns:
        tabla["churn_real"] = g["target"].last()
    else:
        tabla["churn_real"] = None

    tabla = tabla.reset_index()
    tabla["conjunto"] = nombre

    # Agregar info del cliente (territorio, tamaño, etc.)
    clientes_u = clientes.drop_duplicates("customer_id")
    tabla = tabla.merge(clientes_u, on="customer_id", how="left")

    # Agregar coolers (tomo el último registro de cada cliente)
    coolers_ultimo = (
        coolers.sort_values("calmonth")
        .groupby("customer_id").last()
        .reset_index()
        [["customer_id", "num_coolers", "num_doors"]]
    )
    tabla = tabla.merge(coolers_ultimo, on="customer_id", how="left")
    tabla["num_coolers"] = tabla["num_coolers"].fillna(0).astype(int)
    tabla["num_doors"]   = tabla["num_doors"].fillna(0).astype(int)

    print(f"  {len(tabla):,} clientes procesados")
    return tabla


# -------------------------------------------------------------------
# PASO 3: Asignar el semáforo de riesgo
#
# Definición oficial de churn: 1 mes sin comprar
# El semáforo lo definí basándome en esa regla y en los datos:
#   - Los clientes que sí churnearon tienen compras_ultimo_mes ≈ 0
#   - La caída promedio antes del churn es de ~10 transacciones
# -------------------------------------------------------------------
def asignar_semaforo(fila):
    compras     = fila["compras_ultimo_mes"]
    cambio      = fila["cambio_compras"]
    pct         = fila["pct_inactivo"]
    racha       = fila["racha_sin_compra"]
    churn_real  = fila["churn_real"]

    # ⚫ Ya lo perdimos
    if churn_real == 1:
        return "Riesgo Inminente", "Confirmado como churneado (target=1)"
    if compras == 0 and racha >= 2:
        return "Riesgo Inminente", "Sin compras por 2 meses seguidos"

    # 🔴 Está muy cerca de irse
    if compras == 0:
        return "Riesgo Alto", "No compró el último mes"
    if pct >= 0.20:
        return "Riesgo Alto", f"{pct:.0%} de su historia sin ventas"

    # 🟡 Señales de que algo no está bien
    if cambio <= -10:
        return "Riesgo Moderado", f"Bajó {abs(cambio):.0f} compras vs mes anterior"
    if pct >= 0.05:
        return "Riesgo Moderado", f"{pct:.0%} de meses sin ventas"

    # 🟢 Todo bien
    return "Riesgo Bajo", "Cliente activo y estable"


def aplicar_semaforo(df):
    print("\nAplicando semáforo...")

    resultados = df.apply(asignar_semaforo, axis=1)
    df["nivel_riesgo"] = resultados.apply(lambda x: x[0])
    df["razon"]        = resultados.apply(lambda x: x[1])

    # Número para poder ordenar del más crítico al menos
    orden = {"Riesgo Bajo": 1, "Riesgo Moderado": 2,
             "Riesgo Alto": 3, "Riesgo Inminente": 4}
    df["orden"] = df["nivel_riesgo"].map(orden)

    print("\nResultados del semáforo:")
    emojis = {"Riesgo Bajo": "🟢", "Riesgo Moderado": "🟡",
               "Riesgo Alto": "🔴", "Riesgo Inminente": "⚫"}
    total = len(df)
    for nivel in ["Riesgo Bajo", "Riesgo Moderado", "Riesgo Alto", "Riesgo Inminente"]:
        n = (df["nivel_riesgo"] == nivel).sum()
        print(f"  {emojis[nivel]} {nivel:<20} {n:>8,}  ({n/total:.1%})")

    return df


# -------------------------------------------------------------------
# PASO 4: Guardar en MongoDB
# -------------------------------------------------------------------
def guardar_mongo(df):
    print(f"\nConectando a MongoDB...")
    cliente = MongoClient(MONGO, serverSelectionTimeoutMS=5000)

    try:
        cliente.admin.command("ping")
        print("  Conexión ok ✓")
    except Exception as e:
        print(f"  Error al conectar: {e}")
        sys.exit(1)

    db  = cliente[DB]
    col = db[COLECCION]
    log = db[COLECCION_LOG]

    # Índices para que las búsquedas sean rápidas
    col.create_index("customer_id", unique=True)
    col.create_index("nivel_riesgo")
    col.create_index("territory_d")

    ahora = datetime.utcnow()
    docs = df.replace({np.nan: None}).to_dict(orient="records")

    # Guardar en lotes de 1000
    escritos = 0
    print(f"\nGuardando {len(docs):,} clientes en MongoDB...")
    for i in tqdm(range(0, len(docs), 1000), desc="  Progreso"):
        lote = docs[i:i+1000]
        ops = [
            UpdateOne(
                {"customer_id": d["customer_id"]},
                {"$set": {**d, "updated_at": ahora}},
                upsert=True
            )
            for d in lote
        ]
        try:
            r = col.bulk_write(ops, ordered=False)
            escritos += r.upserted_count + r.modified_count
        except BulkWriteError as e:
            print(f"  Error parcial: {e.details}")

    # Guardar un log de esta corrida
    log.insert_one({
        "fecha"          : ahora,
        "total"          : len(df),
        "train"          : int((df["conjunto"] == "train").sum()),
        "test"           : int((df["conjunto"] == "test").sum()),
        "distribucion"   : df["nivel_riesgo"].value_counts().to_dict(),
        "docs_escritos"  : escritos,
    })

    print(f"\n  ✓ {escritos:,} documentos guardados")
    cliente.close()


# -------------------------------------------------------------------
# PASO 5: Consultas para responder las preguntas del reto
# -------------------------------------------------------------------
def consultas(uri, db_name):
    print("\n" + "="*55)
    print("  PREGUNTAS CLAVE DEL RETO — Arca Continental")
    print("="*55)

    c   = MongoClient(uri)
    col = c[db_name][COLECCION]

    campos = {"_id": 0, "customer_id": 1, "nivel_riesgo": 1,
              "territory_d": 1, "num_coolers": 1, "razon": 1,
              "compras_ultimo_mes": 1, "cambio_compras": 1}

    # Pregunta 1: ¿Qué clientes están en mayor riesgo?
    print("\n⚫ Top 10 clientes en RIESGO INMINENTE:")
    for r in col.find({"nivel_riesgo": "Riesgo Inminente"}, campos).limit(10):
        print(f"  {r['customer_id'][:20]}... | {r.get('territory_d','?'):<20} | {r.get('razon','')}")

    # Pregunta 2: ¿El territorio influye?
    print("\n📍 Riesgo por territorio (top 8 más críticos):")
    pipe = [
        {"$group": {
            "_id": "$territory_d",
            "total"     : {"$sum": 1},
            "inminente" : {"$sum": {"$cond": [{"$eq": ["$nivel_riesgo", "Riesgo Inminente"]}, 1, 0]}},
            "alto"      : {"$sum": {"$cond": [{"$eq": ["$nivel_riesgo", "Riesgo Alto"]}, 1, 0]}},
        }},
        {"$addFields": {
            "pct_criticos": {"$round": [
                {"$multiply": [{"$divide": [{"$add": ["$inminente","$alto"]}, "$total"]}, 100]}, 1
            ]}
        }},
        {"$sort": {"pct_criticos": -1}},
        {"$limit": 8}
    ]
    for r in col.aggregate(pipe):
        barra = "█" * int(r["pct_criticos"] / 5)
        print(f"  {str(r['_id']):<25} {r['pct_criticos']:>5.1f}%  {barra}")

    # Pregunta 3: ¿Los coolers ayudan?
    print("\n🧊 ¿Los coolers reducen el churn?")
    pipe2 = [
        {"$group": {
            "_id"      : {"$cond": [{"$gt": ["$num_coolers", 0]}, "Con coolers", "Sin coolers"]},
            "total"    : {"$sum": 1},
            "inminente": {"$sum": {"$cond": [{"$eq": ["$nivel_riesgo", "Riesgo Inminente"]}, 1, 0]}},
            "alto"     : {"$sum": {"$cond": [{"$eq": ["$nivel_riesgo", "Riesgo Alto"]}, 1, 0]}},
        }},
        {"$addFields": {
            "pct_riesgo": {"$round": [
                {"$multiply": [{"$divide": [{"$add": ["$inminente","$alto"]}, "$total"]}, 100]}, 1
            ]}
        }}
    ]
    for r in col.aggregate(pipe2):
        print(f"  {r['_id']:<15} | clientes: {r['total']:>8,} | en riesgo alto/inminente: {r['pct_riesgo']}%")

    # Resumen general
    print("\n📊 Resumen final:")
    total = col.count_documents({})
    for nivel in ["Riesgo Inminente", "Riesgo Alto", "Riesgo Moderado", "Riesgo Bajo"]:
        n   = col.count_documents({"nivel_riesgo": nivel})
        pct = n / total * 100
        emj = {"Riesgo Inminente":"⚫","Riesgo Alto":"🔴",
               "Riesgo Moderado":"🟡","Riesgo Bajo":"🟢"}[nivel]
        print(f"  {emj} {nivel:<20} {n:>8,}  ({pct:.1f}%)")

    c.close()


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    print("=" * 50)
    print("  Churn Hunter — Arca Continental")
    print("=" * 50)

    # Cargar
    train, test, clientes, coolers = cargar_datos()

    # Variables por conjunto
    print("\n--- Procesando train ---")
    vars_train = calcular_variables(train, clientes, coolers, "train")

    print("\n--- Procesando test ---")
    vars_test = calcular_variables(test, clientes, coolers, "test")

    # Unir todo
    todo = pd.concat([vars_train, vars_test], ignore_index=True)
    print(f"\nTotal de clientes: {len(todo):,}")

    # Semáforo
    todo = aplicar_semaforo(todo)

    # Guardar en MongoDB
    guardar_mongo(todo)

    # Consultas del reto
    consultas(MONGO, DB)

    print("\n✅ Pipeline completado exitosamente.\n")


if __name__ == "__main__":
    main()
