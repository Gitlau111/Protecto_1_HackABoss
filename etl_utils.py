import requests
from datetime import datetime
import pandas as pd

def importar_datos(url, fecha_inicio=None, fecha_fin=None):
    resp = requests.get(url)
    resp.raise_for_status()
    datos = resp.json()

    # Si no hay rango, devuelvo todo ordenado
    if not fecha_inicio and not fecha_fin:
        return sorted(datos, key=lambda d: d["date"])

    fmt = "%Y%m%d"
    f_inicio = datetime.strptime(fecha_inicio, fmt) if fecha_inicio else None
    f_fin    = datetime.strptime(fecha_fin,    fmt) if fecha_fin    else None

    filtrados = []
    for r in datos:
        dt = datetime.strptime(str(r["date"]), fmt)

        if f_inicio and dt < f_inicio:
            continue
        if f_fin and dt > f_fin:
            continue

        filtrados.append(r)

    return sorted(filtrados, key=lambda d: d["date"])


def transformar(datos):
    import pandas as pd

    # Si no hay datos, devuelvo un df con las columnas mínimas
    if not datos:
        return pd.DataFrame(columns=["date", "death", "daily_deaths"])
    
    df = pd.DataFrame(datos)

    # Si por alguna razón viene sin 'date', salgo con un df vacío
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date", "death", "daily_deaths"])

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.sort_values("date")
    df["daily_deaths"] = df["death"].diff().fillna(0).astype(int)
    return df
