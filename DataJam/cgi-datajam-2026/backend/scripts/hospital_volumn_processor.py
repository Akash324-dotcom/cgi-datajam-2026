from pathlib import Path
import sys
import pandas as pd
import re

sys.path.append(str(Path(__file__).resolve().parents[1]))
from services.data_loader import load_csv


# time filter
def filter_since(df: pd.DataFrame, cutoff: str = "2023-01-01") -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df[df["Date"] >= cutoff].copy()

""""
    row is invalid if
    1. Measure Name = "Emergency Visits CTAS" and CTAS value is not numerical or in 1-5
    2. Measure Name != "Emergency Visits CTAS" or "Emergency Visits"
""" 
def drop_invalid_ctas_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ctas_numeric = pd.to_numeric(df["CTAS"], errors="coerce")
    allowed_measures = {"Emergency Visits CTAS", "Emergency Visits"}
    is_allowed_measure = df["Measure Name"].isin(allowed_measures)
    is_emergency_ctas_measure = df["Measure Name"].eq("Emergency Visits CTAS")
    invalid_ctas = ~ctas_numeric.between(1, 5, inclusive="both")
    return df[is_allowed_measure & ~(is_emergency_ctas_measure & invalid_ctas)].copy()


# sort
def sort_by_hospital(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = df.copy()
    sorted_df["Hospital"] = sorted_df["Hospital"].astype(str).str.strip()
    return sorted_df.sort_values(
        by=["Hospital", "Date", "Measure Name", "CTAS"],
        kind="stable",
    ).reset_index(drop=True)


def normalize_actual_to_int(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = df.copy()

    def parse_actual(value: object) -> int:
        if pd.isna(value):
            return 0
        if isinstance(value, (int, float)):
            return int(round(value))

        text = str(value).strip()
        if not text:
            return 0

        # Remove wrapping quotes and thousands separators, then validate numeric format.
        text = text.strip("\"'")
        text = text.replace(",", "")
        if re.fullmatch(r"[+-]?\d+(\.\d+)?", text):
            return int(round(float(text)))
        return 0

    normalized_df["Actual"] = normalized_df["Actual"].map(parse_actual).astype(int)
    return normalized_df


def aggregate_hospital_month_rows(df: pd.DataFrame) -> pd.DataFrame:
    aggregated_df = df.copy()
    aggregated_df["CTAS"] = pd.to_numeric(aggregated_df["CTAS"], errors="coerce")

    emergency_totals = aggregated_df[
        aggregated_df["Measure Name"] == "Emergency Visits"
    ].copy()
    emergency_totals = (
        emergency_totals.groupby(
            ["Zone", "Hospital", "Type", "Date", "Measure Name"],
            as_index=False,
            dropna=False,
        )["Actual"]
        .sum()
    )
    emergency_totals["CTAS"] = pd.NA

    emergency_ctas = aggregated_df[
        aggregated_df["Measure Name"] == "Emergency Visits CTAS"
    ].copy()
    emergency_ctas = (
        emergency_ctas.groupby(
            ["Zone", "Hospital", "Type", "Date", "Measure Name", "CTAS"],
            as_index=False,
            dropna=False,
        )["Actual"]
        .sum()
    )
    # Ensure each hospital-month has CTAS 1..5; missing values are filled with 0.
    base_keys = emergency_totals[["Zone", "Hospital", "Type", "Date"]].drop_duplicates()
    ctas_values = pd.DataFrame({"CTAS": [1.0, 2.0, 3.0, 4.0, 5.0]})
    base_keys["_k"] = 1
    ctas_values["_k"] = 1
    expected_ctas = base_keys.merge(ctas_values, on="_k").drop(columns=["_k"])
    expected_ctas["Measure Name"] = "Emergency Visits CTAS"
    emergency_ctas = expected_ctas.merge(
        emergency_ctas,
        on=["Zone", "Hospital", "Type", "Date", "Measure Name", "CTAS"],
        how="left",
    )
    emergency_ctas["Actual"] = emergency_ctas["Actual"].fillna(0)

    combined = pd.concat([emergency_totals, emergency_ctas], ignore_index=True)
    combined["Actual"] = combined["Actual"].astype(int)
    return combined


def to_ctas_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    ctas_df = df[df["Measure Name"] == "Emergency Visits CTAS"].copy()
    ctas_df["CTAS"] = pd.to_numeric(ctas_df["CTAS"], errors="coerce").astype("Int64")
    ctas_df["Date"] = pd.to_datetime(ctas_df["Date"], errors="coerce").dt.strftime("%Y-%m")

    wide_df = (
        ctas_df.pivot_table(
            index=["Hospital", "Date"],
            columns="CTAS",
            values="Actual",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    for ctas_value in [1, 2, 3, 4, 5]:
        if ctas_value not in wide_df.columns:
            wide_df[ctas_value] = 0

    wide_df = wide_df.rename(
        columns={1: "CTAS1", 2: "CTAS2", 3: "CTAS3", 4: "CTAS4", 5: "CTAS5"}
    )
    wide_df = wide_df[["Hospital", "Date", "CTAS1", "CTAS2", "CTAS3", "CTAS4", "CTAS5"]]
    for col in ["CTAS1", "CTAS2", "CTAS3", "CTAS4", "CTAS5"]:
        wide_df[col] = wide_df[col].astype(int)

    wide_df["total_ER"] = (
        wide_df["CTAS1"]
        + wide_df["CTAS2"]
        + wide_df["CTAS3"]
        + wide_df["CTAS4"]
        + wide_df["CTAS5"]
    )
    wide_df["low_severity"] = wide_df["CTAS4"] + wide_df["CTAS5"]
    wide_df["low_ratio"] = (
        wide_df["low_severity"] / wide_df["total_ER"]
    ).where(wide_df["total_ER"] != 0, 0.0)
    wide_df["low_ratio"] = wide_df["low_ratio"].round(2)

    expected_months = wide_df["Date"].nunique()
    complete_hospitals = (
        wide_df.groupby("Hospital")["Date"].nunique().loc[lambda s: s == expected_months].index
    )
    wide_df = wide_df[wide_df["Hospital"].isin(complete_hospitals)].copy()

    return wide_df.sort_values(["Hospital", "Date"], kind="stable").reset_index(drop=True)


def main() -> None:
    df = load_csv("Hospital_Service_Volumes_20260306.csv")

    if df.empty:
        print("Loaded file, but it contains 0 rows.")
        return

    print(f"Loaded OK: {len(df)} rows, {len(df.columns)} columns")
    print("Columns:", list(df.columns))

    # drop rows during covid
    before_count = len(df)
    df = filter_since(df, "2023-01-01")
    print(f"Filtered Date >= 2023-01-01: {before_count} -> {len(df)} rows")

    # drop rows with measure name = "Emergency Visits CTAS" and CTAS = "Blank" or any non numeric value
    before_count = len(df)
    df = drop_invalid_ctas_rows(df)
    print(
        "Dropped invalid measure/CTAS rows: "
        f"{before_count} -> {len(df)} rows"
    )
    
    # for all actual values in the dataset, make them int. ie: some values are marked with "", reduce it.
    df = normalize_actual_to_int(df)

    # output a processed dataset to processed folder
    output_dir = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / "hospital_service_volume"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "Hospital_Service_Volumes_processed.csv"
    # df.to_csv(output_path, index=False)
    # print(f"Saved processed dataset: {output_path}")

    # sort by hospital names
    sorted_output_path = (
        output_dir / "Hospital_Service_Volumes_Processed_20260306.csv"
    )
    sorted_df = sort_by_hospital(aggregate_hospital_month_rows(df))
    sorted_df.to_csv(sorted_output_path, index=False)
    print(f"Saved sorted dataset: {sorted_output_path}")

    ctas_wide_output_path = (
        output_dir / "Hospital_Service_Volumes_CTAS_Wide_20260306.csv"
    )
    ctas_wide_df = to_ctas_wide_format(sorted_df)
    ctas_wide_df.to_csv(ctas_wide_output_path, index=False)
    print(f"Saved CTAS wide dataset: {ctas_wide_output_path}")

    

if __name__ == "__main__":
    main()
