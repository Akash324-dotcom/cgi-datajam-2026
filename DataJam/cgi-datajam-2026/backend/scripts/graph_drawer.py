from pathlib import Path
import pandas as pd


def build_ctas_distribution(df: pd.DataFrame) -> dict[str, int]:
    return {
        "CTAS1": int(df["CTAS1"].sum()),
        "CTAS2": int(df["CTAS2"].sum()),
        "CTAS3": int(df["CTAS3"].sum()),
        "CTAS4": int(df["CTAS4"].sum()),
        "CTAS5": int(df["CTAS5"].sum()),
    }


def draw_bar_chart_svg(data: dict[str, int], low_level_ratio: float, output_path: Path) -> None:
    width, height = 900, 520
    margin_left, margin_right, margin_top, margin_bottom = 90, 40, 80, 90
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    max_value = max(data.values()) if data else 1

    bar_count = len(data)
    slot_width = chart_width / bar_count
    bar_width = slot_width * 0.6
    colors = ["#0b5fff", "#2a8bf2", "#36b9cc", "#7cc36b", "#f4a261"]

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    svg.append(
        '<text x="450" y="42" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
        'font-size="24" font-weight="700" fill="#222">Graph 1 - CTAS Distribution (Total ER)</text>'
    )
    svg.append(
        f'<text x="860" y="70" text-anchor="end" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="15" font-weight="600" fill="#333">Overall low-level ratio: {low_level_ratio:.2%}</text>'
    )

    # Axes
    x0, y0 = margin_left, height - margin_bottom
    x1, y1 = width - margin_right, margin_top
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#333" stroke-width="2"/>')
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333" stroke-width="2"/>')

    # Horizontal guide lines
    for i in range(6):
        y = y0 - (chart_height * i / 5)
        value = int(max_value * i / 5)
        svg.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="#e6e6e6" stroke-width="1"/>')
        svg.append(
            f'<text x="{x0 - 10}" y="{y + 5:.2f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="12" fill="#555">{value}</text>'
        )

    # Bars
    for idx, (label, value) in enumerate(data.items()):
        bar_h = 0 if max_value == 0 else (value / max_value) * chart_height
        x = margin_left + (idx * slot_width) + ((slot_width - bar_width) / 2)
        y = y0 - bar_h
        color = colors[idx % len(colors)]
        svg.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_h:.2f}" '
            f'fill="{color}" rx="3" ry="3"/>'
        )
        svg.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#222">{value}</text>'
        )
        svg.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{y0 + 24:.2f}" text-anchor="middle" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="14" fill="#222">{label}</text>'
        )

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def build_monthly_er_demand(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby("Date", as_index=False)["total_ER"]
        .sum()
        .sort_values("Date", kind="stable")
        .reset_index(drop=True)
    )
    monthly["total_ER"] = monthly["total_ER"].astype(int)
    return monthly


def draw_line_chart_svg(monthly_df: pd.DataFrame, output_path: Path) -> None:
    width, height = 980, 560
    margin_left, margin_right, margin_top, margin_bottom = 90, 50, 90, 100
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    y_min = int(monthly_df["total_ER"].min())
    y_max = int(monthly_df["total_ER"].max())
    y_pad = max(1000, int((y_max - y_min) * 0.1))
    axis_min = max(0, y_min - y_pad)
    axis_max = y_max + y_pad
    span = max(1, axis_max - axis_min)

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    svg.append(
        '<text x="490" y="42" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
        'font-size="24" font-weight="700" fill="#222">Graph 2 - ER Demand Over Time</text>'
    )
    svg.append(
        '<text x="490" y="68" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
        'font-size="15" fill="#444">Monthly ER visits remain consistently high across 2023.</text>'
    )

    x0, y0 = margin_left, height - margin_bottom
    x1, y1 = width - margin_right, margin_top
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#333" stroke-width="2"/>')
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333" stroke-width="2"/>')

    # Horizontal guide lines
    for i in range(6):
        y = y0 - (chart_height * i / 5)
        val = int(axis_min + (span * i / 5))
        svg.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="#e6e6e6" stroke-width="1"/>')
        svg.append(
            f'<text x="{x0 - 10}" y="{y + 5:.2f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="12" fill="#555">{val}</text>'
        )

    # Build points
    n = len(monthly_df)
    step_x = chart_width / max(1, (n - 1))
    points = []
    for i, row in monthly_df.iterrows():
        x = x0 + step_x * i
        y = y0 - ((row["total_ER"] - axis_min) / span) * chart_height
        points.append((x, y, row["Date"], int(row["total_ER"])))

    points_attr = " ".join(f"{x:.2f},{y:.2f}" for x, y, _, _ in points)
    svg.append(
        f'<polyline points="{points_attr}" fill="none" stroke="#0b5fff" stroke-width="3" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )

    for x, y, month, val in points:
        svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#0b5fff"/>')
        svg.append(
            f'<text x="{x:.2f}" y="{y - 10:.2f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="11" fill="#1f2937">{val}</text>'
        )
        svg.append(
            f'<text x="{x:.2f}" y="{y0 + 26:.2f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="11" fill="#222">{month}</text>'
        )

    svg.append(
        f'<text x="{(x0 + x1) / 2:.2f}" y="{height - 35}" text-anchor="middle" '
        'font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#333">Month</text>'
    )
    svg.append(
        f'<text x="22" y="{(y0 + y1) / 2:.2f}" text-anchor="middle" transform="rotate(-90 22 {(y0 + y1) / 2:.2f})" '
        'font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#333">ER visits</text>'
    )

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def build_hospital_low_acuity_ratio(df: pd.DataFrame) -> pd.DataFrame:
    hospital_df = (
        df.groupby("Hospital", as_index=False)[["low_severity", "total_ER"]]
        .sum()
        .copy()
    )
    hospital_df["low_ratio"] = (
        hospital_df["low_severity"] / hospital_df["total_ER"]
    ).where(hospital_df["total_ER"] != 0, 0.0)
    return hospital_df.sort_values("low_ratio", ascending=False, kind="stable").reset_index(drop=True)


def draw_low_acuity_by_hospital_svg(hospital_df: pd.DataFrame, output_path: Path) -> None:
    width, height = 1500, 760
    margin_left, margin_right, margin_top, margin_bottom = 90, 50, 90, 230
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    max_ratio = max(0.01, float(hospital_df["low_ratio"].max()))

    n = len(hospital_df)
    slot_width = chart_width / max(1, n)
    bar_width = slot_width * 0.7

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    svg.append(
        '<text x="750" y="42" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
        'font-size="24" font-weight="700" fill="#222">Graph 3 - Low-Acuity Ratio by Hospital</text>'
    )
    svg.append(
        '<text x="750" y="68" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
        'font-size="15" fill="#444">Low acuity = (CTAS4 + CTAS5) / Total ER visits</text>'
    )

    x0, y0 = margin_left, height - margin_bottom
    x1, y1 = width - margin_right, margin_top
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#333" stroke-width="2"/>')
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333" stroke-width="2"/>')

    for i in range(6):
        y = y0 - (chart_height * i / 5)
        pct = (max_ratio * i / 5) * 100
        svg.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="#e6e6e6" stroke-width="1"/>')
        svg.append(
            f'<text x="{x0 - 10}" y="{y + 5:.2f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="12" fill="#555">{pct:.0f}%</text>'
        )

    for i, row in hospital_df.iterrows():
        ratio = float(row["low_ratio"])
        bar_h = (ratio / max_ratio) * chart_height
        x = margin_left + (i * slot_width) + ((slot_width - bar_width) / 2)
        y = y0 - bar_h
        svg.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_h:.2f}" '
            'fill="#2a8bf2" rx="2" ry="2"/>'
        )
        svg.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{y - 6:.2f}" text-anchor="middle" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="10" fill="#222">{ratio:.2%}</text>'
        )

        label_x = x + bar_width / 2
        label_y = y0 + 8
        hospital_name = str(row["Hospital"]).replace("&", "&amp;")
        svg.append(
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" transform="rotate(55 {label_x:.2f} {label_y:.2f})" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="10" fill="#222">{hospital_name}</text>'
        )

    svg.append(
        f'<text x="{(x0 + x1) / 2:.2f}" y="{height - 28}" text-anchor="middle" '
        'font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#333">Hospital</text>'
    )
    svg.append(
        f'<text x="24" y="{(y0 + y1) / 2:.2f}" text-anchor="middle" transform="rotate(-90 24 {(y0 + y1) / 2:.2f})" '
        'font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#333">Low acuity %</text>'
    )

    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    input_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / "hospital_service_volume"
        / "Hospital_Service_Volumes_CTAS_Wide_20260306.csv"
    )
    df = pd.read_csv(input_path)
    ctas_totals = build_ctas_distribution(df)
    total_er = sum(ctas_totals.values())
    low_level_ratio = ((ctas_totals["CTAS4"] + ctas_totals["CTAS5"]) / total_er) if total_er else 0.0

    output_dir = Path(__file__).resolve().parents[1] / "data" / "processed" / "hospital_service_volume"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "Graph_1_CTAS_Distribution.svg"
    draw_bar_chart_svg(ctas_totals, low_level_ratio, output_path)
    monthly_df = build_monthly_er_demand(df)
    output_path_2 = output_dir / "Graph_2_ER_Demand_Over_Time.svg"
    draw_line_chart_svg(monthly_df, output_path_2)
    hospital_ratio_df = build_hospital_low_acuity_ratio(df)
    output_path_3 = output_dir / "Graph_3_Low_Acuity_Ratio_by_Hospital.svg"
    draw_low_acuity_by_hospital_svg(hospital_ratio_df, output_path_3)

    print("CTAS totals:", ctas_totals)
    print(f"Overall low-level ratio: {low_level_ratio:.4f}")
    print(f"Saved graph: {output_path}")
    print(f"Saved graph: {output_path_2}")
    print(f"Saved graph: {output_path_3}")


if __name__ == "__main__":
    main()
