import pandas as pd
import polars as pl
from bs4 import BeautifulSoup


def parse_run_efforts_table(filename: str) -> pl.DataFrame:
    # --- read & parse --------------------------------------------------------
    with open(filename, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    table = soup.find("div", id="run-efforts-table").find("table")  # type: ignore

    # --- dynamically get column headers --------------------------------------
    header_cells = table.find("thead").find_all("th")  # type: ignore
    cols = [th.get_text(strip=True) for th in header_cells]

    # --- pull rows -----------------------------------------------------------
    rows = []
    for tr in table.select("tbody tr"):  # type: ignore
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) == len(cols):  # guard against malformed rows
            rows.append(cells)

    # --- build DataFrame -----------------------------------------------------
    df = pd.DataFrame(rows, columns=cols)

    # --- optional cleanup ----------------------------------------------------
    # Clean Lap column if present
    if "Lap" in df.columns:
        df["Lap"] = pd.to_numeric(df["Lap"], errors="coerce").astype("Float64")

    # Clean Distance if present
    if "Distance" in df.columns:
        df["Distance_km"] = (
            df["Distance"].str.extract(r"([\d.]+)").astype(float).squeeze()
        )
        df.drop(columns="Distance", inplace=True)

    # Clean HR if present
    if "HR" in df.columns:
        df["HR"] = pd.to_numeric(
            df["HR"].str.extract(r"(\d+)")[0], errors="coerce"
        )

    df["file"] = filename  # Add filename for reference

    if "/2.html" in filename:
        df["Lap"] = list(range(1, len(df) + 1))

    return pl.from_pandas(df)


def parse(
    folder: str = "run-efforts-table", istart: int = 1, istop: int = 35
) -> pl.DataFrame:
    return (
        pl.concat(
            [
                parse_run_efforts_table(f"{folder}/{i}.html").with_columns(
                    Activity=pl.lit(i)
                )
                for i in range(istart, istop + 1)
            ],
            how="diagonal_relaxed",
        )
    ).with_columns(
        pl.col("Time", "Pace", "GAP").map_elements(
            lambda s: (
                float(s.split(":")[0])
                + float(s.split(":")[1].replace("/km", "")) / 60
            )
            if "s" not in s
            else float(s.replace("s", "")) / 60,
            return_dtype=pl.Float64,
        ),
    )
