import argparse
import os

import duckdb


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TPCH tables with DuckDB and write parquet")
    parser.add_argument("--output", default="data/parquet/tpch", help="Output parquet root")
    parser.add_argument("--sf", type=float, default=1.0, help="TPC-H scale factor")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL tpch")
    con.execute("LOAD tpch")
    con.execute(f"CALL dbgen(sf={args.sf})")

    for table in ["customer", "orders"]:
        out_path = os.path.join(args.output, table)
        # Overwrite target directories by writing to parquet files first, then using parquet reader.
        # DuckDB COPY writes files; we keep one file per table under expected folder.
        os.makedirs(out_path, exist_ok=True)
        outfile = os.path.join(out_path, "part-00000.parquet")
        con.execute(f"COPY (SELECT * FROM {table}) TO '{outfile}' (FORMAT PARQUET)")
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"wrote {table}: {count} rows -> {outfile}")

    con.close()


if __name__ == "__main__":
    main()
