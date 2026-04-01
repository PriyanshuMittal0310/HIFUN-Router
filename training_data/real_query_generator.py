"""Generate large, rigorous real-workload DSL query packs for training.

This generator emits high-cardinality variants across real datasets:
- LDBC SNB (graph + mixed graph/SQL)
- OGB graph datasets (pure traversal + traversal aggregations)
- JOB / IMDB (multi-join SQL templates)
- TPC-DS (analytic SQL templates)

Usage:
  python training_data/real_query_generator.py --scale aggressive
"""

import argparse
import copy
import json
import os
from typing import Dict, List


OUTPUT_DIR = "dsl/sample_queries"


def _write(name: str, queries: List[dict]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2)
    print(f"Wrote {len(queries)} queries -> {out_path}")


def _mk_filter(op_id: str, source: str, col: str, op: str, value, fields: List[str], deps=None):
    return {
        "op_id": op_id,
        "type": "FILTER",
        "source": source,
        "fields": fields,
        "predicate": {"column": col, "operator": op, "value": value},
        "depends_on": deps or [],
    }


def _mk_join(op_id: str, source: str, right_source: str, left_key: str, right_key: str, fields: List[str], join_type: str = "INNER", deps=None):
    return {
        "op_id": op_id,
        "type": "JOIN",
        "source": source,
        "fields": fields,
        "join": {
            "right_source": right_source,
            "left_key": left_key,
            "right_key": right_key,
            "join_type": join_type,
        },
        "depends_on": deps or [],
    }


def _mk_agg(op_id: str, source: str, group_by: List[str], funcs: List[Dict[str, str]], fields: List[str], deps=None):
    return {
        "op_id": op_id,
        "type": "AGGREGATE",
        "source": source,
        "fields": fields,
        "aggregate": {"group_by": group_by, "functions": funcs},
        "depends_on": deps or [],
    }


def generate_snb_queries(graph_source: str, intensity: str) -> List[dict]:
    start_ids = [1, 10, 100, 1000, 10000]
    hops = [1, 2, 3, 4] if intensity != "balanced" else [1, 2, 3]
    directions = ["OUT", "BOTH"]
    join_types = ["INNER", "LEFT"]

    if intensity == "aggressive":
        start_ids.extend([20000, 30000, 40000, 50000])

    queries: List[dict] = []
    qn = 0

    for sid in start_ids:
        for h in hops:
            for d in directions:
                qn += 1
                qid = f"q_snb_real_{qn:04d}"
                queries.append({
                    "query_id": qid,
                    "description": f"SNB traversal sid={sid}, hops={h}, dir={d}",
                    "operations": [
                        {
                            "op_id": "s1",
                            "type": "TRAVERSAL",
                            "source": graph_source,
                            "fields": ["id"],
                            "traversal": {
                                "start_vertex_filter": {"column": "id", "operator": "=", "value": sid},
                                "edge_label": "KNOWS",
                                "direction": d,
                                "max_hops": h,
                                "return_fields": ["id"],
                            },
                            "depends_on": [],
                        },
                        _mk_agg(
                            "s2",
                            "s1",
                            ["id"],
                            [{"func": "COUNT", "column": "id", "alias": "fanout"}],
                            ["id", "fanout"],
                            deps=["s1"],
                        ),
                    ],
                })

                for jt in join_types:
                    qn += 1
                    qid2 = f"q_snb_real_{qn:04d}"
                    queries.append({
                        "query_id": qid2,
                        "description": f"SNB traversal joined with messages ({jt})",
                        "operations": [
                            {
                                "op_id": "s1",
                                "type": "TRAVERSAL",
                                "source": graph_source,
                                "fields": ["id"],
                                "traversal": {
                                    "start_vertex_filter": {"column": "id", "operator": "=", "value": sid},
                                    "edge_label": "KNOWS",
                                    "direction": d,
                                    "max_hops": h,
                                    "return_fields": ["id"],
                                },
                                "depends_on": [],
                            },
                            _mk_join(
                                "s2",
                                "s1",
                                "messages",
                                "id",
                                "sender_id",
                                ["id", "message_id", "sender_id", "content_length"],
                                join_type=jt,
                                deps=["s1"],
                            ),
                            _mk_agg(
                                "s3",
                                "s2",
                                ["id"],
                                [
                                    {"func": "COUNT", "column": "message_id", "alias": "msg_count"},
                                    {"func": "AVG", "column": "content_length", "alias": "avg_len"},
                                ],
                                ["id", "msg_count", "avg_len"],
                                deps=["s2"],
                            ),
                        ],
                    })

    return queries


def generate_snb_bi_queries(intensity: str) -> List[dict]:
    # Person table exposes LocationCityId in current SNB conversion.
    city_ids = [1, 10, 50, 100, 500]
    min_lengths = [20, 50, 100, 200]
    join_types = ["INNER", "LEFT"] if intensity != "balanced" else ["INNER"]

    if intensity == "aggressive":
        city_ids.extend([1000, 5000, 10000])
        min_lengths.extend([300, 400])

    queries: List[dict] = []
    qn = 0
    for city_id in city_ids:
        for l in min_lengths:
            for jt in join_types:
                qn += 1
                queries.append({
                    "query_id": f"q_snb_bi_{qn:04d}",
                    "description": "SNB BI style person-message-work analytics",
                    "operations": [
                        _mk_filter(
                            "b1",
                            "person",
                            "LocationCityId",
                            "=",
                            city_id,
                            ["id", "LocationCityId"],
                        ),
                        _mk_join(
                            "b2",
                            "b1",
                            "works_at",
                            "id",
                            "person_id",
                                ["id", "organisation_id", "LocationCityId"],
                            join_type=jt,
                            deps=["b1"],
                        ),
                        _mk_join(
                            "b3",
                            "b2",
                            "messages",
                            "id",
                            "sender_id",
                            ["id", "organisation_id", "content_length"],
                            join_type=jt,
                            deps=["b2"],
                        ),
                        _mk_filter(
                            "b4",
                            "b3",
                            "content_length",
                            ">=",
                            l,
                            ["id", "organisation_id", "content_length"],
                            deps=["b3"],
                        ),
                        _mk_agg(
                            "b5",
                            "b4",
                            ["organisation_id"],
                            [
                                {"func": "COUNT", "column": "id", "alias": "msg_count"},
                                {"func": "AVG", "column": "content_length", "alias": "avg_len"},
                            ],
                            ["organisation_id", "msg_count", "avg_len"],
                            deps=["b4"],
                        ),
                    ],
                })
    return queries


def generate_ogb_queries(graph_source: str, intensity: str) -> List[dict]:
    seeds = [0, 1, 2, 10, 100, 1000, 10000]
    hops = [1, 2, 3, 4] if intensity != "balanced" else [1, 2, 3]
    directions = ["OUT", "BOTH"]

    if intensity == "aggressive":
        seeds.extend([20000, 50000, 90000, 120000])

    queries: List[dict] = []
    qn = 0
    for sid in seeds:
        for h in hops:
            for d in directions:
                qn += 1
                queries.append({
                    "query_id": f"q_ogb_real_{qn:04d}",
                    "description": f"OGB traversal sid={sid}, hops={h}, dir={d}",
                    "operations": [
                        {
                            "op_id": "g1",
                            "type": "TRAVERSAL",
                            "source": graph_source,
                            "fields": ["id"],
                            "traversal": {
                                "start_vertex_filter": {"column": "id", "operator": "=", "value": sid},
                                "edge_label": "KNOWS",
                                "direction": d,
                                "max_hops": h,
                                "return_fields": ["id"],
                            },
                            "depends_on": [],
                        },
                        _mk_agg(
                            "g2",
                            "g1",
                            ["id"],
                            [{"func": "COUNT", "column": "id", "alias": "reach"}],
                            ["id", "reach"],
                            deps=["g1"],
                        ),
                    ],
                })
    return queries


def generate_job_queries(intensity: str) -> List[dict]:
    # JOB CSV exports are converted to generic c0..cN columns.
    selectivities = [1000, 5000, 10000, 20000, 50000]

    if intensity == "aggressive":
        selectivities.extend([75000, 100000, 150000, 200000])

    templates = []
    for year in selectivities:
        templates.append({
            "query_id": f"q_job_real_y{year}_base",
            "description": "JOB: title table selectivity + aggregation",
            "operations": [
                _mk_filter("j1", "title", "c0", ">=", year, ["c0", "c1", "c2"]),
                _mk_agg("j2", "j1", ["c1"], [{"func": "COUNT", "column": "c0", "alias": "n_rows"}], ["c1", "n_rows"], deps=["j1"]),
            ],
        })

        templates.append({
            "query_id": f"q_job_kw_y{year}_base",
            "description": "JOB: keyword frequency aggregation",
            "operations": [
                _mk_filter("k1", "movie_keyword", "c0", ">=", year, ["c0", "c1", "c2"]),
                _mk_agg("k2", "k1", ["c2"], [{"func": "COUNT", "column": "c0", "alias": "cnt"}], ["c2", "cnt"], deps=["k1"]),
            ],
        })

    return templates


def generate_tpcds_queries(intensity: str) -> List[dict]:
    # Uses generic c0..cN names from data/scripts/tpcds_to_parquet.py conversion.
    filters = [1, 5, 10, 25, 50]

    if intensity == "aggressive":
        filters.extend([75, 100, 250])

    queries: List[dict] = []
    for f in filters:
        queries.append({
            "query_id": f"q_tpcds_real_{f}_base",
            "description": "TPC-DS style store_sales aggregation",
            "operations": [
                _mk_filter("t1", "store_sales", "c0", ">", f, ["c0", "c1", "c2", "c3"]),
                _mk_agg("t2", "t1", ["c1"], [{"func": "SUM", "column": "c3", "alias": "revenue"}], ["c1", "revenue"], deps=["t1"]),
            ],
        })

        queries.append({
            "query_id": f"q_tpcds_date_{f}_base",
            "description": "TPC-DS date_dim bucketing",
            "operations": [
                _mk_filter("d1", "date_dim", "c0", ">", f, ["c0", "c1", "c2"]),
                _mk_agg("d2", "d1", ["c1"], [{"func": "COUNT", "column": "c0", "alias": "n_days"}], ["c1", "n_days"], deps=["d1"]),
            ],
        })
    return queries


def generate_all(scale: str, ogb_graph_source: str) -> Dict[str, int]:
    if scale not in {"balanced", "aggressive"}:
        raise ValueError("scale must be one of: balanced, aggressive")

    snb = generate_snb_queries("snb", intensity=scale)
    snb_bi = generate_snb_bi_queries(intensity=scale)
    ogb = generate_ogb_queries(ogb_graph_source, intensity=scale)
    job = generate_job_queries(intensity=scale)
    tpcds = generate_tpcds_queries(intensity=scale)

    _write("snb_real_queries.json", snb)
    _write("snb_bi_real_queries.json", snb_bi)
    _write("ogb_real_queries.json", ogb)
    _write("job_real_queries.json", job)
    _write("tpcds_real_queries.json", tpcds)

    return {
        "snb_real_queries": len(snb),
        "snb_bi_real_queries": len(snb_bi),
        "ogb_real_queries": len(ogb),
        "job_real_queries": len(job),
        "tpcds_real_queries": len(tpcds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate real-world high-volume DSL query packs")
    parser.add_argument("--scale", default="aggressive", choices=["balanced", "aggressive"], help="Generation intensity")
    parser.add_argument(
        "--ogb-graph-source",
        default="ogbn_arxiv",
        help="Graph source name used in OGB traversal queries",
    )
    args = parser.parse_args()

    counts = generate_all(scale=args.scale, ogb_graph_source=args.ogb_graph_source)
    total = sum(counts.values())
    print("Generated query packs:")
    for name, n in counts.items():
        print(f"  {name}: {n}")
    print(f"Total queries: {total}")


if __name__ == "__main__":
    main()
