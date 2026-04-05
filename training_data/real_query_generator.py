"""Generate large, rigorous real-workload DSL query packs for training.

This generator emits high-cardinality variants across real datasets:
- LDBC SNB (graph + mixed graph/SQL)
- OGB graph datasets (pure traversal + traversal aggregations)
- JOB / IMDB (multi-join SQL templates)
- TPC-DS (analytic SQL templates)

Usage:
    python training_data/real_query_generator.py --scale aggressive --focus-mode all
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


def generate_snb_queries(graph_source: str, intensity: str, focus_mode: str) -> List[dict]:
    start_ids = [1, 10, 100, 1000, 10000]
    hops = [1, 2, 3, 4] if intensity != "balanced" else [1, 2, 3]
    directions = ["OUT", "BOTH"]
    join_types = ["INNER", "LEFT"]

    if intensity == "aggressive":
        start_ids.extend([20000, 30000, 40000, 50000])

    if focus_mode == "graph_win":
        # Deep traversal-heavy patterns with minimal relational joins.
        hops = [2, 3, 4, 5]
        directions = ["OUT", "BOTH"]
        join_types = []
    elif focus_mode == "sql_win":
        # Keep traversal shallow and emphasize mixed relational follow-ups.
        hops = [1, 2]

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

    # Add extra traversal-only stress queries in graph focus mode.
    if focus_mode == "graph_win":
        for sid in start_ids:
            for h in [4, 5, 6]:
                qn += 1
                queries.append({
                    "query_id": f"q_snb_real_{qn:04d}",
                    "description": f"SNB graph-focus deep traversal sid={sid}, hops={h}",
                    "operations": [
                        {
                            "op_id": "s1",
                            "type": "TRAVERSAL",
                            "source": graph_source,
                            "fields": ["id"],
                            "traversal": {
                                "start_vertex_filter": {"column": "id", "operator": "=", "value": sid},
                                "edge_label": "KNOWS",
                                "direction": "BOTH",
                                "max_hops": h,
                                "return_fields": ["id"],
                            },
                            "depends_on": [],
                        }
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


def generate_snb_bi_graph_queries(intensity: str) -> List[dict]:
    seeds = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000]
    hops = [2, 3, 4, 5] if intensity != "balanced" else [2, 3]
    directions = ["OUT", "BOTH"] if intensity != "balanced" else ["BOTH"]
    queries: List[dict] = []
    qn = 0
    for sid in seeds:
        for h in hops:
            for d in directions:
                qn += 1
                queries.append({
                    "query_id": f"q_snb_bi_graph_{qn:04d}",
                    "description": "SNB-BI graph-projection traversal",
                    "operations": [
                        {
                            "op_id": "bg1",
                            "type": "TRAVERSAL",
                            "source": "snb",
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
                            "bg2",
                            "bg1",
                            ["id"],
                            [{"func": "COUNT", "column": "id", "alias": "reach"}],
                            ["id", "reach"],
                            deps=["bg1"],
                        ),
                    ],
                })
    return queries


def generate_ogb_queries(graph_source: str, intensity: str, focus_mode: str) -> List[dict]:
    seeds = [0, 1, 2, 10, 100, 1000, 10000]
    hops = [1, 2, 3, 4] if intensity != "balanced" else [1, 2, 3]
    directions = ["OUT", "BOTH"]

    if intensity == "aggressive":
        seeds.extend([20000, 50000, 90000, 120000])

    if focus_mode == "graph_win":
        hops = [3, 4, 5]
        directions = ["OUT", "BOTH"]
    elif focus_mode == "sql_win":
        hops = [1, 2]

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
    if focus_mode == "graph_win":
        for sid in seeds[: min(12, len(seeds))]:
            for h in [5, 6]:
                qn += 1
                queries.append({
                    "query_id": f"q_ogb_real_{qn:04d}",
                    "description": f"OGB graph-focus deep traversal sid={sid}, hops={h}",
                    "operations": [
                        {
                            "op_id": "g1",
                            "type": "TRAVERSAL",
                            "source": graph_source,
                            "fields": ["id"],
                            "traversal": {
                                "start_vertex_filter": {"column": "id", "operator": "=", "value": sid},
                                "edge_label": "KNOWS",
                                "direction": "BOTH",
                                "max_hops": h,
                                "return_fields": ["id"],
                            },
                            "depends_on": [],
                        }
                    ],
                })

    return queries


def generate_job_queries(intensity: str, focus_mode: str) -> List[dict]:
    # JOB CSV exports are converted to generic c0..cN columns.
    selectivities = [1000, 5000, 10000, 20000, 50000]

    if intensity == "aggressive":
        selectivities.extend([75000, 100000, 150000, 200000])

    templates = []
    if focus_mode == "graph_win":
        # Keep SQL-heavy packs smaller in graph focus mode.
        selectivities = selectivities[:3]
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


def generate_job_graph_queries(intensity: str) -> List[dict]:
    seeds = [
        "movie_keyword:c0:1", "movie_keyword:c0:10", "movie_keyword:c0:100",
        "cast_info:c0:1", "cast_info:c0:10",
        "movie_companies:c0:1", "movie_companies:c0:10",
        "movie_link:c0:1", "movie_link:c0:10",
    ]
    hops = [2, 3, 4, 5] if intensity != "balanced" else [2, 3]
    directions = ["OUT", "BOTH"] if intensity != "balanced" else ["BOTH"]
    queries: List[dict] = []
    qn = 0
    for sid in seeds:
        for h in hops:
            for d in directions:
                qn += 1
                queries.append({
                    "query_id": f"q_job_graph_{qn:04d}",
                    "description": "JOB graph-projection traversal",
                    "operations": [
                        {
                            "op_id": "jg1",
                            "type": "TRAVERSAL",
                            "source": "job_real_queries",
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
                            "jg2",
                            "jg1",
                            ["id"],
                            [{"func": "COUNT", "column": "id", "alias": "reach"}],
                            ["id", "reach"],
                            deps=["jg1"],
                        ),
                    ],
                })
    return queries


def generate_tpcds_queries(intensity: str, focus_mode: str) -> List[dict]:
    # Uses generic c0..cN names from data/scripts/tpcds_to_parquet.py conversion.
    filters = [1, 5, 10, 25, 50]

    if intensity == "aggressive":
        filters.extend([75, 100, 250])

    if focus_mode == "graph_win":
        filters = filters[:3]

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


def generate_tpcds_graph_queries(intensity: str) -> List[dict]:
    seeds = [
        "store_sales:c0:1", "store_sales:c0:10", "store_sales:c0:100",
        "catalog_sales:c0:1", "catalog_sales:c0:10", "catalog_sales:c0:100",
        "web_sales:c0:1", "web_sales:c0:10", "web_sales:c0:100",
    ]
    hops = [2, 3, 4, 5] if intensity != "balanced" else [2, 3]
    directions = ["OUT", "BOTH"] if intensity != "balanced" else ["BOTH"]
    queries: List[dict] = []
    qn = 0
    for sid in seeds:
        for h in hops:
            for d in directions:
                qn += 1
                queries.append({
                    "query_id": f"q_tpcds_graph_{qn:04d}",
                    "description": "TPCDS graph-projection traversal",
                    "operations": [
                        {
                            "op_id": "tg1",
                            "type": "TRAVERSAL",
                            "source": "tpcds_real_queries",
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
                            "tg2",
                            "tg1",
                            ["id"],
                            [{"func": "COUNT", "column": "id", "alias": "reach"}],
                            ["id", "reach"],
                            deps=["tg1"],
                        ),
                    ],
                })
    return queries


def generate_all(
    scale: str,
    ogb_graph_source: str,
    focus_mode: str,
    include_sql_families_in_graph_focus: bool,
) -> Dict[str, int]:
    if scale not in {"balanced", "aggressive"}:
        raise ValueError("scale must be one of: balanced, aggressive")
    if focus_mode not in {"all", "graph_win", "sql_win"}:
        raise ValueError("focus_mode must be one of: all, graph_win, sql_win")

    snb = generate_snb_queries("snb", intensity=scale, focus_mode=focus_mode)
    snb_bi: List[dict] = []
    ogb = generate_ogb_queries(ogb_graph_source, intensity=scale, focus_mode=focus_mode)
    job: List[dict] = []
    tpcds: List[dict] = []

    if focus_mode != "graph_win" or include_sql_families_in_graph_focus:
        snb_bi = generate_snb_bi_queries(intensity=scale)
        job = generate_job_queries(intensity=scale, focus_mode=focus_mode)
        tpcds = generate_tpcds_queries(intensity=scale, focus_mode=focus_mode)

    if focus_mode in {"all", "graph_win"}:
        snb_bi.extend(generate_snb_bi_graph_queries(intensity=scale))
        job.extend(generate_job_graph_queries(intensity=scale))
        tpcds.extend(generate_tpcds_graph_queries(intensity=scale))

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
    parser.add_argument(
        "--focus-mode",
        default="all",
        choices=["all", "graph_win", "sql_win"],
        help="Bias generated workload families toward graph-win or sql-win patterns",
    )
    parser.add_argument(
        "--include-sql-families-in-graph-focus",
        action="store_true",
        help="In graph_win mode, also emit SNB-BI/JOB/TPCDS families (disabled by default)",
    )
    args = parser.parse_args()

    counts = generate_all(
        scale=args.scale,
        ogb_graph_source=args.ogb_graph_source,
        focus_mode=args.focus_mode,
        include_sql_families_in_graph_focus=args.include_sql_families_in_graph_focus,
    )
    total = sum(counts.values())
    print("Generated query packs:")
    for name, n in counts.items():
        print(f"  {name}: {n}")
    print(f"Total queries: {total}")


if __name__ == "__main__":
    main()
