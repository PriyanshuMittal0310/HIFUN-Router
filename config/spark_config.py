"""Spark + Hadoop session factory for HIFUN Router.

Supports three execution modes:
  local   — single-process, no Hadoop (unit tests / quick experiments)
  cluster — Spark standalone cluster (docker-compose spark-master:7077)
  yarn    — YARN resource manager (full Hadoop stack)

Environment variables consumed:
  HIFUN_SPARK_MASTER   — override master URL (default: local[*])
  HIFUN_HDFS_ROOT      — HDFS root URI, e.g. hdfs://namenode:9000/hifun
  HIFUN_DRIVER_MEM     — driver memory (default: 4g)
  HIFUN_EXECUTOR_MEM   — executor memory (default: 4g)
  HIFUN_EXECUTOR_CORES — executor cores  (default: 2)
  HIFUN_HISTORY_SERVER — Spark History Server log dir (default: /tmp/spark-events)
"""

import os
import shutil
from pyspark.sql import SparkSession

# ─── GraphFrames JAR (Spark 3.4, Scala 2.12) ─────────────────────────────────
_GF_PACKAGE = "graphframes:graphframes:0.8.3-spark3.4-s_2.12"

# ─── Delta Lake (optional, for ACID table writes) ─────────────────────────────
_DELTA_PACKAGE = "io.delta:delta-core_2.12:2.4.0"

# ─── Read environment overrides ───────────────────────────────────────────────
_MASTER          = os.environ.get("HIFUN_SPARK_MASTER",   "local[*]")
_HDFS_ROOT       = os.environ.get("HIFUN_HDFS_ROOT",      "")
_DRIVER_MEM      = os.environ.get("HIFUN_DRIVER_MEM",     "4g")
_EXECUTOR_MEM    = os.environ.get("HIFUN_EXECUTOR_MEM",   "4g")
_EXECUTOR_CORES  = os.environ.get("HIFUN_EXECUTOR_CORES", "2")
_HISTORY_LOG_DIR = os.environ.get("HIFUN_HISTORY_SERVER", "/tmp/spark-events")


def _ensure_java_home() -> None:
    """Best-effort JAVA_HOME discovery for local Spark runs.

    PySpark frequently fails with a generic Java gateway error when JAVA_HOME is
    unset even if `java` is available on PATH. This helper infers JAVA_HOME
    from the resolved java binary path and sets it for the current process.
    """
    if os.environ.get("JAVA_HOME"):
        return

    java_bin = shutil.which("java")
    if not java_bin:
        # Common user-local JRE location used in this project environment.
        local_root = os.path.expanduser("~/.local/java")
        if os.path.isdir(local_root):
            candidates = sorted(
                [
                    os.path.join(local_root, d, "bin", "java")
                    for d in os.listdir(local_root)
                ]
            )
            for c in reversed(candidates):
                if os.path.exists(c):
                    java_bin = c
                    break
    if not java_bin:
        return

    real_java = os.path.realpath(java_bin)
    # Expected: /usr/lib/jvm/<jdk>/bin/java -> JAVA_HOME=/usr/lib/jvm/<jdk>
    parent = os.path.dirname(real_java)
    if os.path.basename(parent) == "bin":
        os.environ["JAVA_HOME"] = os.path.dirname(parent)


def get_spark_session(
    app_name: str = "HIFUN_Router",
    master: str = _MASTER,
    driver_memory: str = _DRIVER_MEM,
    executor_memory: str = _EXECUTOR_MEM,
    executor_cores: str = _EXECUTOR_CORES,
    enable_delta: bool = False,
    extra_configs: dict = None,
) -> SparkSession:
    """Build and return a configured SparkSession.

    Args:
        app_name:        Spark application name shown in the Spark UI.
        master:          Spark master URL.
                         - 'local[*]'            → all local cores
                         - 'local[4]'            → 4 local cores
                         - 'spark://host:7077'   → standalone cluster
                         - 'yarn'                → YARN cluster
        driver_memory:   JVM heap for the driver process.
        executor_memory: JVM heap per executor.
        executor_cores:  vCPU cores per executor.
        enable_delta:    If True, add Delta Lake JAR and configure the
                         Delta catalog extension.
        extra_configs:   Additional {key: value} Spark config overrides.

    Returns:
        A configured SparkSession (or the existing active session).
    """
    _ensure_java_home()

    packages = _GF_PACKAGE
    if enable_delta:
        packages = f"{packages},{_DELTA_PACKAGE}"

    # Spark fails hard if event logging is enabled and the directory is absent.
    event_log_enabled = True
    try:
        os.makedirs(_HISTORY_LOG_DIR, exist_ok=True)
    except OSError:
        event_log_enabled = False

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)

        # ── Core memory / parallelism ─────────────────────────────────────
        .config("spark.driver.memory",          driver_memory)
        .config("spark.executor.memory",        executor_memory)
        .config("spark.executor.cores",         executor_cores)

        # ── GraphFrames (and optionally Delta Lake) JARs ──────────────────
        .config("spark.jars.packages",          packages)

        # ── Shuffle / partitioning ────────────────────────────────────────
        .config("spark.sql.shuffle.partitions", "200")   # matches default HDFS block split

        # ── Adaptive Query Execution (AQE) — Spark 3.x ───────────────────
        .config("spark.sql.adaptive.enabled",                         "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled",      "true")
        .config("spark.sql.adaptive.coalescePartitions.minPartitionNum", "1")
        .config("spark.sql.adaptive.skewJoin.enabled",                "true")
        .config("spark.sql.adaptive.localShuffleReader.enabled",      "true")

        # ── Broadcast join threshold (10 MB default) ──────────────────────
        .config("spark.sql.autoBroadcastJoinThreshold",  str(10 * 1024 * 1024))

        # ── Parquet optimisations ─────────────────────────────────────────
        .config("spark.sql.parquet.filterPushdown",               "true")
        .config("spark.sql.parquet.mergeSchema",                  "false")
        .config("spark.hadoop.parquet.enable.summary-metadata",   "false")

        # ── Column statistics for better cardinality estimates ────────────
        .config("spark.sql.statistics.histogram.enabled",       "true")
        .config("spark.sql.statistics.fallBackToHdfs",          "true")

        # ── Dynamic resource allocation (YARN / standalone) ───────────────
        .config("spark.dynamicAllocation.enabled",              "false")  # stable for experiments
        .config("spark.executor.instances",                     "2")

        # ── Kryo serializer (faster than Java default) ────────────────────
        .config("spark.serializer",           "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "512m")

        # ── Event log (Spark History Server) ─────────────────────────────
        .config("spark.eventLog.enabled",  "true" if event_log_enabled else "false")
        .config("spark.eventLog.dir",      _HISTORY_LOG_DIR)
        .config("spark.history.fs.logDirectory", _HISTORY_LOG_DIR)

        # ── UI port ───────────────────────────────────────────────────────
        .config("spark.ui.port", "4040")
    )

    # HDFS / Hadoop integration
    if _HDFS_ROOT:
        namenode = _HDFS_ROOT.split("/")[2] if "://" in _HDFS_ROOT else ""
        if namenode:
            builder = builder.config(
                "spark.hadoop.fs.defaultFS", f"hdfs://{namenode}"
            )

    # Delta Lake catalog extension
    if enable_delta:
        builder = (
            builder
            .config("spark.sql.extensions",
                    "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog",
                    "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )

    # User overrides applied last
    if extra_configs:
        for key, value in extra_configs.items():
            builder = builder.config(key, value)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    return spark


def get_local_spark(app_name: str = "HIFUN_Local", cores: int = 4) -> SparkSession:
    """Convenience factory for fast local-mode sessions used in unit tests."""
    return get_spark_session(
        app_name=app_name,
        master=f"local[{cores}]",
        driver_memory="2g",
        executor_memory="2g",
        extra_configs={
            "spark.sql.shuffle.partitions": "4",
            "spark.eventLog.enabled": "false",
        },
    )


def get_hdfs_root() -> str:
    """Return the configured HDFS root URI, or empty string if not set."""
    return _HDFS_ROOT
