from config.spark_config import get_spark_session
import time

def test_environment():
    print("Initializing Spark Session...")
    print("(If this is your first run, it may take a minute to download GraphFrames...)")
    
    t0 = time.time()
    spark = get_spark_session()
    
    # 1. Test Spark Core
    print(f"\n✅ Success! Spark Version: {spark.version}")
    
    # 2. Test DataFrame creation
    data = [("Alice", 34), ("Bob", 45), ("Charlie", 28)]
    df = spark.createDataFrame(data, ["Name", "Age"])
    print("\n✅ Success! DataFrame created:")
    df.show()
    
    elapsed = time.time() - t0
    print(f"Test completed in {elapsed:.2f} seconds.")
    
    spark.stop()

if __name__ == "__main__":
    test_environment()