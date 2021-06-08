import ray
import raydp


spark = raydp.init_spark('word_count',
                         num_executors=2,
                         executor_cores=2,
                         executor_memory='1G')

spark.sparkContext.setLogLevel("INFO")

text_file = spark.textFile("hdfs://hadoop-bdwg-g3-ns01/tmp/opal/wordcount.txt")

counts = text_file.flatMap(lambda line: line.split(" ")) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b)

counts.show()

counts.saveAsTextFile("hdfs://hadoop-bdwg-g3-ns01/tmp/opal/wordcount_results.txt")

raydp.stop_spark()