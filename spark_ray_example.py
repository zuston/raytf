import ray
import raydp

ray.init(address='auto')

spark = raydp.init_spark('word_count',
                         num_executors=2,
                         executor_cores=2,
                         executor_memory='1G')

spark.sparkContext.setLogLevel("INFO")

spark.read.text("hdfs://namenode02-bdwg-g3.qiyi.hadoop/tmp/cloud_service/opal/wordcount.txt").show()

# 无法读取到响应的 ns 配置，Hadoop classpath 应该没引入
text_file = spark.textFile("hdfs://hadoop-bdwg-g3-ns01/tmp/cloud_service/opal/wordcount.txt")

counts = text_file.flatMap(lambda line: line.split(" ")) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b)

counts.show()

counts.saveAsTextFile("hdfs://hadoop-bdwg-g3-ns01/tmp/cloud_service/opal/wordcount_results.txt")

raydp.stop_spark()