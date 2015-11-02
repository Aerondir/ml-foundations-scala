import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql.{DataFrame, SQLContext}


object Cells {
  val conf = new SparkConf().setMaster("local[*]")
    .setAppName("metricSeeker")
    .set("spark.executor.memory", "512m")
  val sc = new SparkContext(conf)


  sc.version

  /* ... new cell ... */
  //  :dp com.databricks % spark-csv_2.10 % 1.2.0

  /* ... new cell ... */
  val sqlContext = new SQLContext(sc)

  /* ... new cell ... */
  val mnt = "/Users/maxim.razumov/lab/ml-foundations-scala"

  /* ... new cell ... */
  val sqlContext = new SQLContext(sc)


  val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load(s"$mnt/data/people-example.csv")

  /* ... new cell ... */
  df //we can view first few lines of table

  /* ... new cell ... */
  df.head()

  /* ... new cell ... */
  df.show()

  /* ... new cell ... */
  val allc = df.agg("age" -> "count").map(_.getAs[Long](0)).collect.head.toFloat

  /* ... new cell ... */
  val grage = df.groupBy("age").count().map(r => (r.getInt(0),r.getLong(1)/allc*100)).collect.sortBy(_._2)
  /* ... new cell ... */
  import notebook.front.third.wisp._

  /* ... new cell ... */
  Plot(Seq(Pairs(grage, "pie")))

  /* ... new cell ... */
}
              