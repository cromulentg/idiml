package com.idibon.ml.train.datagenerator

import java.io.File

import com.idibon.ml.common.Engine
import com.idibon.ml.feature.FeaturePipeline
import com.idibon.ml.train.datagenerator.scales.DataSetScale
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext, SaveMode}
import org.json4s._

/** Converts annotated documents into Spark DataFrames for model training
  *
  * @param scale object used to re-balance and pad raw training data
  * @param lpg generates labeled points from JSON documents
  */
abstract class SparkDataGenerator(scale: DataSetScale,
  lpg: LabeledPointGenerator) extends StrictLogging {

  /** Produces a DataFrame of LabeledPoints for one or more models
    *
    * Given a set of annotated training documents matching the
    * {@link com.idibon.ml.train.datagenerator.json.Document} schema, returns
    * DataFrames containing LabeledPoint training data for one or more models.
    *
    * @param engine the current idiml engine context
    * @param pipeline a FeaturePipeline to use for processing documents. Already primed.
    * @param docs a callback function returning the training documents
    * @return a list of models to train
    */
  def apply(engine: Engine, pipeline: FeaturePipeline,
    docs: () => TraversableOnce[JObject]): Seq[ModelData] = {

    // create a temporary directory for all of the intermediate files
    val temp = FileUtils.deleteAtExit(
      FileUtils.createTemporaryDirectory("idiml-datagen"))

    val sql = new SQLContext(engine.sparkContext)

    // process documents in batches to limit the amount of data in-memory
    val modelsAndLabels = docs().toStream.grouped(1000)
      .map(persistDocBatch(temp, sql, engine.sparkContext, pipeline, _))
      .foldLeft(Map[String, Map[String, Double]]())({
        (a, p) => SparkDataGenerator.deepMerge(a, p)
      })

    modelsAndLabels.toSeq.map({ case (modelId, labelClassMap) => {
      val name = SparkDataGenerator.safeName(modelId)
      val file = new File(temp, s"${name}.parquet")
      val frame = sql.read.parquet(file.getAbsolutePath)
      ModelData(modelId, labelClassMap, frame)
    }})

    // FIXME: balancing!
  }

  /** Creates training data for a batch of documents and persists to parquet
    *
    * @param trainerTemp temporary directory
    * @param sql SQLContext instance to use for creating DataFrames
    * @param spark SparkContext instance for creating RDDs
    * @param pipeline primed FeaturePipeline
    * @param batch a set of documents to process
    * @return a map of model identifiers and each model's label-to-class map
    */
  private[this] def persistDocBatch(temp: File, sql: SQLContext,
    spark: SparkContext, pipeline: FeaturePipeline, batch: Stream[JObject]):
      Map[String, Map[String, Double]] = {

    val modelsAndPoints = batch.par.flatMap(lpg(pipeline, _)).groupBy(_.modelId)

    modelsAndPoints.aggregate(Map[String, Map[String, Double]]())(
      (accum, entry) => {
        val name = SparkDataGenerator.safeName(entry._1)
        val file = new File(temp, s"${name}.parquet")
        // extract the labeled points for this model and append to Parquet
        sql.createDataFrame(spark.parallelize(entry._2.map(_.p).seq))
          .write.mode(SaveMode.Append).parquet(file.getAbsolutePath)

        /* add the label => class entries from this partition for this model
         * into the accumulated set */
        SparkDataGenerator.deepMerge(accum,
          Map(entry._1 -> entry._2.map(t => (t.labelName -> t.p.label)).toMap))

      }, (a, b) => SparkDataGenerator.deepMerge(a, b))
  }
}

/** Companion object for SparkDataGenerator */
private[datagenerator] object SparkDataGenerator {
  private type ModelMap = collection.GenMap[String, collection.GenMap[String, Double]]

  /** Returns a unique, filesystem-safe name for a model ID
    *
    * @param modelId model identifier string
    * @return hashed name using only fs-safe characters
    */
  def safeName(modelId: String): String = {
    val hash = java.security.MessageDigest.getInstance("SHA-1")
    val digest = hash.digest(modelId.getBytes("UTF-8"))
    digest.map(b => f"$b%02X").mkString.substring(0, 12)
  }

  /** Deeply merges 2 maps-of-maps into a resulting map
    *
    * @param x input map
    * @param y input map
    * @return new map consisting of all keys with merged value maps from x and y
    */
  def deepMerge(x: ModelMap, y: ModelMap): Map[String, Map[String, Double]] = {
    Map[String, Map[String, Double]]((x.keys ++ y.keys).map(k => {
      (k -> Map[String, Double](
        (x.getOrElse(k, Map[String, Double]()).toSeq.seq ++
          y.getOrElse(k, Map[String, Double]()).toSeq.seq): _*))
    }).toSeq.seq: _*)
  }

}

/** Generator for multi-nomial (multi-class) logistic regression models
  *
  * @param b configuration object
  */
class MultiClassDataFrameGenerator(b: MultiClassDataFrameGeneratorBuilder)
    extends SparkDataGenerator(b.scale.build, new MulticlassLabeledPointGenerator)

/** Generator for multiple binary logistic regression models
  *
  * @param b configuration object
  */
class KClassDataFrameGenerator(b: KClassDataFrameGeneratorBuilder)
    extends SparkDataGenerator(b.scale.build, new KClassLabeledPointGenerator)

/** Defines a model that may be trained
  *
  * @param id model identifier
  * @param labels map of label name to class identifier for classes in the frame
  * @param frame training data for the model
  */
case class ModelData(id: String, labels: Map[String, Double], frame: DataFrame)
