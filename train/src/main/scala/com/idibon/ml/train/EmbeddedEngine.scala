package com.idibon.ml.train

import java.io.File

import com.idibon.ml.alloy.{Codec,IntentAlloy}
import com.idibon.ml.feature.{ContentExtractor, FeaturePipelineLoader, FeaturePipelineBuilder}
import com.idibon.ml.feature.indexer.IndexTransformer
import com.idibon.ml.feature.tokenizer.TokenTransformer
import org.apache.commons.io.FileUtils
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.Saveable
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.native.JsonMethods.{parse, render, compact}

import scala.collection.mutable.HashMap

/** EmbeddedEngine
  *
  * Performs training, given a set of documents and annotations.
  *
  */
class EmbeddedEngine extends com.idibon.ml.common.Engine {

  val sparkContext = EmbeddedEngine.sparkContext

  /** Produces an RDD of LabeledPoints for each distinct label name.
    *
    * @param labeledPoints: a set of training data
    * @return a trained model based on the MLlib logistic regression model with LBFGS, trained on the provided
    *         dataset
    */
  def getLogisticRegressionModel(labeledPoints: RDD[LabeledPoint]) : LogisticRegressionModel = {
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(labeledPoints)

    model
  }

  /** Writes a model to the filesystem.
    *
    * @param sc: the SparkContext for this application
    * @param model: the trained model to be saved
    * @param path: the filesystem path to be written
    *
    * @return a trained model based on the MLlib logistic regression model with LBFGS, trained on the provided
    *         dataset
    */
  def saveMllibModel(sc: SparkContext, model: Saveable, path: String) = {
    // TODO: Insert standard storage location
    model.save(sc, path)
  }

  /**
    * Currently only one SparkContext can exist per JVM, hence the use of this companion object
    */
  object EmbeddedEngine {
    val sparkContext = {
      val conf = new SparkConf().setMaster("local").setAppName("idiml")
      new SparkContext(conf)
    }
  }

  /** Trains a model and saves it at the given filesystem location
    *
    * @param infilePath: the location of the ididat dump file generated by the export_training_to_idiml.rb tool,
    *                    found in idibin
    * @param modelStoragePath: the filesystem path for saving models
    */
  def start(infilePath: String, modelStoragePath: String): Unit = {
    // Instantiate the Spark environment
    val conf = new SparkConf().setAppName("idiml").setMaster("local[8]").set("spark.driver.host", "localhost")
    val sc = new SparkContext(conf)

    // Define a pipeline that generates feature vectors
    val pipeline = (FeaturePipelineBuilder.named("IntentPipeline")
      += (FeaturePipelineBuilder.entry("convertToIndex", new IndexTransformer, "convertToTokens"))
      += (FeaturePipelineBuilder.entry("convertToTokens", new TokenTransformer, "contentExtractor"))
      += (FeaturePipelineBuilder.entry("contentExtractor", new ContentExtractor, "$document"))
      := ("convertToIndex"))

    val training: Option[HashMap[String, RDD[LabeledPoint]]] = new RDDGenerator()
      .getLabeledPointRDDs(sc, infilePath, pipeline)
    if (training.isEmpty) {
      println("Error generating training points; Exiting.")
      return
    }
    val trainingData = training.get

    val logisticRegressionModels = HashMap[String, LogisticRegressionModel]()
    for ((label, labeledPoints) <- trainingData) {
      // Perform training
      val model = getLogisticRegressionModel(labeledPoints)

      // Create a file-safe label name
      val fileSafeLabelName = label.replace(' ', '-')
      // Remove the directory if it already exists
      FileUtils.deleteDirectory(new File(s"${modelStoragePath}/${fileSafeLabelName}/LogisticRegressionModel"))
      // Save the model
      saveMllibModel(sc, model, s"${modelStoragePath}/${fileSafeLabelName}/LogisticRegressionModel")

      logisticRegressionModels(label) = model
    }

    val alloy = new IntentAlloy()

    // Save the pipeline definition
    val test = pipeline.save(alloy.writer.within("IntentPipeline"))
    val alloyMetaJson: JObject = ("IntentPipeline" -> test)
    val writer = alloy.writer.within("IntentPipeline").resource("config.json")
    Codec.String.write(writer, compact(render(alloyMetaJson)))

    // Load the pipeline definition again
    val reader = alloy.reader.within("IntentPipeline").resource("config.json")
    val config = Codec.String.read(reader)
    val newPipelineConfig: JObject = (parse(config) \ "IntentPipeline").asInstanceOf[JObject]
    val newPipeline2 = (new FeaturePipelineLoader).load(this, alloy.reader().within("IntentPipeline"),
                                                        Some(newPipelineConfig))

  }
}
