package com.idibon.ml.train.alloy

import scala.io.Source
import scala.concurrent.Await
import scala.concurrent.duration.Duration

import com.idibon.ml.alloy.Alloy
import com.idibon.ml.common.Engine
import com.idibon.ml.train.TrainOptions
import com.idibon.ml.predict._

import org.json4s._
import org.json4s.native.JsonMethods

/** Create static Java methods from the companion object */
class BasicTrain {}

/** Simple top-level interface for training alloys from a set of files */
object BasicTrain {

  /** Reads a JSON object from a text file; assumes the entire text file
    * (possibly multi-line) stores exactly 1 JSON object.
    *
    * @param filename filename to read
    */
  def readJsonFile(filename: String): JObject = {
    val file = Source.fromFile(filename)
    try {
      JsonMethods.parse(file.mkString).asInstanceOf[JObject]
    } finally {
      file.close()
    }
  }

  /** Returns a function that reads a newline-delimited JSON file into
    * a TraversableOnce[JObject].
    */
  def lazyFileReader(filename: String): () => TraversableOnce[JObject] = {
    implicit val formats = org.json4s.DefaultFormats
    () => {
      Source.fromFile(filename).getLines.map(line => {
        JsonMethods.parse(line).extract[JObject]
      })
    }
  }

  /** Trains an Alloy using configuration and data located in local files
    *
    * @param engine The engine context to use for training
    * @param alloyName A user-friendly name to identify the alloy
    * @param trainingDataFile A file containing JSON training data
    * @param taskConfigFile A JSON file containing label and rule configuration
    * @param alloyConfigFile A JSON file containing the trainer parameters
    */
  def trainFromFiles(engine: Engine,
    alloyName: String,
    trainingDataFile: String,
    taskConfigFile: String,
    alloyConfigFile: String): Alloy[_] = {

    implicit val formats = org.json4s.DefaultFormats

    val alloyConfigJSON = readJsonFile(alloyConfigFile)
    val alloyConfig = alloyConfigJSON.extract[ForgeConfiguration]
    val taskConfigJSON = readJsonFile(taskConfigFile)
    val taskConfig = taskConfigJSON.extract[AlloyConfiguration]

    taskConfig.task_type match {
      case "extraction.bio_ner" => {
        val labels = taskConfig.uuid_to_label
          .map({ case (uuid, name) => new Label(uuid, name) }).toSeq
        val options = TrainOptions()
          .addDocuments(lazyFileReader(trainingDataFile)())
        val trainer = AlloyForge[Span](engine, alloyConfig.forgeName, alloyName, labels,
          alloyConfig.forgeConfig)
        val evaluator = trainer.getEvaluator(engine, "extraction.bio_ner")
        Await.result(trainer.forge(options.build(labels), evaluator), Duration.Inf)
      }
      case _ => {
        val trainer = AlloyFactory.getTrainer(engine, alloyConfig.trainerConfig)

        trainer.trainAlloy(alloyName,
          lazyFileReader(trainingDataFile),
          taskConfigJSON, Some(alloyConfigJSON))
      }
    }
  }
}

// === JSON schema ====

/** Schema for for label_and_rules (i.e., task / alloy) configuration file
  *
  * @param task_type type of task being trained (classification vs extraction)
  * @param uuid_to_label mapping of UUID strings to the (current) label name
  * @param rules opaque rules data
  */
case class AlloyConfiguration(
                               task_type: String,
                               uuid_to_label: Map[String, String],
                               rules: Option[JObject])

/** Schema for JSON training configuration file */
case class ForgeConfiguration(forgeName: String = "", forgeConfig: JObject = null, trainerConfig: JObject = null, configVersion: String = "0.0.1")
