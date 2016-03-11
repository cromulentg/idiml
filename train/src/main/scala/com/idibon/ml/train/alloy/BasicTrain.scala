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

    val alloyConfig = readJsonFile(alloyConfigFile)
    val taskConfig = readJsonFile(taskConfigFile)
    val taskType = (taskConfig \ "task_type").extract[String]

    taskType match {
      case "extraction.bio_ner" => {
        val labels = (alloyConfig \ "uuid_to_label")
          .extract[List[(String, String)]]
          .map({ case (uuid, name) => new Label(uuid, name) })
        val options = TrainOptions()
          .addDocuments(lazyFileReader(trainingDataFile)())
        val trainer = AlloyTrainer2[Span](engine, alloyName, labels.toSeq,
          (alloyConfig \ "trainerConfig").extract[JObject])
        Await.result(trainer.train(options.build()), Duration.Inf)
      }
      case _ => {
        val trainer = AlloyFactory.getTrainer(engine,
          (alloyConfig \ "trainerConfig").asInstanceOf[JObject])

        trainer.trainAlloy(alloyName,
          lazyFileReader(trainingDataFile),
          taskConfig, Some(alloyConfig))
      }
    }
  }
}
