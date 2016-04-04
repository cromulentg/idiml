package com.idibon.ml.train

import com.idibon.ml.predict.Label
import com.idibon.ml.train.alloy.{DataSetInfo, TrainingDataSet}

import scala.concurrent.duration.{SECONDS, Duration}
import scala.collection.mutable.ListBuffer

import org.json4s.JObject

/** Configures various run-time training behaviors.
  *
  * @param maxTrainTime the maximum time allowed for each train calls
  * @param dataSet the dataset to use as training data (and also testing)
  */
case class TrainOptions(maxTrainTime: Duration,
                        dataSet: TrainingDataSet,
                        labels: Seq[Label],
                        rules: Seq[Rule] = Seq())

object TrainOptions {

  /** Creates a new builder to configure a TrainOptions */
  def apply() = new TrainOptionsBuilder
}

class TrainOptionsBuilder {

  /** Configures the maximum time allowed for each train operation
    *
    * By default, model training may take an unlimited amount of time; use
    * this method to configure a strict time limit.
    *
    * @param max maximum allowed time
    * @return this
    */
  def withMaxTrainTime(max: Duration): this.type = {
    this.maxTrainTime = max
    this
  }

  /** Configures the maximum time allowed for each train operation, in seconds
    *
    * This is equivalent to calling withMaxTrainTime(Duration(max, SECONDS))
    *
    * @param max maximum allowed time, in seconds
    * @return this
    */
  def withMaxTrainTime(max: Double): this.type = {
    withMaxTrainTime(Duration(max, SECONDS))
  }

  /** Adds documents for use as training data
    *
    * Documents are loaded lazily by any trainer that uses them. If this method
    * is called multiple times, the training documents will be concatenated.
    */
  def addDocuments(chunk: => TraversableOnce[JObject]): this.type = {
    this.documents += chunk _
    this
  }

  /**
    * Adds rules to train options.
    *
    * @param rules a sequence of (label: String, expression: String, weight: Float)
    * @return this object.
    */
  def addRules(rules: Seq[(String, String, Float)]) = {
    implicit val formats = org.json4s.DefaultFormats
    this.rules = rules.map({
      case (label, expression, weight) => new Rule(label, expression, weight)
    })
    this
  }

  /** Iterates over all documents in all submitted document lists */
  private[this] def documentIterator: TraversableOnce[JObject] = {
    new Traversable[JObject] {
      override def foreach[U](f: (JObject) => U) {
        documents.foreach(chunk => chunk().foreach(f))
      }
    }
  }

  /** Configures a TrainOptions from the values set on this builder.
    *
    * @return TrainOptions
    */
  def build(labels: Seq[Label]): TrainOptions = {
    val labelToDouble = labels.zipWithIndex.map({ case (label, index) => (label, index.toDouble) }).toMap
    TrainOptions(
      this.maxTrainTime,
      new TrainingDataSet(new DataSetInfo(0, 1.0, labelToDouble), () => this.documentIterator),
      labels,
      rules)
  }

  private[this] var maxTrainTime: Duration = Duration.Inf
  private[this] val documents = ListBuffer[() => TraversableOnce[JObject]]()
  private[this] var rules: Seq[Rule] = Seq()
}

/** Internal train options rule representation class **/
case class Rule(label: String, expression: String, weight: Float)
