package com.idibon.ml.train.datagenerator

import com.idibon.ml.feature.FeaturePipeline

import org.apache.spark.mllib.regression.LabeledPoint
import org.json4s.JObject

/** Creates mllib LabeledPoint data from training JSON documents */
trait LabeledPointGenerator {

  /** Converts an annotated document into training data for regression models
    *
    * Implementations are allowed to return training data for multiple models
    * from a single document, or for multiple labels within a single model, or
    * any combination thereof. Each returned data point is tagged with the
    * model and label (within that model) that it trains.
    *
    * Label names in the returned training data will exactly correspond to the
    * discrete label value in the LabeledPoint object. Model identifiers use
    * implementation-dependent names, e.g., a LabeledPointGenerator for a
    * k-binary classifier model (i.e., separate binary classifiers for each
    * annotated label) may use the original label names as model identifiers,
    * whereas a multi-class classifier may use a single globally-declared value.
    *
    * Implementations must ensure that this method is thread-safe.
    *
    * @param pipeline primed feature pipeline for document => vector generation
    * @param json JSON training document, with annotations
    * @return sequence of LabeledPoint training data
    */
  def apply(pipeline: FeaturePipeline, json: JObject): Seq[TrainingPoint]
}

/** LabeledPoint training item for a specific model and label.
  *
  * @param modelId model identifier
  * @param labelName label name
  * @param p training LabeledPoint
  */
case class TrainingPoint(modelId: String, labelName: String, p: LabeledPoint)

/** LabeledPointGenerator for k-binary classifier models
  *
  * Every LabeledPoint will have a class of either 0.0 (for negative training
  * data) or 1.0 (for positive training data). Each annotated label in the
  * document set is assigned a unique model with a single label.
  */
class KClassLabeledPointGenerator extends LabeledPointGenerator {

  /** Converts an annotated document into training data for regression models */
  def apply(pipeline: FeaturePipeline, j: JObject): Seq[TrainingPoint] = {
    implicit val formats = org.json4s.DefaultFormats
    val document = j.extract[json.Document]

    if (document.annotations.exists(_.isSpan))
      throw new IllegalArgumentException("Detected span annotations")

    val fv = pipeline(j)

    if (fv.numActives < 1) {
      // don't create training data for all-OOV documents
      Seq()
    } else {
      document.annotations.map(ann => {
        val klass = if (ann.isPositive) 1.0 else 0.0
        val name = if (ann.isPositive) "positive" else "negative"
        TrainingPoint(ann.label.name, name, LabeledPoint(klass, fv))
      })
    }
  }
}

/** LabeledPointGenerator for multi-class / multinomial models
  *
  * Each annotated label in the document set is assigned a unique class value
  * (discrete double values corresponding to the order in which positively-
  * annotated documents for that label are processed), and the class value
  * in the generated LabeledPoints is this value. All returned training data
  * will be for a common model.
  */
class MulticlassLabeledPointGenerator extends LabeledPointGenerator {

  // keep track of label IDs as they are found
  private[this] val _labelIds = collection.mutable.HashMap[String, Int]()

  def apply(pipeline: FeaturePipeline, j: JObject): Seq[TrainingPoint] = {
    implicit val formats = org.json4s.DefaultFormats
    val document = j.extract[json.Document]

    if (document.annotations.exists(_.isSpan))
      throw new IllegalArgumentException("Detected span annotations")

    val fv = pipeline(j)

    if (fv.numActives < 1) {
      Seq()
    } else {
      document.annotations.filter(_.isPositive).map(ann => {
        val klass = _labelIds.synchronized {
          _labelIds.getOrElseUpdate(ann.label.name, _labelIds.size)
        }
        TrainingPoint("model", ann.label.name, LabeledPoint(klass, fv))
      })
    }
  }
}
