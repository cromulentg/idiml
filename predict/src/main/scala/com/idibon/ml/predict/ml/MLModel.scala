package com.idibon.ml.predict.ml

import com.idibon.ml.predict._
import com.idibon.ml.feature.FeaturePipeline
import org.apache.spark.mllib.linalg.Vector
import org.json4s.JObject

/** Base class for linear machine learning models
  *
  * Models may optionally include feature pipelines to transform raw
  * document content into suitable feature vectors, or may perform
  * the prediction operation using a pre-transformed vector.
  */
abstract class MLModel[+T <: PredictResult](featureExtract: (JObject) => Vector)
    extends PredictModel[T] {

  def predict(document: Document, options: PredictOptions): Seq[T] = {

    val features = featureExtract(document.json)
    predictVector(features, options)
  }

  /** Performs a prediction using a transformed vector. To be over-ridden
    * by implementations.
    *
    * @param features Transformed feature vector
    * @param options requested optional prediction behavior
    */
  protected def predictVector(features: Vector,
    options: PredictOptions): Seq[T] = ???
}
