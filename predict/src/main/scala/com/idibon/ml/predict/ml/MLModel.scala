package com.idibon.ml.predict.ml

import com.idibon.ml.predict._
import com.idibon.ml.feature.{FeaturePipeline, Feature}
import org.apache.spark.mllib.linalg.Vector
import org.json4s.JObject

/** Base class for linear machine learning models
  *
  * Models may optionally include feature pipelines to transform raw
  * document content into suitable feature vectors, or may perform
  * the prediction operation using a pre-transformed vector.
  */
abstract class MLModel[+T <: PredictResult](
  override val featurePipeline: Option[FeaturePipeline])
    extends PredictModel[T] with CanHazPipeline {

  def predict(document: Document, options: PredictOptions): Seq[T] = {
    applyPipelineIfPresent(document) match {
      case Document(_, Some((v, i))) => predictVector(v, i, options)
      case _ => throw new UnsupportedOperationException("No feature pipeline")
    }
  }

  /** Performs a prediction using a transformed vector. To be over-ridden
    * by implementations.
    *
    * @param features Transformed feature vector
    * @param invertFeatureFn function to map vector dimensions to features
    * @param options requested optional prediction behavior
    */
  protected def predictVector(features: Vector,
    invertFeatureFn: (Vector) => Seq[Option[Feature[_]]],
    options: PredictOptions): Seq[T] = ???
}
