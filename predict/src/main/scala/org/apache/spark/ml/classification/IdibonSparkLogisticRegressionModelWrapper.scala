package org.apache.spark.ml.classification;

import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.mllib.linalg.{SparseVector, Vectors, Vector}

import scala.collection.mutable.ListBuffer

/**
  * This class extends LR so we can do predictions at the atomic
  * Vector level, since we want to bypass using DataFrames for
  * predicting on single feature vectors.
  *
  * @author Stefan Krawczyk <stefan@idibon.com>
  *
  *
  * @param uid
  * @param coefficients
  * @param intercept
  */
class IdibonSparkLogisticRegressionModelWrapper(override val uid: String,
                                                override val coefficients: Vector,
                                                override val intercept: Double)
  extends LogisticRegressionModel(uid, coefficients, intercept) with StrictLogging {

  /**
    * Makes this method Public for us to access.
    *
    * @param features features to use for prediction.
    * @return a Vector where each index corresponds to the label index,
    *         and the value there is the probability.
    *         e.g. a binary classifier will have two results returned.
    */
  override def predictProbability(features: Vector): Vector = {
    if(coefficients.size != features.size) {
      val delta = features.size - coefficients.size
      assert(delta > 0) // delta should always be greater than 0, else FeaturePipeline is wonky.
      logger.trace(s"Predicting with ${delta} OOV dimensions.")
      val sparseVector = features.asInstanceOf[SparseVector]
      // can take slice since indices are always are in order of value, and thus new features
      // will always be at the end.
      val modifiedFeatures = Vectors.sparse(
        coefficients.size,
        sparseVector.indices.slice(0, coefficients.size - delta),
        sparseVector.values.slice(0, coefficients.size - delta))
      super.predictProbability(modifiedFeatures)
    } else {
      super.predictProbability(features)
    }
  }

  /**
    * Goes through the features passed in and checks their weights. It then only keeps the features
    * (indexes) where the probability (computed from the weight) is >= than the threshold passed in.
    *
    * @param features
    * @param threshold
    * @return
    */
  def getSignificantFeatures(features: Vector, threshold: Float): List[(Int, Float)] = {
    val sigFeatures = new ListBuffer[(Int, Float)]
    features.foreachActive((index, value) => {
      val weight = coefficients(index)
      if (weight > 0.0 || weight < 0.0) {
        // just predict the probabilty on one feature (this takes into account the intercept)
        val prob = super.predictProbability(Vectors.sparse(features.size, Array(index), Array(value)))(1)
        if (prob >= threshold) {
          sigFeatures += ((index, prob.toFloat))
        }
      }
    })
    sigFeatures.result()
  }
}

object IdibonSparkLogisticRegressionModelWrapper {
  /**
    * Helper method to wrap a create Spark Logistic Regression Model
    * @param model
    * @return
    */
  def wrap(model: LogisticRegressionModel): IdibonSparkLogisticRegressionModelWrapper = {
    val newModel = new IdibonSparkLogisticRegressionModelWrapper(
      model.uid, model.coefficients, model.intercept)
    if (model.hasSummary) {
      newModel.setSummary(model.summary)
    }
    newModel
  }
}
