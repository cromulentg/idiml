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
      // delta should always be greater than 0, else FeaturePipeline is wonky.
      assert(delta > 0, s"Expected ${coefficients.size} but got ${features.size} which was smaller.")
      logger.trace(s"Predicting with ${delta} OOV dimensions.")
      val sparseVector = features.asInstanceOf[SparseVector]
      val stoppingIndex = {
        val index = sparseVector.indices.indexWhere(_ >= coefficients.size)
        if (index > -1) index
        else sparseVector.indices.size
      }
      // can take slice since indices are always are in order of value, and thus new features
      // will always be at the end.
      val modifiedFeatures = Vectors.sparse(
        coefficients.size,
        sparseVector.indices.slice(0, stoppingIndex),
        sparseVector.values.slice(0, stoppingIndex))
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
  def getSignificantDimensions(features: Vector, threshold: Float): SparseVector = {
    /* pre-allocate the storage for the returned sparse vector; it is
     * guaranteed to have no more active dimensions than the feature vector */
    val indices = new Array[Int](features.numActives)
    val probs = new Array[Double](features.numActives)
    var count = 0

    /* compute the probability of each feature individually; pre-allocate a 1-
     * entry sparse vector to avoid repeated allocations. we don't care about
     * how many times each feature appears in the feature vector, only the
     * feature's absolute weight. */
    val tempVector = Vectors.sparse(features.size, Array(0), Array(1.0)).toSparse

    features.foreachActive((index, _) => {
      val weight = coefficients(index)
      if (weight != 0.0 && !weight.isNaN) {
        tempVector.indices(0) = index
        // only look at the positive class (1)
        val prob = super.predictProbability(tempVector)(1)
        if (prob >= threshold) {
          indices(count) = index
          probs(count) = prob
          count += 1
        }
      }
    })

    if (count != 0) {
      Vectors.sparse(features.size, indices.take(count), probs.take(count)).toSparse
    } else {
      Vectors.zeros(features.size).toSparse
    }
  }

  /**
    * Returns a summary of training metrics and only makes sense post-training
    *
    *
    * @return summary of metrics post-training
    */
  def getSummary: BinaryLogisticRegressionSummary = {
    /*
    * TODO: Handle the pre-training scenario
    * TODO: Figure out whether this still works on a cluster. Since we're running things locally for now, the summary
    *       should be available, but perhaps not in a distributed environment (there's an unclear note in the Spark
    *       documentation referencing availability only to the driver).
    */

    this.summary.asInstanceOf[BinaryLogisticRegressionSummary]
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
