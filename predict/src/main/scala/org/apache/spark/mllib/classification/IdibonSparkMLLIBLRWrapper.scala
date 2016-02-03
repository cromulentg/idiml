package org.apache.spark.mllib.classification

import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.mllib.linalg.BLAS._
import org.apache.spark.mllib.linalg.{SparseVector, DenseVector, Vectors, Vector}

import scala.collection.mutable.ListBuffer

/**
  * Extends an MLLIB LR implementation.
  *
  * Namely it exposes the probability calculation, and significant features.
  */
class IdibonSparkMLLIBLRWrapper(weights: Vector,
                                intercept: Double,
                                numFeatures: Int,
                                numClasses: Int)
  extends LogisticRegressionModel(weights, intercept, numFeatures, numClasses) with StrictLogging {

  private val dataWithBiasSize: Int = weights.size / (numClasses - 1)

  private val weightsArray: Array[Double] = weights match {
    case dv: DenseVector => dv.values
    case _ =>
      throw new IllegalArgumentException(
        s"weights only supports dense vector but got type ${weights.getClass}.")
  }

  /**
    * Method we expose to get at the internals for single vector prediction and
    * returning probabilities for all classes.
    *
    * @param features
    * @return
    */
  def predictProbability(features: Vector): Vector = {
    if(numFeatures != features.size) {
      val delta = features.size - numFeatures
      // delta should always be greater than 0, else FeaturePipeline is wonky.
      assert(delta > 0, s"Expected ${numFeatures} but got ${features.size} which was smaller.")
      logger.trace(s"Predicting with ${delta} OOV dimensions.")
      val sparseVector = features.asInstanceOf[SparseVector]
      val stoppingIndex = {
        val index = sparseVector.indices.indexWhere(_ >= numFeatures)
        if (index > -1) index
        else sparseVector.indices.size
      }
      // can take slice since indices are always are in order of value, and thus new features
      // will always be at the end.
      val modifiedFeatures = Vectors.sparse(
        weights.size,
        sparseVector.indices.slice(0, stoppingIndex),
        sparseVector.values.slice(0, stoppingIndex))
      computeProbabilities(modifiedFeatures)
    } else {
      computeProbabilities(features)
    }
  }

  /**
    * Helper method to compute probabilities of classes given a feature vector.
    * @param features
    * @return
    */
  protected def computeProbabilities(features: Vector): Vector = {
    if (numClasses == 2) {
      val margin = dot(weights, features) + intercept
      val score = 1.0 / (1.0 + math.exp(-margin))
      Vectors.sparse(2, Array(0, 1), Array(1.0 - score, score))
    } else {
      var bestClass = 0
      var maxMargin = 0.0
      val withBias = features.size + 1 == dataWithBiasSize
      val margins = (0 until numClasses - 1).map { i =>
        var margin = 0.0
        features.foreachActive { (index, value) =>
          if (value != 0.0) margin += value * weightsArray((i * dataWithBiasSize) + index)
        }
        // Intercept is required to be added into margin.
        if (withBias) {
          margin += weightsArray((i * dataWithBiasSize) + features.size)
        }
        if (margin > maxMargin) {
          maxMargin = margin
          bestClass = i + 1
        }
        margin
        // not sure about the bits here -- guesstimating without really thinking
      }.map(margin => {
        margin - math.max(0, maxMargin) //only subtract if maxMargin is greater than 0
      }).map(margin => 1.0 / (1.0 + math.exp(-margin))).toList
      val leftOver = 1.0 - margins.sum
      Vectors.sparse(numClasses, (0 until numClasses).toArray, (leftOver :: margins).toArray)
    }
  }

  /**
    * Returns a list of labelIndex -> List of significant features.
    *
    * @param features
    * @param threshold
    * @return
    */
  def getSignificantFeatures(features: Vector, threshold: Float): List[(Int, List[(Int, Float)])] = {
    val sigFeatures = (0 until numClasses).map(i => (i, new ListBuffer[(Int, Float)]())).toMap
    features.foreachActive((index, value) => {
      val prob = this.predictProbability(Vectors.sparse(features.size, Array(index), Array(value)))
      prob.foreachActive((labelIndex, probability) => {
        if (probability >= threshold){
          sigFeatures(labelIndex) += ((index, probability.toFloat))
        }
      })
    })
    sigFeatures.map(x => (x._1, x._2.toList)).toList
  }

  /**
    * Helper method to return a vector of indicies that correspond to features used by the model.
    * We cannot just return the weights, since the weight vector indicies cover weights for
    * all labels in the multinomial case. In the binomial case we can just return the weights
    * vector.
    *
    * @return Vector where the indicies represent features used in the model.
    */
  def getFeaturesUsed(): Vector = {
    if (numClasses == 2) {
      assert(numFeatures == weights.numActives, "number of features and weights should match in MLLIB LR model.")
      weights.toSparse
    } else {
      // want to take first numFeatures non-zero entries in dense vector
      val indicies = new Array[Int](numFeatures)
      val values = new Array[Double](numFeatures)
      assert(numFeatures <= dataWithBiasSize, "number of features should be equal to or smaller than nonZero values")
      var i = 0
      var k = 0
      while (i < numFeatures && k < numFeatures) {
        val v = weightsArray(k)
        if (v != 0.0) {
          indicies(i) = k
          values(i) = v
          i += 1
        }
        k += 1
      }
      if (i < numFeatures)
        Vectors.sparse(numFeatures, indicies.slice(0, i), values.slice(0, i))
      else
        Vectors.sparse(numFeatures, indicies, values)
    }
  }
}

/**
  * Static class to house static methods.
  */
object IdibonSparkMLLIBLRWrapper extends StrictLogging {

  /**
    * Creates an IdibonSparkMLLIBLRWrapper object from a MLLIB LR model.
    *
    * @param lrm
    * @return
    */
  def wrap(lrm: LogisticRegressionModel): IdibonSparkMLLIBLRWrapper = {
    new IdibonSparkMLLIBLRWrapper(lrm.weights, lrm.intercept, lrm.numFeatures, lrm.numClasses)
  }
}
