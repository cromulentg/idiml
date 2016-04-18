package com.idibon.ml.train.datagenerator.scales

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.{SQLContext, DataFrame}

import com.typesafe.scalalogging.StrictLogging

/**
  * Classes that deal with balancing datasets.
  *
  */

/** Modifies training data sets to produce more-optimal label distributions
  *
  * Some Furnaces will produce models that fail to generalize if the training
  * data is heavily skewed toward a small number of labels, or will fail to
  * train if labels are missing. The DataSetScale implementations can be used
  * to improve model performance and protect against degeneracy.
  */
abstract class DataSetScale {

  /** Rebalances the training data set
    *
    * @param sql current spark SQLContext
    * @param trainingData the input raw training data
    * @return a possibly-modified set of training data
    */
  def apply(sql: SQLContext, trainingData: DataFrame): DataFrame = {
    DataSetScale.ensureTwoClasses(sql, trainingData)
      .getOrElse(balance(sql, trainingData))
  }

  /** Implementation of balancing algorithm
    *
    * @param sql current spark SQLContext
    * @param trainingData input raw training data
    * @return a possibly-modified set of training data
    */
  protected def balance(sql: SQLContext, trainingData: DataFrame): DataFrame
}

object DataSetScale {

  /** Ensures that trainind data exists for at least two classes
    *
    * Logistic regression fitting fails when the training data only includes
    * examples for a single label. In these degenerate cases, this method
    * produces a second label (0.0 if all examples are for class 1.0, 1.0 if
    * all examples are for class 0.0) with identical training items as the
    * present label, so that model predictions have 50% confidence.
    *
    * @param sql current spark SQLContext
    * @param data raw training data
    * @return a new data frame if only one class exists in rawData, else None
    */
  def ensureTwoClasses(sql: SQLContext, data: DataFrame): Option[DataFrame] = {
    data.groupBy("label").count.count match {
      case 0 => throw new RuntimeException("No training data")
      case 1 =>
        val missingLabel = data.head.getAs[Double]("label") match {
          case 0.0 => 1.0
          case _ => 0.0
        }
        /* the padding data is just all the original feature vectors with
         * the missing label assigned */
        val padding = sql.createDataFrame(data.map(row => {
          LabeledPoint(missingLabel, row.getAs[Vector]("features"))
        }))
        Some(data.unionAll(padding))
      case _ => None
    }
  }
}

/** Randomly filters binary classifier data to produce a desired distribution
  *
  * If the ratio between training items is below a user-defined threshold, the
  * balanced data set will randomly discard elements from the more-popular
  * label so that the overall label distribution matches the threshold.
  *
  * @param builder configuration for the scale
  */
case class BalancedBinaryScale(builder: BalancedBinaryScaleBuilder)
    extends DataSetScale with StrictLogging {

  val tolerance = builder.tolerance
  val seed = builder.seed

  /** Balances the training set, if needed
    *
    * {@link com.idibon.ml.train.datagenerator.scales.DataSetScale#balance}
    */
  protected def balance(sql: SQLContext, data: DataFrame): DataFrame = {
    val dist = data.groupBy("label").count.orderBy("count").collect
    require(dist.size == 2, "Only binary classifiers supported")

    // ratio of less-frequent to more-frequent label
    val r = dist(0).getAs[Long]("count").toDouble / dist(1).getAs[Long]("count")

    if (r < tolerance) {
      // randomly discard elements from the more-frequent label to reach parity
      val a = dist(0).getAs[Double]("label")
      val b = dist(1).getAs[Double]("label")

      val dataA = data.filter(s"label = $a")
      val dataB = data.filter(s"label = $b")

      dataA.unionAll(dataB.sample(false, r, seed))
    } else {
      data
    }
  }
}

/** Trivial no-op balancer that returns the input training data */
case class NoOpScale() extends DataSetScale {

  protected def balance(sql: SQLContext, data: DataFrame) = data
}
