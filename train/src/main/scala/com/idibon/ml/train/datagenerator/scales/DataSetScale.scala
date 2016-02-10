package com.idibon.ml.train.datagenerator.scales

import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Classes that deal with balancing datasets.
  *
  */

/**
  * Trait that deals with scaling datasets to appropriate proportions of
  * negative & positive examples.
  */
trait DataSetScale {

  /**
    * Function to balance a training set.
    *
    * This strategy could be a pass through or could actually do something.
    *
    * @param label the label we're balancing.
    * @param labeledPoints the set of RDDs for that label.
    * @return a balance RDD dataset for the passed in label.
    */
  def balance(label: String, labeledPoints: RDD[LabeledPoint]): RDD[LabeledPoint]
}

/**
  * Creates balanced datasets if the ratio of binary data is below the tolerance threshold.
  *
  * @param builder
  */
case class BalancedBinaryScale(builder: BalancedBinaryScaleBuilder)
  extends DataSetScale with StrictLogging {
  val tolerance = builder.tolerance
  val seed = builder.seed

  /**
    * Function to balance a training set.
    *
    * This particular strategy randomly removes negatives or positives if either ratio compared to
    * the other is below `tolerance`.
    *
    * @param label         the label we're balancing.
    * @param labeledPoints the set of RDDs for that label.
    * @return a balance RDD dataset for the passed in label.
    */
  override def balance(label: String, labeledPoints: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    val byPosNeg = labeledPoints.groupBy(x => x.label)
    val polarityCounts: Map[Double, Int] = byPosNeg.map({ case (label, points) => (label, points.size) })
      .toLocalIterator.map(x => x).toMap
    if (polarityCounts.size != 2) {
      logger.warn(s"Skipping balancing since there are ${polarityCounts.size} classes," +
        s" rather than the needed 2.")
      return labeledPoints
    }
    val neg_ratio = polarityCounts.get(1.0).get.toFloat / polarityCounts.get(0.0).get.toFloat
    val pos_ratio = 1.0 / neg_ratio

    if (neg_ratio < tolerance) {
      logger.info(s"Ratio is $neg_ratio for $label; Balancing!")
      val targetNum = polarityCounts.get(0.0).get.toFloat * neg_ratio
      val negTraining = labeledPoints.filter(l => l.label == 0.0).sample(false, neg_ratio, seed)
      val posTraining = labeledPoints.filter(l => l.label == 1.0)
      posTraining.union(negTraining)
    } else if (pos_ratio < tolerance) {
      logger.info(s"Ratio is $pos_ratio for $label; Balancing!")
      val negTraining = labeledPoints.filter(l => l.label == 0.0)
      val posTraining = labeledPoints.filter(l => l.label == 1.0).sample(false, pos_ratio, seed)
      posTraining.union(negTraining)
    } else {
      logger.info(s"Ratio is $neg_ratio & $pos_ratio for $label; Within tolerance. No balancing needed.")
      labeledPoints
    }
  }
}

/**
  * No-op scale.
  */
case class NoOpScale() extends DataSetScale {
  /**
    * Function to balance a training set.
    *
    * This one does nothing to it.
    *
    * @param label         the label we're balancing.
    * @param labeledPoints the set of RDDs for that label.
    * @return a balance RDD dataset for the passed in label.
    */
  override def balance(label: String, labeledPoints: RDD[LabeledPoint]): RDD[LabeledPoint] = labeledPoints

}
