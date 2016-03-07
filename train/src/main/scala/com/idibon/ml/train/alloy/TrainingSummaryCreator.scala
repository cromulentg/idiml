package com.idibon.ml.train.alloy

import java.util

import com.idibon.ml.common.Engine
import com.idibon.ml.predict.Classification
import com.idibon.ml.predict.ml.TrainingSummary
import com.idibon.ml.predict.ml.metrics.{MetricClass, MetricTypes, FloatMetric, MetricHelper}
import org.apache.spark.mllib.evaluation.{MultilabelMetrics, MulticlassMetrics}
import scala.collection.JavaConversions._

/**
  * @author "Stefan Krawczyk <stefan@idibon.com>" on 3/3/16.
  */
/**
  * Training summary creator that handles explaining the behavior expected for
  * creating training summaries.
  *
  * TODO: see about using this in the other trainers.
  */
trait TrainingSummaryCreator extends MetricHelper {
  /**
    * Creates a tuple of (prediction(s), label(s)).
    * This is because this is the raw data point that spark's internal metrics
    * classes use, which we use to compute statistics from.
    *
    * @param labelToDouble
    * @param goldSet
    * @param classifications
    * @param thresholds
    */
  def createEvaluationDataPoint(labelToDouble: Map[String, Double],
                                goldSet: Set[String],
                                classifications: util.List[Classification],
                                thresholds: Map[String, Float]): (Array[Double], Array[Double])

  /**
    * Creates a training summary from a sequence of data points.
    *
    * @param engine
    * @param dataPoints
    * @param labelToDouble
    * @param summaryName
    * @param portion
    * @return
    */
  def createTrainingSummary(engine: Engine,
                            dataPoints: Seq[(Array[Double], Array[Double])],
                            labelToDouble: Map[String, Double],
                            summaryName: String,
                            portion: Double = 1.0): TrainingSummary

}

/**
  * This handles the mutually exclusive label case.
  *
  * @param defaultThreshold
  */
case class MultiClassMetricsEvaluator(defaultThreshold: Float) extends TrainingSummaryCreator {
  /**
    * Creates a tuple of (prediction(s), label(s)).
    * This is because this is the raw data point that spark's internal metrics
    * classes use, which we use to compute statistics from.
    *
    * @param labelToDouble
    * @param goldSet
    * @param classifications
    * @param thresholds
    */
  override def createEvaluationDataPoint(labelToDouble: Map[String, Double],
                                         goldSet: Set[String],
                                         classifications: util.List[Classification],
                                         thresholds: Map[String, Float]):
  (Array[Double], Array[Double]) = {
    val goldLabel = goldSet.map(gl => labelToDouble(gl)).head
    val maxLabel: Classification = getMaxLabel(classifications, thresholds)
    (Array[Double](labelToDouble(maxLabel.label)), Array[Double](goldLabel))
  }

  /**
    * Handles returning the maximal label. In the mutually exclusive case we NEED
    * to return a label for every prediction. Since by definition one of the labels
    * must be chosen.
    *
    * @param classifications
    * @param thresholds
    * @return
    */
  def getMaxLabel(classifications: util.List[Classification],
                  thresholds: Map[String, Float]): Classification = {
    /*
      Take arg-max of ones over threshold, if none, then take plain arg-max.
      We do this weird thing because we could be dealing with a k-binary
      classifiers, and since we're simulating a multiclass classifier we NEED
      to output a prediction.
    */
    val labelsOverThreshold = classifications.filter(c => {
      val threshold = thresholds.getOrDefault(c.label, defaultThreshold)
      c.probability >= threshold
    })
    val maxLabel = labelsOverThreshold.nonEmpty match {
      case true => labelsOverThreshold.maxBy(c => c.probability)
      case false => classifications.maxBy(c => c.probability)
    }
    maxLabel
  }

  /**
    * Creates a training summary from a sequence of data points.
    *
    * @param engine
    * @param dataPoints
    * @param labelToDouble
    * @param summaryName
    * @param portion
    * @return
    */
  override def createTrainingSummary(engine: Engine,
                                     dataPoints: Seq[(Array[Double], Array[Double])],
                                     labelToDouble: Map[String, Double],
                                     summaryName: String,
                                     portion: Double = 1.0): TrainingSummary = {
    val predictionRDDs = engine.sparkContext.parallelize(dataPoints.map(p => (p._1.head, p._2.head)))
    val multiClass = new MulticlassMetrics(predictionRDDs)
    val metrics = createMultiClassMetrics(multiClass, labelToDouble.map(x => (x._2, x._1)))
    new TrainingSummary(
      summaryName,
      metrics ++ Seq(new FloatMetric(MetricTypes.Portion, MetricClass.Multiclass, portion.toFloat)))
  }
}

/**
  * Handles the multi-label case.
  *
  * @param defaultThreshold
  */
case class MultiLabelMetricsEvaluator(defaultThreshold: Float) extends TrainingSummaryCreator {
  /**
    * Creates a tuple of (prediction(s), label(s)).
    * This is because this is the raw data point that spark's internal metrics
    * classes use, which we use to compute statistics from.
    *
    * @param labelToDouble
    * @param goldSet
    * @param classifications
    * @param thresholds
    */
  override def createEvaluationDataPoint(labelToDouble: Map[String, Double],
                                         goldSet: Set[String],
                                         classifications: util.List[Classification],
                                         thresholds: Map[String, Float]):
  (Array[Double], Array[Double]) = {
    val goldLabels = goldSet.map(gl => labelToDouble(gl)).toArray
    val predictedLabels = // filter to only those over threshold -- this could potentially be empty.
      classifications.filter(c => {
        c.probability >= thresholds.getOrDefault(c.label, defaultThreshold)
      }).map(c => labelToDouble(c.label)).toArray
    (predictedLabels, goldLabels)
  }

  /**
    * Creates a training summary from a sequence of data points.
    *
    * @param engine
    * @param dataPoints
    * @param labelToDouble
    * @param summaryName
    * @param portion
    * @return
    */
  override def createTrainingSummary(engine: Engine,
                                     dataPoints: Seq[(Array[Double], Array[Double])],
                                     labelToDouble: Map[String, Double],
                                     summaryName: String,
                                     portion: Double = 1.0): TrainingSummary = {
    val predictionRDDs = engine.sparkContext.parallelize(dataPoints)
    val multiLabel = new MultilabelMetrics(predictionRDDs)
    val metrics = createMultilabelMetrics(multiLabel, labelToDouble.map(x => (x._2, x._1)))
    new TrainingSummary(
      summaryName,
      metrics ++ Seq(new FloatMetric(MetricTypes.Portion, MetricClass.Multilabel, portion.toFloat)))
  }
}
