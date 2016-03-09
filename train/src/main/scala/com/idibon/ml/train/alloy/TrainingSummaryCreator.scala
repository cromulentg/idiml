package com.idibon.ml.train.alloy

import java.util

import com.idibon.ml.common.Engine
import com.idibon.ml.feature.Buildable
import com.idibon.ml.predict.Classification
import com.idibon.ml.predict.ml.TrainingSummary
import com.idibon.ml.predict.ml.metrics._
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MultilabelMetrics, MulticlassMetrics}
import org.apache.spark.sql.functions._
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
                                thresholds: Map[String, Float]): EvaluationDataPoint

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
                            dataPoints: Seq[EvaluationDataPoint],
                            labelToDouble: Map[String, Double],
                            summaryName: String,
                            portion: Double = 1.0): TrainingSummary


  /**
    * Helper method to create metrics for each label using the raw probabilities.
    *
    * @param labelToDouble
    * @param dataPoints
    * @param metricClass
    * @return
    */
  def createPerLabelMetricsFromProbabilities(engine: Engine,
                                             labelToDouble: Map[String, Double],
                                             dataPoints: Seq[EvaluationDataPoint],
                                             metricClass: MetricClass.Value):
  Seq[Metric with Buildable[_, _]] = {
    val doubleToLabel = labelToDouble.map(x => (x._2, x._1))
    val labelProbs: Seq[Metric with Buildable[_, _]] = collatePerLabelProbabilities(
      dataPoints.flatMap(e => e.rawProbabilities), doubleToLabel, metricClass)
    val labelThresholds: Seq[Metric with Buildable[_, _]] = getSuggestedLabelThreshold(
      engine, dataPoints, doubleToLabel, metricClass)
    labelProbs ++ labelThresholds
  }

  /**
    * Helper method to create LabelProbabilities metrics.
    *
    * @param dataPoints
    * @param doubleToLabel
    * @param metricClass
    * @return
    */
  def collatePerLabelProbabilities(dataPoints: Seq[(Double, Float)],
                                   doubleToLabel: Map[Double, String],
                                   metricClass: MetricClass.Value): Seq[LabelFloatListMetric] = {
    val doubleLabelToProb = dataPoints
      .groupBy({ case (label, prob) => label })
    // create sequence of metrics, one per label
    doubleLabelToProb.map({ case (doubleLabel, probs) =>
      val points = probs.map(x => x._2).sortBy(x => x)
      new LabelFloatListMetric(
        MetricTypes.LabelProbabilities, metricClass, doubleToLabel(doubleLabel), points)
    }).toSeq
  }

  /**
    * Returns a sequence of label float metrics containing the suggested threshold for that label.
    *
    * For a label creates "binary label" data from the gold data and uses the probabilities passed in
    * to compute the threshold that achieves the optimum F1.
    *
    * @param engine
    * @param points
    * @param doubleToLabel
    * @param metricClass
    * @return
    */
  def getSuggestedLabelThreshold(engine: Engine,
                                 points: Seq[EvaluationDataPoint],
                                 doubleToLabel: Map[Double, String],
                                 metricClass: MetricClass.Value): Seq[LabelFloatMetric] = {
    val sqlContext = new org.apache.spark.sql.SQLContext(engine.sparkContext)
    val byLabel = points.flatMap(e => {
      e.rawProbabilities.map({case (dblLabel, prob) =>
        val binaryLabel = if (e.gold.contains(dblLabel)) 1.0 else 0.0
        (dblLabel, prob, binaryLabel)
      })
    }).groupBy({case (dblLabel, prob, binaryLabel) => dblLabel})
    byLabel.map({case (label, grouped) =>
      val bestThreshold = computeBestF1Threshold(engine, sqlContext,
        grouped.map({case (dblLabel, prob, binaryLabel) => (prob.toDouble, binaryLabel)}))
      new LabelFloatMetric(
        MetricTypes.LabelBestF1Threshold, metricClass, doubleToLabel(label), bestThreshold)
    }).toSeq
  }

  /**
    * Helper method to extract the float threshold value.
    *
    * Constructs the RDD & delegates to Spark's BinaryClassificationMetrics before doing
    * the query to find the right threshold.
    *
    * @param engine
    * @param sqlContext
    * @param labelValues
    * @return
    */
  def computeBestF1Threshold(engine: Engine,
                             sqlContext: org.apache.spark.sql.SQLContext,
                             labelValues: Seq[(Double, Double)]): Float = {
    val predictionRDDs = engine.sparkContext.parallelize(labelValues)
    // use 100 since that's what spark uses internally
    val binaryMetrics = new BinaryClassificationMetrics(predictionRDDs, 100)
    val fMeasure = sqlContext.createDataFrame(binaryMetrics.fMeasureByThreshold())
    // _1 is threshold, _2 is metric
    val maxFMeasure = fMeasure.select(max("_2")).head().getDouble(0)
    val bestThreshold = fMeasure.where(fMeasure.col("_2") === maxFMeasure)
      .select("_1").head().getDouble(0)
    bestThreshold.toFloat
  }

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
                                         thresholds: Map[String, Float]): EvaluationDataPoint = {
    val goldLabel = goldSet.map(gl => labelToDouble(gl)).head
    val maxLabel: Classification = getMaxLabel(classifications, thresholds)
    val rawProbabilities = classifications.map(c => (labelToDouble(c.label), c.probability)).toSeq
    new EvaluationDataPoint(
      Array[Double](labelToDouble(maxLabel.label)), Array[Double](goldLabel), rawProbabilities)
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
                                     dataPoints: Seq[EvaluationDataPoint],
                                     labelToDouble: Map[String, Double],
                                     summaryName: String,
                                     portion: Double = 1.0): TrainingSummary = {
    val labeledPoints = dataPoints.map(e => (e.predicted.head, e.gold.head))
    val predictionRDDs = engine.sparkContext.parallelize(labeledPoints)
    val multiClass = new MulticlassMetrics(predictionRDDs)
    val metrics = createMultiClassMetrics(multiClass, labelToDouble.map(x => (x._2, x._1)))
    new TrainingSummary(
      summaryName,
      metrics ++
        Seq(new FloatMetric(MetricTypes.Portion, MetricClass.Multiclass, portion.toFloat)) ++
        createPerLabelMetricsFromProbabilities(engine, labelToDouble, dataPoints, MetricClass.Multiclass))
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
                                         thresholds: Map[String, Float]): EvaluationDataPoint = {
    val goldLabels = goldSet.map(gl => labelToDouble(gl)).toArray
    val predictedLabels = // filter to only those over threshold -- this could potentially be empty.
      classifications.filter(c => {
        c.probability >= thresholds.getOrDefault(c.label, defaultThreshold)
      }).map(c => labelToDouble(c.label)).toArray
    val rawProbabilities = classifications.map(c => (labelToDouble(c.label), c.probability)).toSeq
    new EvaluationDataPoint(predictedLabels, goldLabels, rawProbabilities)
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
                                     dataPoints: Seq[EvaluationDataPoint],
                                     labelToDouble: Map[String, Double],
                                     summaryName: String,
                                     portion: Double = 1.0): TrainingSummary = {
    val labeledPoints = dataPoints.map(e => (e.predicted, e.gold))
    val predictionRDDs = engine.sparkContext.parallelize(labeledPoints)
    val multiLabel = new MultilabelMetrics(predictionRDDs)
    val metrics = createMultilabelMetrics(multiLabel, labelToDouble.map(x => (x._2, x._1)))
    new TrainingSummary(
      summaryName,
      metrics ++
        Seq(new FloatMetric(MetricTypes.Portion, MetricClass.Multilabel, portion.toFloat)) ++
        createPerLabelMetricsFromProbabilities(engine, labelToDouble, dataPoints, MetricClass.Multilabel))
  }
}

/**
  * Class to help store data from an evaluation.
  *
  * @param predicted
  * @param gold
  * @param rawProbabilities
  */
case class EvaluationDataPoint(predicted: Array[Double],
                               gold: Array[Double],
                               rawProbabilities: Seq[(Double, Float)])
