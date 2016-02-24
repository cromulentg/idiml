package com.idibon.ml.predict.ml.metrics

import com.idibon.ml.feature.{FeaturePipeline, Buildable}
import com.idibon.ml.predict.{Classification, PredictModel}
import org.apache.spark.mllib.evaluation.{MultilabelMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.json4s.JsonAST.{JBool, JString}
import org.json4s._

import scala.collection.mutable

/**
  * Functions to help with metrics.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>" on 2/11/16.
  */
trait MetricHelper {

  /**
    * Helper method to get the count of data points per label.
    *
    * @param data The dataframe to use.
    * @param colName The name of the column to group by and count.
    *                Default is "label". The type of this column must
    *                be Double.
    * @return sequence of pairs corresponding to
    *         (label, count of items for label)
    */
  def getLabelCounts(data: DataFrame,
                     colName: String = "label"): Seq[(Double, Int)] = {
    data
      .groupBy(colName)
      .count()
      .map(row => (row.getAs[Double](0), row.getAs[Long](1).toInt))
      .collect()
      .toSeq
  }

  /**
    * Method to create a sequence of metrics from a multiclass metrics object.
    *
    * @param metrics
    * @param doubleToLabel
    * @return
    */
  def createMultiClassMetrics(metrics: MulticlassMetrics,
                              doubleToLabel: Map[Double, String],
                              metricClass: MetricClass.Value = MetricClass.Multiclass):
  Seq[Metric with Buildable[_, _]] = {
    val metricSeq = Seq[Metric with Buildable[_, _]](
      new FloatMetric(MetricTypes.Precision, metricClass, metrics.precision.toFloat),
      new FloatMetric(MetricTypes.Recall, metricClass, metrics.recall.toFloat),
      new FloatMetric(MetricTypes.F1, metricClass, metrics.fMeasure.toFloat),
      new FloatMetric(MetricTypes.WeightedPrecision, metricClass,
        metrics.weightedPrecision.toFloat),
      new FloatMetric(MetricTypes.WeightedRecall, metricClass,
        metrics.weightedRecall.toFloat),
      new FloatMetric(MetricTypes.WeightedF1, metricClass,
        metrics.weightedFMeasure.toFloat),
      new FloatMetric(MetricTypes.WeightedFPR, metricClass,
        metrics.weightedFalsePositiveRate.toFloat)
    )
    val labelMetrics = metrics.labels.flatMap(label => {
      Seq[Metric with Buildable[_, _]](
        new LabelFloatMetric(MetricTypes.LabelPrecision, metricClass,
          doubleToLabel(label), metrics.precision(label).toFloat),
        new LabelFloatMetric(MetricTypes.LabelRecall, metricClass,
          doubleToLabel(label), metrics.recall(label).toFloat),
        new LabelFloatMetric(MetricTypes.LabelF1, metricClass,
          doubleToLabel(label), metrics.fMeasure(label).toFloat),
        new LabelFloatMetric(MetricTypes.LabelFPR, metricClass,
          doubleToLabel(label), metrics.falsePositiveRate(label).toFloat)
      )
    })
    val labelToDouble = doubleToLabel.map(x => (x._2, x._1))
    val confusionMatrix = Seq(new ConfusionMatrixMetric(
      MetricTypes.ConfusionMatrix,
      metricClass,
      labelToDouble.flatMap(x => {
        labelToDouble.map(y => {
          (x._1, y._1, metrics.confusionMatrix(x._2.toInt, y._2.toInt).toFloat)
        })
      }).toSeq))
    metricSeq ++ labelMetrics ++ confusionMatrix
  }

  /**
    * Method to create a string for logging some metrics.
    *
    * @param metrics
    */
  def stringifyMulticlassMetrics(metrics: MulticlassMetrics): String = {
    // Overall Statistics
    val precision = metrics.precision
    val recall = metrics.recall // same as true positive rate
    val f1Score = metrics.fMeasure
    val strBuilder = new StringBuilder()
      .append("Summary Statistics\n")
      .append(s"Precision = $precision\n")
      .append(s"Recall = $recall\n")
      .append(s"F1 Score = $f1Score\n")

    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      strBuilder.append(s"Precision($l) = ${metrics.precision(l)}\n")
    }
    // Recall by label
    labels.foreach { l =>
      strBuilder.append(s"Recall($l) = ${metrics.recall(l)}\n")
    }
    // False positive rate by label
    labels.foreach { l =>
      strBuilder.append(s"FPR($l) = ${metrics.falsePositiveRate(l)}\n")
    }
    // F-measure by label
    labels.foreach { l =>
      strBuilder.append(s"F1-Score($l) = ${metrics.fMeasure(l)}\n")
    }
    // Weighted stats
    strBuilder.append(s"Weighted precision: ${metrics.weightedPrecision}\n")
      .append(s"Weighted recall: ${metrics.weightedRecall}\n")
      .append(s"Weighted F1 score: ${metrics.weightedFMeasure}\n")
      .append(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}\n")
      // confusion matrix
      .append(s"Confusion Matrix:\n")
      .append(metrics.confusionMatrix)
    strBuilder.mkString
  }

  /**
    * Method to create a string for logging some metrics.
    *
    * @param metrics
    */
  def stringifyMultilabelMetrics(metrics: MultilabelMetrics): String = {
    // Overall Statistics
    val precision = metrics.precision
    val recall = metrics.recall // same as true positive rate
    val f1Score = metrics.f1Measure
    val strBuilder = new StringBuilder()
      .append("Summary Statistics\n")
      .append(s"Precision = $precision\n")
      .append(s"Recall = $recall\n")
      .append(s"F1 Score = $f1Score\n")
      .append(s"Accuracy  = ${metrics.accuracy}\n")
      .append(s"Subset Accuracy  = ${metrics.subsetAccuracy}\n")

    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      strBuilder.append(s"Precision($l) = ${metrics.precision(l)}\n")
    }
    // Recall by label
    labels.foreach { l =>
      strBuilder.append(s"Recall($l) = ${metrics.recall(l)}\n")
    }
    // F-measure by label
    labels.foreach { l =>
      strBuilder.append(s"F1-Score($l) = ${metrics.f1Measure(l)}\n")
    }
    // Weighted stats
    strBuilder.append(s"Weighted precision: ${metrics.microPrecision}\n")
      .append(s"Weighted recall: ${metrics.microRecall}\n")
      .append(s"Weighted F1 score: ${metrics.microF1Measure}\n")
      // confusion matrix
      .append(s"Accuracy: ${metrics.accuracy}\n")
      .append(s"Subset Accuracy: ${metrics.subsetAccuracy}\n")
    strBuilder.mkString
  }

  /**
    * Helper function to get suggested model thresholds from the freshly trained models.
    *
    * @param models
    * @return
    */
  def getModelThresholds(models: List[(String, PredictModel[Classification])]): Map[String, Float] = {
    models.map({ case (uuidLabel, model) => {
      val threshold = model.getTrainingSummary() match {
        case Some(seqTs) => {
          seqTs
            // this probably isn't needed, but incase there are more then one training summary underneath...
            .filter(ts => ts.identifier.equals(uuidLabel))
            // from the training summary grab the metric that corresponds to the BestF1Threshold
            .map(ts => ts.metrics
            .filter(m => m.metricType == MetricTypes.BestF1Threshold)
            .head.asInstanceOf[FloatMetric].float
          ).head
        }
        case None => 0.0f
      }
      (uuidLabel, threshold)
    }
    }).toMap
  }


  /**
    * Method to create a sequence of metrics from a multiclass metrics object.
    *
    * @param metrics
    * @param doubleToLabel
    * @return
    */
  def createMultilabelMetrics(metrics: MultilabelMetrics,
                              doubleToLabel: Map[Double, String],
                              metricClass: MetricClass.Value = MetricClass.Multilabel):
  Seq[Metric with Buildable[_, _]] = {
    val metricSeq = Seq[Metric with Buildable[_, _]](
      new FloatMetric(MetricTypes.Precision, metricClass, metrics.precision.toFloat),
      new FloatMetric(MetricTypes.Recall, metricClass, metrics.recall.toFloat),
      new FloatMetric(MetricTypes.F1, metricClass, metrics.f1Measure.toFloat),
      new FloatMetric(MetricTypes.MicroF1, metricClass, metrics.microF1Measure.toFloat),
      new FloatMetric(MetricTypes.MicroPrecision, metricClass, metrics.microPrecision.toFloat),
      new FloatMetric(MetricTypes.MicroRecall, metricClass, metrics.microRecall.toFloat),
      new FloatMetric(MetricTypes.HammingLoss, metricClass, metrics.hammingLoss.toFloat),
      new FloatMetric(MetricTypes.SubsetAccuracy, metricClass, metrics.subsetAccuracy.toFloat),
      new FloatMetric(MetricTypes.Accuracy, metricClass, metrics.accuracy.toFloat)
    )
    val labelMetrics = metrics.labels.flatMap(label => {
      Seq[Metric with Buildable[_, _]](
        new LabelFloatMetric(MetricTypes.LabelPrecision, metricClass,
          doubleToLabel(label), metrics.precision(label).toFloat),
        new LabelFloatMetric(MetricTypes.LabelRecall, metricClass,
          doubleToLabel(label), metrics.recall(label).toFloat),
        new LabelFloatMetric(MetricTypes.LabelF1, metricClass,
          doubleToLabel(label), metrics.f1Measure(label).toFloat)
      )
    })
    metricSeq ++ labelMetrics
  }

  /**
    * Helper function to create a set of feature vectors with positive labels.
    *
    * We use this to evaluate both our multiclass and multilabel gang models.
    *
    * @param pipeline
    * @param docs
    * @return
    */
  def createPositiveLPs(pipeline: FeaturePipeline,
                        docs: () => TraversableOnce[JObject]):
  (Map[Double, String], Seq[(List[Double], Vector)]) = {
    implicit val formats = org.json4s.DefaultFormats
    val labelToDoubleMap = mutable.HashMap[String, Double]()
    var numClasses = 0.0
    val positivesWithVector = docs().flatMap(document => {
      // Run the pipeline to generate the feature vector
      val featureVector = pipeline(document)
      if (featureVector.numActives < 1) {
        None // we will skip it
      } else {
        // get annotations
        val annotations = (document \ "annotations").extract[JArray]
        // get me the positives
        val positives = annotations.arr.map({ jsonValue => {
          // for each annotation, we assume it was provided so we can make a training point out of it.
          val JString(label) = jsonValue \ "label" \ "name"
          val JBool(isPositive) = jsonValue \ "isPositive"
          // If we haven't seen this label before, instantiate a list
          if (!labelToDoubleMap.contains(label)) {
            labelToDoubleMap.put(label, numClasses)
            numClasses += 1.0
          }
          val labelNumber = labelToDoubleMap.get(label).get
          (isPositive, labelNumber)
        }}) // discard negative polarity annotations (can I do that in the above step?)
          .filter({ case (isPositive, lbPt) => isPositive })
          // map it to just integer labels
          .map({ case (_, label) => label })
        Some(positives, featureVector)
      }
    }).filter({case (positives, fVect) => positives.nonEmpty}).toSeq
    (labelToDoubleMap.map(x=> (x._2, x._1)).toMap, positivesWithVector)
  }
}
