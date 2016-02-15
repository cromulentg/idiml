package com.idibon.ml.predict.ml.metrics

/*
 * @author "Stefan Krawczyk <stefan@idibon.com>" on 2/10/16.
 */

/**
  * A metric type corresponds to a particular metric we could get back.
  *
  * Note: don't change the names of metrics, since this will break
  * backwards compatibility as we serialize the names. You should feel
  * free to reorder though.
  */
object MetricType extends Enumeration {
  type MetricType = Value
  val BestF1Threshold,
      AreaUnderROC,
      ReceiverOperatingCharacteristic,
      PrecisionRecallCurve,
      F1ByThreshold,
      PrecisionByThreshold,
      RecallByThreshold,
      Precision,
      Recall,
      F1,
      LabelPrecision,
      LabelRecall,
      LabelF1,
      LabelFPR,
      WeightedPrecision,
      WeightedRecall,
      WeightedF1,
      WeightedFPR,
      ConfusionMatrix,
      LabelCount,
      HyperparameterProperties,
      MicroF1,
      MicroPrecision,
      MicroRecall,
      HammingLoss,
      SubsetAccuracy,
      Accuracy
      = Value
}

/**
  * When paird with a metric type, tells us where it came from.
  *
  * Note: don't change the names of metrics, since this will break
  * backwards compatibility as we serialize the names. You should feel
  * free to reorder though.
  */
object MetricClass extends Enumeration {
  type MetricClass = Value
  val Binary,
      Multiclass,
      Multilabel,
      Hyperparameter
      = Value
}
