package com.idibon.ml.predict.ml.metrics

/*
 * @author "Stefan Krawczyk <stefan@idibon.com>" on 2/10/16.
 */

/**
  * When paired with a metric type, tells us where it came from.
  *
  * Note: don't change the names of metrics, since this will break
  * backwards compatibility as we serialize the names. You should feel
  * free to reorder though.
  */
object MetricClass extends Enumeration {
  type MetricClass = Value
  val Alloy,
      Binary,
      Multiclass,
      Multilabel,
      Hyperparameter
      = Value
}
