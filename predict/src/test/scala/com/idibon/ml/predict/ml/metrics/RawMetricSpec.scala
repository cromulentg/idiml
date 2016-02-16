package com.idibon.ml.predict.ml.metrics

import org.scalatest.{BeforeAndAfter, BeforeAndAfterAll, FunSpec, Matchers}


/** Verifies the functionality of the RawMetric constructor.
  *
  */
class RawMetricSpec extends FunSpec with Matchers
  with BeforeAndAfter with BeforeAndAfterAll {

  describe("constructor tests") {
    it("should throw error on bad match") {
      intercept[IllegalArgumentException] {
        new FloatMetric(MetricTypes.F1ByThreshold, MetricClass.Binary, 0.5f)
      }
    }
    it("should not throw error on good data type match") {
      new FloatMetric(MetricTypes.BestF1Threshold, MetricClass.Binary, 0.5f)
      new LabelFloatMetric(MetricTypes.LabelF1, MetricClass.Binary, "label", 0.5f)
      new LabelIntMetric(MetricTypes.LabelCount, MetricClass.Binary, "label", 4)
      new PointsMetric(MetricTypes.F1ByThreshold, MetricClass.Binary, Seq((0.5f, 0.3f)))
      new PropertyMetric(MetricTypes.HyperparameterProperties, MetricClass.Binary, Seq(("p1", "v1")))
      new ConfusionMatrixMetric(MetricTypes.ConfusionMatrix, MetricClass.Binary, Seq(("a", "b", 1.0f)))
    }
  }
}
