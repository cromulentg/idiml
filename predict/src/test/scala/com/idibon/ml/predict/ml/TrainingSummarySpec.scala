package com.idibon.ml.predict.ml

import com.idibon.ml.feature.Buildable
import com.idibon.ml.predict.ml.metrics._
import org.scalatest.{BeforeAndAfter, FunSpec, Matchers}


/**
  * Class to test the training summary functions
  */
class TrainingSummarySpec extends FunSpec with Matchers with BeforeAndAfter {

  describe("getNotesValues tests") {
    it("handles no notes") {
      // empty metrics
      val ts = new TrainingSummary("id", Seq())
      ts.getNotesValues("key") shouldBe Seq()
      // no notes metric
      val ts2 = new TrainingSummary("id",
        Seq(new FloatMetric(MetricTypes.F1, MetricClass.Alloy, 0.5f)))
      ts2.getNotesValues("key") shouldBe Seq()
    }
    it("handles empty property sequence") {
      val metrics = Seq[Metric with Buildable[_, _]](
        new FloatMetric(MetricTypes.F1, MetricClass.Alloy, 0.5f),
        new PropertyMetric(MetricTypes.Notes, MetricClass.Alloy, Seq()))
      val ts = new TrainingSummary("id", metrics)
      ts.getNotesValues("key") shouldBe Seq()
    }
    it("handles not having that particular note") {
      val metrics = Seq[Metric with Buildable[_, _]](
        new FloatMetric(MetricTypes.F1, MetricClass.Alloy, 0.5f),
        new PropertyMetric(MetricTypes.Notes, MetricClass.Alloy, Seq(("a", "b"))))
      val ts = new TrainingSummary("id", metrics)
      ts.getNotesValues("key") shouldBe Seq()
    }
    it("handles find that particular note") {
      val metrics = Seq[Metric with Buildable[_, _]](
        new FloatMetric(MetricTypes.F1, MetricClass.Alloy, 0.5f),
        new PropertyMetric(MetricTypes.Notes, MetricClass.Alloy,
          Seq(("key", "b"))))
      val ts = new TrainingSummary("id", metrics)
      ts.getNotesValues("key") shouldBe Seq("b")
    }
    it("handles find those particular notes") {
      val metrics = Seq[Metric with Buildable[_, _]](
        new FloatMetric(MetricTypes.F1, MetricClass.Alloy, 0.5f),
        new PropertyMetric(MetricTypes.Notes, MetricClass.Alloy,
          Seq(("key", "b"), ("key", "z"))))
      val ts = new TrainingSummary("id", metrics)
      ts.getNotesValues("key") shouldBe Seq("b", "z")
    }
  }
}
