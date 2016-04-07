package com.idibon.ml.predict.ml

import com.idibon.ml.feature.Buildable
import com.idibon.ml.predict.ml.metrics._
import org.scalatest.{BeforeAndAfter, FunSpec, Matchers}


/**
  * Class to test the training summary functions
  */
class TrainingSummarySpec extends FunSpec with Matchers with BeforeAndAfter {

  describe("to String tests") {
    it("creates correct string for empty metrics") {
      val ts = new TrainingSummary("tests", Seq())
      val actual = ts.toString
      val expected = "----- Training summary for tests ---- Total = 0\n"
      actual shouldBe expected
    }
    it("creates correct string") {
      val ts = new TrainingSummary(
        "tests", Seq(new FloatMetric(MetricTypes.F1, MetricClass.Alloy, 0.5f)))
      val actual = ts.toString
      val expected = "----- Training summary for tests ---- Total = 1\n[Alloy, F1, 0.5]\n"
      actual shouldBe expected
    }
  }

  describe("average summaries tests") {
    it("averages as simple float metrics") {
      val ts1 = new TrainingSummary(
        "tests", Seq(new FloatMetric(MetricTypes.F1, MetricClass.Alloy, 0.4f)))
      val ts2 = new TrainingSummary(
        "tests", Seq(new FloatMetric(MetricTypes.F1, MetricClass.Alloy, 0.6f)))
      val actual = TrainingSummary.averageSummaries(
        "test-avg", Seq(ts1, ts2), MetricClass.Multiclass)
      actual.identifier shouldBe "test-avg"
      actual.metrics shouldBe Seq(new FloatMetric(MetricTypes.F1, MetricClass.Multiclass, 0.5f))
    }
  }

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
