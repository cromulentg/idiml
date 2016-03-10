package com.idibon.ml.predict.ml.metrics

import com.idibon.ml.feature.Buildable
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
      new LabelPointsMetric(MetricTypes.LearningCurveLabelF1, MetricClass.Binary, "label", Seq((0.5f, 0.3f)))
      new PropertyMetric(MetricTypes.HyperparameterProperties, MetricClass.Binary, Seq(("p1", "v1")))
      new LabelFloatListMetric(MetricTypes.LabelProbabilities, MetricClass.Binary, "label", Seq(0.4f, 0.5f, 0.6f))
      new ConfusionMatrixMetric(MetricTypes.ConfusionMatrix, MetricClass.Binary, Seq(("a", "b", 1.0f)))
    }
  }
}


/**
  * Verifies the functionality of the Metric Averages in Metric.scala
  *
  */
class AverageMetricSpec extends FunSpec with Matchers
  with BeforeAndAfter with BeforeAndAfterAll {

  describe("static metric average tests"){
    it("catches different metric subclass types") {
      val mType = MetricTypes.F1
      val mClass = MetricClass.Multiclass
      val input = Seq[Metric with Buildable[_, _]](
        new FloatMetric(mType, mClass, 0.6f),
        new FloatMetric(mType, mClass, 0.6f),
        new PointsMetric(MetricTypes.F1ByThreshold, mClass, Seq((0.3f, 0.2f))))
      intercept[AssertionError] {
        Metric.average(input)
      }
    }
    it("catches different metric types") {
      val mType = MetricTypes.F1
      val mClass = MetricClass.Multiclass
      val input = Seq[Metric with Buildable[_, _]](
        new FloatMetric(mType, mClass, 0.6f),
        new FloatMetric(MetricTypes.Precision, mClass, 0.6f),
        new FloatMetric(mType, mClass, 0.6f))
      intercept[AssertionError] {
        Metric.average(input)
      }
    }
    it("catches different metric classes") {
      val mType = MetricTypes.F1
      val mClass = MetricClass.Multiclass
      val input = Seq[Metric with Buildable[_, _]](
        new FloatMetric(mType, mClass, 0.6f),
        new FloatMetric(mType, MetricClass.Binary, 0.6f),
        new FloatMetric(mType, mClass, 0.6f))
      intercept[AssertionError] {
        Metric.average(input)
      }
    }
    it("works as intended") {
      val mType = MetricTypes.F1
      val mClass = MetricClass.Multiclass
      val input = Seq[Metric with Buildable[_, _]](
        new FloatMetric(mType, mClass, 0.4f),
        new FloatMetric(mType, mClass, 0.5f),
        new FloatMetric(mType, mClass, 0.3f))
      val actual = Metric.average(input)
      val expected = Seq(new FloatMetric(mType, mClass, 0.4f))
      actual shouldBe expected
    }
  }

  describe("individual metric average tests") {
    it("float metric computes average properly") {
      val mType = MetricTypes.F1
      val mClass = MetricClass.Multiclass
      val input = Seq(
        new FloatMetric(mType, mClass, 0.6f),
        new FloatMetric(mType, mClass, 0.1f),
        new FloatMetric(mType, mClass, 0.5f))
      val actual = Metric.average(input)
      val expected = Seq(new FloatMetric(mType, mClass, 0.4f))
      actual shouldBe expected
    }
    it("label float metric computes average properly") {
      val mType = MetricTypes.LabelF1
      val mClass = MetricClass.Multiclass
      val input = Seq(
        new LabelFloatMetric(mType, mClass, "label1", 0.5f),
        new LabelFloatMetric(mType, mClass, "label1", 0.1f),
        new LabelFloatMetric(mType, mClass, "label2", 0.5f),
        new LabelFloatMetric(mType, mClass, "label2", 0.3f)
      )
      val actual = Metric.average(input).sortBy(m => m.asInstanceOf[LabelFloatMetric].label)
      val expected = Seq(
        new LabelFloatMetric(mType, mClass, "label1", 0.3f),
        new LabelFloatMetric(mType, mClass, "label2", 0.4f)
      )
      actual shouldBe expected
    }
    it("label int metric computes average properly") {
      val mType = MetricTypes.LabelCount
      val mClass = MetricClass.Multiclass
      val input = Seq(
        new LabelIntMetric(mType, mClass, "label1", 5),
        new LabelIntMetric(mType, mClass, "label1", 1),
        new LabelIntMetric(mType, mClass, "label2", 5),
        new LabelIntMetric(mType, mClass, "label2", 3))
      val actual = Metric.average(input).sortBy(m => m.asInstanceOf[LabelIntMetric].label)
      val expected = Seq(
        new LabelIntMetric(mType, mClass, "label1", 3),
        new LabelIntMetric(mType, mClass, "label2", 4)
      )
      actual shouldBe expected
    }
    it("points metric computes average properly") {
      val mType = MetricTypes.PrecisionByThreshold
      val mClass = MetricClass.Multiclass
      val input = Seq(
        new PointsMetric(mType, mClass, Seq((0.1f, 0.2f), (0.4f, 0.3f))),
        new PointsMetric(mType, mClass, Seq((0.1f, 0.3f), (0.4f, 0.1f))),
        new PointsMetric(mType, mClass, Seq((0.1f, 0.4f), (0.4f, 0.2f))))
      val actual = Metric.average(input)
      val expected = Seq(new PointsMetric(mType, mClass, Seq((0.1f, 0.3f), (0.4f, 0.2f))))
      actual.size shouldBe expected.size
      actual.zip(expected).foreach({case (a, e) =>
        a.metricType shouldBe e.mType
        a.metricClass shouldBe e.mClass
        val points = a.asInstanceOf[PointsMetric].points.sortBy(x => x._1)
        points.zip(e.points).foreach( {case (actualPoint, expectedPoint) =>
          actualPoint._1 shouldBe expectedPoint._1
          actualPoint._2 shouldBe (expectedPoint._2 +- 0.001f)
        })
      })
    }
    it("label points metric computes average properly") {
      val mType = MetricTypes.LearningCurveLabelF1
      val mClass = MetricClass.Multiclass
      val input = Seq(
        new LabelPointsMetric(mType, mClass, "label1", Seq((0.1f, 0.2f), (0.4f, 0.3f))),
        new LabelPointsMetric(mType, mClass, "label2", Seq((0.2f, 0.6f), (0.4f, 0.1f))),
        new LabelPointsMetric(mType, mClass, "label1", Seq((0.1f, 0.4f), (0.4f, 0.5f))),
        new LabelPointsMetric(mType, mClass, "label2", Seq((0.2f, 0.4f), (0.4f, 0.3f))))
      val actual = Metric.average(input).sortBy(m => m.asInstanceOf[LabelPointsMetric].label)
      val expected = Seq(
        new LabelPointsMetric(mType, mClass, "label1", Seq((0.1f, 0.3f), (0.4f, 0.4f))),
        new LabelPointsMetric(mType, mClass, "label2", Seq((0.2f, 0.5f), (0.4f, 0.2f))))
      actual.size shouldBe expected.size
      actual.zip(expected).foreach({case (a, e) =>
        val aLPM = a.asInstanceOf[LabelPointsMetric]
        aLPM.metricType shouldBe e.mType
        aLPM.metricClass shouldBe e.mClass
        aLPM.label shouldBe e.label
        val points = aLPM.points.sortBy(x => x._1)
        points.zip(e.points).foreach( {case (actualPoint, expectedPoint) =>
          actualPoint._1 shouldBe expectedPoint._1
          actualPoint._2 shouldBe (expectedPoint._2 +- 0.001f)
        })
      })
    }
    it("property metric computes average properly") {
      val mType = MetricTypes.HyperparameterProperties
      val mClass = MetricClass.Multiclass
      val input = Seq(
        new PropertyMetric(mType, mClass, Seq(("a", "b"))),
        new PropertyMetric(mType, mClass, Seq(("a", "b"))),
        new PropertyMetric(mType, mClass, Seq(("b", "c"))))
      val actual = Metric.average(input)
      val expected = input
      actual shouldBe expected
    }
    it("confusion matrix metric computes sum properly") {
      val mType = MetricTypes.ConfusionMatrix
      val mClass = MetricClass.Multiclass
      val input = Seq(
        new ConfusionMatrixMetric(mType, mClass, Seq(("a", "b", 0.1f))),
        new ConfusionMatrixMetric(mType, mClass, Seq(("a", "b", 0.5f))),
        new ConfusionMatrixMetric(mType, mClass, Seq(("a", "b", 0.03f))))
      val actual = Metric.average(input)
      val expected = Seq(new ConfusionMatrixMetric(mType, mClass, Seq(("a", "b", 0.21f))))
      actual shouldBe expected
    }

    it("LabelFloatListMetric computes concat properly") {
      val mType = MetricTypes.LabelProbabilities
      val mClass = MetricClass.Multiclass
      val input = Seq(
        new LabelFloatListMetric(mType, mClass, "x", Seq(0.1f, 0.2f, 0.6f)),
        new LabelFloatListMetric(mType, mClass, "x", Seq(0.5f, 0.7f, 0.8f)),
        new LabelFloatListMetric(mType, mClass, "y", Seq(0.03f, 0.4f, 0.6f)))
      val actual = Metric.average(input).sortBy(x => x.asInstanceOf[LabelFloatListMetric].label)
      val expected = Seq(
        new LabelFloatListMetric(mType, mClass, "x", Seq(0.1f, 0.2f, 0.5f, 0.6f, 0.7f, 0.8f)),
        new LabelFloatListMetric(mType, mClass, "y", Seq(0.03f, 0.4f, 0.6f))
      )
      actual shouldBe expected
    }
  }
}
