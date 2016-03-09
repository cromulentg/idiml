package com.idibon.ml.train.alloy


import com.idibon.ml.common.EmbeddedEngine
import com.idibon.ml.feature.Buildable
import com.idibon.ml.predict.Classification
import com.idibon.ml.predict.ml.metrics._
import org.scalatest._

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._


/**
  * Tests the TrainingSummaryCreator classes
  */
class TrainingSummaryCreatorSpec extends FunSpec
  with Matchers with BeforeAndAfter with ParallelTestExecution with BeforeAndAfterAll {

  val engine = new EmbeddedEngine

  describe("MultiClassMetricsEvaluator tests") {
    it("createEvaluationDataPoint works as expected"){
      val me = new MultiClassMetricsEvaluator(0.5f)
      val classifications = List(
        new Classification("m", 0.35f, 1, 0, Seq()),
        new Classification("n", 0.3f, 1, 0, Seq())
      ).asJava
      val actual = me.createEvaluationDataPoint(
        Map("m" -> 0.0, "n" -> 1.0), Set("m"), classifications, Map("m" -> 0.5f))
      val expected = (Array(0.0), Array(0.0))
      actual.predicted(0) shouldBe expected._1(0)
      actual.gold(0) shouldBe expected._2(0)
    }
    it("getMaxLabel something over default threshold, no thresholds passed") {
      val me = new MultiClassMetricsEvaluator(0.5f)
      val classifications = List(
        new Classification("m", 0.35f, 1, 0, Seq()),
        new Classification("n", 0.6f, 1, 0, Seq())
      ).asJava
      val actual = me.getMaxLabel(classifications, Map())
      actual shouldBe classifications(1)
    }
    it("getMaxLabel something over threshold, with thresholds passed") {
      val me = new MultiClassMetricsEvaluator(0.5f)
      val classifications = List(
        new Classification("m", 0.35f, 1, 0, Seq()),
        new Classification("n", 0.6f, 1, 0, Seq())
      ).asJava
      val actual = me.getMaxLabel(classifications, Map("m" -> 0.3f, "n" -> 0.7f))
      actual shouldBe classifications(0)
    }
    it("getMaxLabel none over any threshold") {
      val me = new MultiClassMetricsEvaluator(0.7f)
      val classifications = List(
        new Classification("m", 0.35f, 1, 0, Seq()),
        new Classification("n", 0.6f, 1, 0, Seq())
      ).asJava
      val actual = me.getMaxLabel(classifications, Map("m" -> 0.7f))
      actual shouldBe classifications(1)
    }
    it("createTrainingSummary works as expected") {
      val me = new MultiClassMetricsEvaluator(0.7f)
      val actual = me.createTrainingSummary(engine, Seq(
                new EvaluationDataPoint(Array(1.0), Array(1.0), Seq()),
                new EvaluationDataPoint(Array(0.0), Array(0.0), Seq())
              ), Map("a" -> 1.0, "b" -> 0.0), "name")
      actual.identifier shouldBe "name"
      actual.metrics.size shouldBe 17
      val expected = Some(new FloatMetric(MetricTypes.F1, MetricClass.Multiclass, 1.0f))
      actual.metrics.find(m => m.metricType == MetricTypes.F1) shouldBe expected
    }
  }

  describe("MultiLabelMetricsEvaluator tests") {
    it("createEvaluationDataPoint some over default threshold"){
      val me = new MultiLabelMetricsEvaluator(0.2f)
      val classifications = List(
        new Classification("m", 0.35f, 1, 0, Seq()),
        new Classification("n", 0.3f, 1, 0, Seq())
      ).asJava
      val actual = me.createEvaluationDataPoint(
        Map("m" -> 0.0, "n" -> 1.0), Set("m", "n"), classifications, Map("m" -> 0.5f))
      val expected = (Array(1.0), Array(0.0, 1.0))
      actual.predicted(0) shouldBe expected._1(0)
      actual.gold(0) shouldBe expected._2(0)
      actual.gold(1) shouldBe expected._2(1)
    }
    it("createEvaluationDataPoint some over passed in threshold"){
      val me = new MultiLabelMetricsEvaluator(0.5f)
      val classifications = List(
        new Classification("m", 0.35f, 1, 0, Seq()),
        new Classification("n", 0.3f, 1, 0, Seq())
      ).asJava
      val actual = me.createEvaluationDataPoint(
        Map("m" -> 0.0, "n" -> 1.0), Set("m", "n"), classifications, Map("m" -> 0.3f))
      val expected = (Array(0.0), Array(0.0, 1.0))
      actual.predicted(0) shouldBe expected._1(0)
      actual.gold(0) shouldBe expected._2(0)
      actual.gold(1) shouldBe expected._2(1)
    }
    it("createEvaluationDataPoint none over threshold"){
      val me = new MultiLabelMetricsEvaluator(0.7f)
      val classifications = List(
        new Classification("m", 0.35f, 1, 0, Seq()),
        new Classification("n", 0.3f, 1, 0, Seq())
      ).asJava
      val actual = me.createEvaluationDataPoint(
        Map("m" -> 0.0, "n" -> 1.0), Set("m", "n"), classifications, Map("m" -> 0.5f))
      val expected = (Array(), Array(0.0, 1.0))
      actual.predicted.length shouldBe expected._1.length
      actual.gold(0) shouldBe expected._2(0)
      actual.gold(1) shouldBe expected._2(1)
    }
    it("createTrainingSummary works as expected") {
      val me = new MultiLabelMetricsEvaluator(0.7f)
      val actual = me.createTrainingSummary(engine, Seq(
                new EvaluationDataPoint(Array(1.0), Array(1.0), Seq()),
                new EvaluationDataPoint(Array(0.0), Array(0.0), Seq())
              ), Map("a" -> 1.0, "b" -> 0.0), "name")
      actual.identifier shouldBe "name"
      actual.metrics.size shouldBe 16
      val expected = Some(new FloatMetric(MetricTypes.F1, MetricClass.Multilabel, 1.0f))
      actual.metrics.find(m => m.metricType == MetricTypes.F1) shouldBe expected
    }
  }

  describe("TrainingSummaryCreator tests") {
    it("createPerLabelMetricsFromProbabilities correctly") {
      val edps = Seq(
        new EvaluationDataPoint(Array(0.0), Array(1.0), Seq((0.0, 0.2f), (1.0, 0.3f))),
        new EvaluationDataPoint(Array(1.0), Array(0.0), Seq((0.0, 0.3f), (1.0, 0.4f)))
      )
      val ltodb = Map("a" -> 0.0, "b" -> 1.0)
      val x = new DummyTrainingSummaryCreator()
      val actual = x.createPerLabelMetricsFromProbabilities(engine, ltodb, edps, MetricClass.Binary)
      actual.sortBy(x => x match {
        case a: LabelFloatListMetric => (a.label, a.metricType)
        case b: LabelFloatMetric => (b.label, b.metricType)
      }) shouldBe Seq[Metric with Buildable[_, _]](
        new LabelFloatMetric(
          MetricTypes.LabelBestF1Threshold, MetricClass.Binary, "a", 0.3f),
        new LabelFloatListMetric(
          MetricTypes.LabelProbabilities, MetricClass.Binary, "a", Seq(0.2f, 0.3f)),
        new LabelFloatMetric(
          MetricTypes.LabelBestF1Threshold, MetricClass.Binary, "b", 0.3f),
      new LabelFloatListMetric(
            MetricTypes.LabelProbabilities, MetricClass.Binary, "b", Seq(0.3f, 0.4f))
      )
    }
    it("collatePerLabelProbabilities correctly") {
      val dps = Seq((0.0, 0.2f), (1.0, 0.3f), (0.0, 0.3f), (1.0, 0.4f))
      val dbtl = Map(0.0 -> "a", 1.0 -> "b")
      val x = new DummyTrainingSummaryCreator()
      val actual = x.collatePerLabelProbabilities(dps, dbtl, MetricClass.Alloy)
      actual.sortBy(x => x.label) shouldBe Seq(
        new LabelFloatListMetric(
          MetricTypes.LabelProbabilities, MetricClass.Alloy, "a", Seq(0.2f, 0.3f)),
        new LabelFloatListMetric(
          MetricTypes.LabelProbabilities, MetricClass.Alloy, "b", Seq(0.3f, 0.4f))
      )
    }
    it("creates best label f1 metrics properly") {
      val edps = Seq(
        new EvaluationDataPoint(Array(0.0), Array(1.0), Seq((0.0, 0.4f), (1.0, 0.3f))),
        new EvaluationDataPoint(Array(1.0), Array(0.0), Seq((0.0, 0.3f), (1.0, 0.4f))),
        new EvaluationDataPoint(Array(0.0), Array(0.0), Seq((0.0, 0.6f), (1.0, 0.4f))),
        new EvaluationDataPoint(Array(1.0), Array(1.0), Seq((0.0, 0.3f), (1.0, 0.5f))),
        new EvaluationDataPoint(Array(1.0), Array(1.0), Seq((0.0, 0.35f), (1.0, 0.56f)))
      )
      val dbletol = Map(0.0 -> "a", 1.0 -> "b")
      val x = new DummyTrainingSummaryCreator()
      val actual = x.getSuggestedLabelThreshold(engine, edps, dbletol, MetricClass.Alloy)
      val expected = Seq(
        new LabelFloatMetric(MetricTypes.LabelBestF1Threshold, MetricClass.Alloy, "a", 0.6f),
        new LabelFloatMetric(MetricTypes.LabelBestF1Threshold, MetricClass.Alloy, "b", 0.5f)
      )
      actual.sortBy(x => x.label) shouldBe expected
    }

    it("computes computeBestF1Threshold correctly") {
      val sqlContext = new org.apache.spark.sql.SQLContext(engine.sparkContext)
      val x = new DummyTrainingSummaryCreator()
      val values = Seq((0.5, 1.0), (0.4, 0.0), (0.3, 0.0), (0.5, 1.0), (0.6, 1.0))
      val actual = x.computeBestF1Threshold(engine, sqlContext, values)
      actual shouldBe 0.5f
    }
  }
}
