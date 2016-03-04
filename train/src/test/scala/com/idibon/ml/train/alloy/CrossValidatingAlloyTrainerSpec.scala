package com.idibon.ml.train.alloy

import com.idibon.ml.alloy.{HasTrainingSummary, BaseAlloy}
import com.idibon.ml.common.EmbeddedEngine
import com.idibon.ml.predict.{Classification, Label}
import com.idibon.ml.predict.ml.TrainingSummary
import com.idibon.ml.predict.ml.metrics._
import org.json4s.JsonAST.JObject
import org.json4s._
import org.json4s.native.JsonMethods._
import org.scalatest._

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

/**
  * Tests the Cross Validation Alloy Trainer
  */
class CrossValidatingAlloyTrainerSpec extends FunSpec
  with Matchers with BeforeAndAfter with ParallelTestExecution with BeforeAndAfterAll {

  val engine = new EmbeddedEngine

  describe("getTrainingSummaryCreator tests"){
    it("creates MultiClassMetricsEvaluator") {
      val trainer = new CrossValidatingAlloyTrainerBuilder().build(engine)
      val actual = trainer.getTrainingSummaryCreator(AlloyTrainer.DOCUMENT_MUTUALLY_EXCLUSIVE, 4)
      actual shouldBe new MultiClassMetricsEvaluator(0.25f)
    }
    it("creates MultiLabelMetricsEvaluator") {
      val trainer = new CrossValidatingAlloyTrainerBuilder().build(engine)
      val actual = trainer.getTrainingSummaryCreator(AlloyTrainer.DOCUMENT_MULTI_LABEL, 4)
      actual shouldBe new MultiLabelMetricsEvaluator(0.5f)
    }
  }

  describe("average metrics tests") {
    it("creates averages correctly") {
      val trainingSummaries = Seq[TrainingSummary](
        new TrainingSummary("0", Seq(
          new FloatMetric(MetricTypes.F1, MetricClass.Binary, 1.0f),
          new FloatMetric(MetricTypes.Precision, MetricClass.Binary, 1.0f),
          new FloatMetric(MetricTypes.Recall, MetricClass.Binary, 1.0f)
        )),
        new TrainingSummary("1", Seq(
          new FloatMetric(MetricTypes.F1, MetricClass.Binary, 0.98f),
          new FloatMetric(MetricTypes.Precision, MetricClass.Binary, 0.98f),
          new FloatMetric(MetricTypes.Recall, MetricClass.Binary, 0.98f)
        ))
      )
      val trainer = new CrossValidatingAlloyTrainerBuilder().build(engine)
      val actual = trainer.averageMetrics("testsummary", trainingSummaries)
      val expected = new TrainingSummary("testsummary", Seq(
        new FloatMetric(MetricTypes.F1, MetricClass.Alloy, 0.99f),
        new FloatMetric(MetricTypes.Precision, MetricClass.Alloy, 0.99f),
        new FloatMetric(MetricTypes.Recall, MetricClass.Alloy, 0.99f)
      ))
      actual.identifier shouldBe expected.identifier
      actual.metrics.sortBy(m => m.metricType) shouldBe expected.metrics.sortBy(m => m.metricType)
    }
  }
}
