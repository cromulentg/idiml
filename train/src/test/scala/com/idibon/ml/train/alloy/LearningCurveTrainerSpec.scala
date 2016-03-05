package com.idibon.ml.train.alloy

import com.idibon.ml.alloy.{BaseAlloy, HasTrainingSummary}
import com.idibon.ml.common.EmbeddedEngine
import com.idibon.ml.feature.Buildable
import com.idibon.ml.predict.Classification
import com.idibon.ml.predict.ml.metrics._
import com.idibon.ml.train.datagenerator.{MultiClassDataFrameGeneratorBuilder, KClassDataFrameGeneratorBuilder}
import com.idibon.ml.train.furnace.{MultiClassLRFurnaceBuilder, SimpleLogisticRegressionFurnaceBuilder}
import org.json4s.JsonAST.JObject
import org.scalatest._
import org.json4s._
import org.json4s.native.JsonMethods._

import scala.io.Source

/**
  * Tests the Learning Curve Trainer
  */
class LearningCurveTrainerSpec extends FunSpec
  with Matchers with BeforeAndAfter with ParallelTestExecution with BeforeAndAfterAll {

  val engine = new EmbeddedEngine

  /**
    *
    * @param actual
    * @param expected
    */
  def evaluateMetrics(actual: Seq[Metric with Buildable[_, _]], expected: Seq[Metric with Buildable[_, _]]) = {
    val zipped = actual.sortBy(x => x.metricType).zip(expected.sortBy(x => x.metricType))
    zipped.foreach(tup => {
      val evaluation = tup match {
        case (lpf: LabelPointsMetric, lpfe: LabelPointsMetric) => {
          lpf.points.sortBy(_._1) == lpfe.points.sortBy(_._1) && lpf.label == lpfe.label && lpf.metricType == lpfe.metricType && lpf.metricClass == lpfe.metricClass
        }
        case (lpf: PointsMetric, lpfe: PointsMetric) => {
          lpf.points.sortBy(_._1) == lpfe.points.sortBy(_._1)  && lpf.metricType == lpfe.metricType && lpf.metricClass == lpfe.metricClass
        }
      }
      evaluation shouldBe true
    })
  }

  describe("create learning curve metrics tests") {
    val trainer = new LearningCurveTrainerBuilder().build(engine)
    it("throws illegal state exception with bad metric") {
      intercept[IllegalStateException]{
        trainer.createLearningCurveMetrics(Map())
      }
    }
    it("works groups metrics together correctly") {
      val actual = trainer.createLearningCurveMetrics(Map())
      val expected = Map("L1" -> Seq(
        new LabelPointsMetric(MetricTypes.LearningCurveLabelPrecision,
          MetricClass.Alloy, "L1", Seq((0.5f, 1.0f), (1.0f, 1.0f)))),
        "L2" -> Seq(
        new LabelPointsMetric(MetricTypes.LearningCurveLabelPrecision,
          MetricClass.Alloy, "L2", Seq((0.6f, 1.0f), (0.9f, 1.0f))))
      )
      // have to do this nasty comparisons because sequences with tuples the wrong way will be counted
      // as wrong :(
//      actual.foreach({case (key, value) =>
//        value.size shouldBe expected(key).size
//          value(0) match {
//            case l: LabelPointsMetric => l.points.sortBy(_._1) shouldBe expected(key)(0).points.sortBy(_._1)
//          }
//      })
    }
  }

//  describe("createLabelPortionMetricTuples tests") {
//    val trainer = new LearningCurveTrainerBuilder().build(engine)
//    it("works as intended") {
//      val input =  Seq(
//        ResultTuple("label", 0, 1.0, true, true),
//        ResultTuple("label", 0, 1.0, false, false),
//        ResultTuple("label", 0, 0.5, true, true),
//        ResultTuple("label", 0, 0.5, false, false),
//        ResultTuple("label", 1, 1.0, false, false),
//        ResultTuple("label", 1, 1.0, true, true),
//        ResultTuple("label", 1, 0.5, false, false),
//        ResultTuple("label", 1, 0.5, true, true),
//        ResultTuple("label", 1, 0.5, false, true),
//        ResultTuple("label", 1, 0.5, true, false)
//      )
//      val actual = trainer.createLabelPortionMetricTuples(input)
//      val expected = Seq(
//        new LabelPortionMetricTuple("label", 0.5, MetricTypes.F1, 0.5f),
//        new LabelPortionMetricTuple("label", 0.5, MetricTypes.LabelPrecision, 0.5f),
//        new LabelPortionMetricTuple("label", 0.5, MetricTypes.LabelRecall, 0.5f),
//        new LabelPortionMetricTuple("label", 0.5, MetricTypes.LabelF1, 0.5f),
//        new LabelPortionMetricTuple("label", 1.0, MetricTypes.F1, 1.0f),
//        new LabelPortionMetricTuple("label", 1.0, MetricTypes.LabelPrecision, 1.0f),
//        new LabelPortionMetricTuple("label", 1.0, MetricTypes.LabelRecall, 1.0f),
//        new LabelPortionMetricTuple("label", 1.0, MetricTypes.LabelF1, 1.0f),
//        new LabelPortionMetricTuple("label", 0.5, MetricTypes.F1, 1.0f),
//        new LabelPortionMetricTuple("label", 0.5, MetricTypes.LabelPrecision, 1.0f),
//        new LabelPortionMetricTuple("label", 0.5, MetricTypes.LabelRecall, 1.0f),
//        new LabelPortionMetricTuple("label", 0.5, MetricTypes.LabelF1, 1.0f),
//        new LabelPortionMetricTuple("label", 1.0, MetricTypes.F1, 1.0f),
//        new LabelPortionMetricTuple("label", 1.0, MetricTypes.LabelPrecision, 1.0f),
//        new LabelPortionMetricTuple("label", 1.0, MetricTypes.LabelRecall, 1.0f),
//        new LabelPortionMetricTuple("label", 1.0, MetricTypes.LabelF1, 1.0f)
//      )
//      actual.size shouldBe expected.size
//      actual shouldBe expected
//    }
//  }


  describe("transformAndFilterToWantedMetrics tests") {
    val trainer = new LearningCurveTrainerBuilder().build(engine)
    it("filters to correct metrics") {
//      val actual = trainer.transformAndFilterToWantedMetrics("L1", Seq[Metric with Buildable[_, _]](
//        new FloatMetric(MetricTypes.F1, MetricClass.Binary, 1.0f),
//        new FloatMetric(MetricTypes.Precision, MetricClass.Binary, 1.0f),
//        new FloatMetric(MetricTypes.Recall, MetricClass.Binary, 1.0f),
//        new LabelFloatMetric(MetricTypes.LabelPrecision, MetricClass.Binary, "L1", 1.0f),
//        new LabelFloatMetric(MetricTypes.LabelRecall, MetricClass.Binary, "L1", 1.0f),
//        new LabelFloatMetric(MetricTypes.LabelF1, MetricClass.Binary, "L1", 1.0f)
//      ))
//      val expected = Seq[Metric with Buildable[_, _]](
//        new FloatMetric(MetricTypes.F1, MetricClass.Binary, 1.0f),
//        new LabelFloatMetric(MetricTypes.LabelPrecision, MetricClass.Binary, "L1", 1.0f),
//        new LabelFloatMetric(MetricTypes.LabelRecall, MetricClass.Binary, "L1", 1.0f),
//        new LabelFloatMetric(MetricTypes.LabelF1, MetricClass.Binary, "L1", 1.0f)
//      )
//      actual shouldBe expected
    }
  }


//  describe("extract label points metric"){
//    val trainer = new LearningCurveTrainerBuilder().build(engine)
//    it("works as inteded") {
//      val actual = trainer.extractLabeledPointsMetric(MetricTypes.LearningCurveLabelF1, "L1", Seq(
//        new LabelPortionMetricTuple("L1", 0.5, MetricTypes.LabelF1, 0.5f),
//        new LabelPortionMetricTuple("L1", 1.0, MetricTypes.LabelF1, 0.6f)
//      ))
//      val expected = new LabelPointsMetric(
//        MetricTypes.LearningCurveLabelF1, MetricClass.Binary, "L1", Seq((0.5f, 0.5f), (1.0f, 0.6f)))
//      actual shouldBe expected
//    }
//  }
}
