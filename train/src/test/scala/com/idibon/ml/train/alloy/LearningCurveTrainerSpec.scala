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

  describe("Per label learning curve metrics tests") {

    it("works on empty tuples") {
      val trainer = new LearningCurveTrainerBuilder().build(engine)
      val input = Seq()
      trainer.createPerLabelLCMetrics(input) shouldBe Map()
    }

    it("works on single portion metrics") {
      val trainer = new LearningCurveTrainerBuilder().build(engine)
      val input = Seq(ResultTuple("label", 0, 1.0, true, true), ResultTuple("label", 1, 1.0, false, false),
        ResultTuple("label", 0, 1.0, false, false), ResultTuple("label", 1, 1.0, true, true))
      val expected = List[Metric with Buildable[_, _]](
        new LabelPointsMetric(MetricTypes.LearningCurveLabelPrecision, MetricClass.Binary,"label", Seq((1.0f,1.0f))),
        new PointsMetric(MetricTypes.LearningCurveF1, MetricClass.Binary,List((1.0f,1.0f))),
        new LabelPointsMetric(MetricTypes.LearningCurveLabelF1, MetricClass.Binary, "label",Seq((1.0f,1.0f))),
        new LabelPointsMetric(MetricTypes.LearningCurveLabelRecall, MetricClass.Binary,"label",Seq((1.0f,1.0f))))
      evaluateMetrics(trainer.createPerLabelLCMetrics(input)("label"), expected)
    }

    it("works on single fold metrics") {
      val trainer = new LearningCurveTrainerBuilder().build(engine)
      val input = Seq(ResultTuple("label", 0, 1.0, true, true), ResultTuple("label", 0, 0.5, false, false),
        ResultTuple("label", 0, 1.0, false, false), ResultTuple("label", 0, 0.5, true, true))
      val expected = List[Metric with Buildable[_, _]](
        new LabelPointsMetric(MetricTypes.LearningCurveLabelPrecision, MetricClass.Binary,"label", Seq((1.0f,1.0f), (0.5f, 1.0f))),
        new PointsMetric(MetricTypes.LearningCurveF1, MetricClass.Binary,List((1.0f,1.0f), (0.5f, 1.0f))),
        new LabelPointsMetric(MetricTypes.LearningCurveLabelF1, MetricClass.Binary, "label",Seq((1.0f,1.0f), (0.5f, 1.0f))),
        new LabelPointsMetric(MetricTypes.LearningCurveLabelRecall, MetricClass.Binary,"label",Seq((1.0f,1.0f), (0.5f, 1.0f))))
      evaluateMetrics(trainer.createPerLabelLCMetrics(input)("label"), expected)
    }

    it("works on regular metrics with multiple portions and folds") {
      val trainer = new LearningCurveTrainerBuilder().build(engine)
      val input =  Seq(ResultTuple("label", 0, 1.0, true, true), ResultTuple("label", 1, 1.0, false, false),
        ResultTuple("label", 0, 1.0, false, false), ResultTuple("label", 1, 1.0, true, true),
        ResultTuple("label", 0, 0.5, true, true), ResultTuple("label", 1, 0.5, false, false),
        ResultTuple("label", 0, 0.5, false, false), ResultTuple("label", 1, 0.5, true, true),
        ResultTuple("label", 0, 1.0, true, true), ResultTuple("label", 1, 1.0, false, false))

      val expected = List[Metric with Buildable[_, _]](
        new LabelPointsMetric(MetricTypes.LearningCurveLabelPrecision, MetricClass.Binary,"label", Seq((1.0f,1.0f), (0.5f, 1.0f))),
        new PointsMetric(MetricTypes.LearningCurveF1, MetricClass.Binary,List((1.0f,1.0f), (0.5f, 1.0f))),
        new LabelPointsMetric(MetricTypes.LearningCurveLabelF1, MetricClass.Binary, "label",Seq((1.0f,1.0f), (0.5f, 1.0f))),
        new LabelPointsMetric(MetricTypes.LearningCurveLabelRecall, MetricClass.Binary,"label",Seq((1.0f,1.0f), (0.5f, 1.0f))))
      evaluateMetrics(trainer.createPerLabelLCMetrics(input)("label"), expected)
    }
  }

  describe("average across folds tests") {
    val trainer = new LearningCurveTrainerBuilder().build(engine)
    it("averages as expected across label portion &  metrics") {
      val actual = trainer.averageAcrossFolds(Seq(
        new LabelPortionMetricTuple("L1", 1.0, MetricTypes.LabelF1, 0.5f),
        new LabelPortionMetricTuple("L1", 1.0, MetricTypes.LabelF1, 1.0f),
        new LabelPortionMetricTuple("L1", 1.0, MetricTypes.LabelPrecision, 1.0f),
        new LabelPortionMetricTuple("L2", 1.0, MetricTypes.LabelPrecision, 1.0f),
        new LabelPortionMetricTuple("L2", 0.5, MetricTypes.LabelPrecision, 1.0f),
        new LabelPortionMetricTuple("L2", 0.5, MetricTypes.LabelPrecision, 0.0f)
      )).toList.sortBy(x => (x.label, x.portion, x.metric))
      val expected = List(new LabelPortionMetricTuple("L1", 1.0, MetricTypes.LabelF1, 0.75f),
        new LabelPortionMetricTuple("L1", 1.0, MetricTypes.LabelPrecision, 1.0f),
        new LabelPortionMetricTuple("L2", 1.0, MetricTypes.LabelPrecision, 1.0f),
        new LabelPortionMetricTuple("L2", 0.5, MetricTypes.LabelPrecision, 0.5f))
        .sortBy(x => (x.label, x.portion, x.metric))
      actual shouldBe expected
    }
  }

  describe("create learning curve metrics tests") {
    val trainer = new LearningCurveTrainerBuilder().build(engine)
    it("throws illegal state exception with bad metric") {
      intercept[IllegalStateException]{
        trainer.createLearningCurveMetrics(List(new LabelPortionMetricTuple("L1", 1.0, MetricTypes.F1ByThreshold, 1.0f) ))
      }
    }
    it("works groups metrics together correctly") {
      val actual = trainer.createLearningCurveMetrics(List(
        new LabelPortionMetricTuple("L1", 1.0, MetricTypes.LabelPrecision, 1.0f),
        new LabelPortionMetricTuple("L2", 0.9, MetricTypes.LabelPrecision, 1.0f),
        new LabelPortionMetricTuple("L1", 0.5, MetricTypes.LabelPrecision, 1.0f),
        new LabelPortionMetricTuple("L2", 0.6, MetricTypes.LabelPrecision, 1.0f)
      ))
      val expected = Map("L1" -> Seq(
        new LabelPointsMetric(MetricTypes.LearningCurveLabelPrecision,
          MetricClass.Binary, "L1", Seq((0.5f, 1.0f), (1.0f, 1.0f)))),
        "L2" -> Seq(
        new LabelPointsMetric(MetricTypes.LearningCurveLabelPrecision,
          MetricClass.Binary, "L2", Seq((0.6f, 1.0f), (0.9f, 1.0f))))
      )
      // have to do this nasty comparisons because sequences with tuples the wrong way will be counted
      // as wrong :(
      actual.foreach({case (key, value) =>
        value.size shouldBe expected(key).size
          value(0) match {
            case l: LabelPointsMetric => l.points.sortBy(_._1) shouldBe expected(key)(0).points.sortBy(_._1)
          }
      })
    }
  }

  describe("createLabelPortionMetricTuples tests") {
    val trainer = new LearningCurveTrainerBuilder().build(engine)
    it("works as intended") {
      val input =  Seq(
        ResultTuple("label", 0, 1.0, true, true),
        ResultTuple("label", 0, 1.0, false, false),
        ResultTuple("label", 0, 0.5, true, true),
        ResultTuple("label", 0, 0.5, false, false),
        ResultTuple("label", 1, 1.0, false, false),
        ResultTuple("label", 1, 1.0, true, true),
        ResultTuple("label", 1, 0.5, false, false),
        ResultTuple("label", 1, 0.5, true, true),
        ResultTuple("label", 1, 0.5, false, true),
        ResultTuple("label", 1, 0.5, true, false)
      )
      val actual = trainer.createLabelPortionMetricTuples(input)
      val expected = Seq(
        new LabelPortionMetricTuple("label", 0.5, MetricTypes.F1, 0.5f),
        new LabelPortionMetricTuple("label", 0.5, MetricTypes.LabelPrecision, 0.5f),
        new LabelPortionMetricTuple("label", 0.5, MetricTypes.LabelRecall, 0.5f),
        new LabelPortionMetricTuple("label", 0.5, MetricTypes.LabelF1, 0.5f),
        new LabelPortionMetricTuple("label", 1.0, MetricTypes.F1, 1.0f),
        new LabelPortionMetricTuple("label", 1.0, MetricTypes.LabelPrecision, 1.0f),
        new LabelPortionMetricTuple("label", 1.0, MetricTypes.LabelRecall, 1.0f),
        new LabelPortionMetricTuple("label", 1.0, MetricTypes.LabelF1, 1.0f),
        new LabelPortionMetricTuple("label", 0.5, MetricTypes.F1, 1.0f),
        new LabelPortionMetricTuple("label", 0.5, MetricTypes.LabelPrecision, 1.0f),
        new LabelPortionMetricTuple("label", 0.5, MetricTypes.LabelRecall, 1.0f),
        new LabelPortionMetricTuple("label", 0.5, MetricTypes.LabelF1, 1.0f),
        new LabelPortionMetricTuple("label", 1.0, MetricTypes.F1, 1.0f),
        new LabelPortionMetricTuple("label", 1.0, MetricTypes.LabelPrecision, 1.0f),
        new LabelPortionMetricTuple("label", 1.0, MetricTypes.LabelRecall, 1.0f),
        new LabelPortionMetricTuple("label", 1.0, MetricTypes.LabelF1, 1.0f)
      )
      actual.size shouldBe expected.size
      actual shouldBe expected
    }
  }


  describe("Filter metrics tests") {
    val trainer = new LearningCurveTrainerBuilder().build(engine)
    it("filters to correct metrics") {
      val actual = trainer.filterMetrics("L1", Seq[Metric with Buildable[_, _]](
        new FloatMetric(MetricTypes.F1, MetricClass.Binary, 1.0f),
        new FloatMetric(MetricTypes.Precision, MetricClass.Binary, 1.0f),
        new FloatMetric(MetricTypes.Recall, MetricClass.Binary, 1.0f),
        new LabelFloatMetric(MetricTypes.LabelPrecision, MetricClass.Binary, "L1", 1.0f),
        new LabelFloatMetric(MetricTypes.LabelRecall, MetricClass.Binary, "L1", 1.0f),
        new LabelFloatMetric(MetricTypes.LabelF1, MetricClass.Binary, "L1", 1.0f)
      ))
      val expected = Seq[Metric with Buildable[_, _]](
        new FloatMetric(MetricTypes.F1, MetricClass.Binary, 1.0f),
        new LabelFloatMetric(MetricTypes.LabelPrecision, MetricClass.Binary, "L1", 1.0f),
        new LabelFloatMetric(MetricTypes.LabelRecall, MetricClass.Binary, "L1", 1.0f),
        new LabelFloatMetric(MetricTypes.LabelF1, MetricClass.Binary, "L1", 1.0f)
      )
      actual shouldBe expected
    }
    it("filters to correct labels") {
      val actual = trainer.filterMetrics("L1", Seq[Metric with Buildable[_, _]](
        new FloatMetric(MetricTypes.F1, MetricClass.Binary, 1.0f),
        new FloatMetric(MetricTypes.Precision, MetricClass.Binary, 1.0f),
        new FloatMetric(MetricTypes.Recall, MetricClass.Binary, 1.0f),
        new LabelFloatMetric(MetricTypes.LabelPrecision, MetricClass.Binary, "L2", 1.0f),
        new LabelFloatMetric(MetricTypes.LabelRecall, MetricClass.Binary, "L2", 1.0f),
        new LabelFloatMetric(MetricTypes.LabelF1, MetricClass.Binary, "L1", 1.0f)
      ))
      val expected = Seq[Metric with Buildable[_, _]](
        new FloatMetric(MetricTypes.F1, MetricClass.Binary, 1.0f),
        new LabelFloatMetric(MetricTypes.LabelF1, MetricClass.Binary, "L1", 1.0f)
      )
      actual shouldBe expected
    }
  }

  describe("createDataForRDD tests") {
    val trainer = new LearningCurveTrainerBuilder().build(engine)
    it("correctly produces sequence of data ") {
      trainer.createDataForRDD(Seq(
        ResultTuple("label", 0, 1.0, true, true),
        ResultTuple("label", 0, 1.0, false, false),
        ResultTuple("label", 0, 0.5, true, true),
        ResultTuple("label", 0, 0.5, false, false),
        ResultTuple("label", 1, 1.0, false, false),
        ResultTuple("label", 1, 1.0, true, true),
        ResultTuple("label", 1, 0.5, false, false),
        ResultTuple("label", 1, 0.5, true, true),
        ResultTuple("label", 1, 0.5, false, true),
        ResultTuple("label", 1, 0.5, true, false)
      )) shouldBe Seq((1.0, 1.0), (0.0, 0.0), (1.0, 1.0), (0.0, 0.0), (0.0, 0.0),
        (1.0, 1.0), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0))
    }
  }

  describe("extract label points metric"){
    val trainer = new LearningCurveTrainerBuilder().build(engine)
    it("works as inteded") {
      val actual = trainer.extractLabeledPointsMetric(MetricTypes.LearningCurveLabelF1, "L1", Seq(
        new LabelPortionMetricTuple("L1", 0.5, MetricTypes.LabelF1, 0.5f),
        new LabelPortionMetricTuple("L1", 1.0, MetricTypes.LabelF1, 0.6f)
      ))
      val expected = new LabelPointsMetric(
        MetricTypes.LearningCurveLabelF1, MetricClass.Binary, "L1", Seq((0.5f, 0.5f), (1.0f, 0.6f)))
      actual shouldBe expected
    }
  }

  describe("createResultsForAggregation tests") {
    val trainer = new LearningCurveTrainerBuilder().build(engine)
    it("creates a flat list as expected") {
      val fold0 = new Fold(0, Stream(), Seq())
      val fold1 = new Fold(1, Stream(), Seq())
      val pPredictions0 = Seq(
        new PortionPredictions(0.5,
          Seq(new Prediction("l1", true, true), new Prediction("l1", false, false))),
        new PortionPredictions(1.0,
          Seq(new Prediction("l1", true, true), new Prediction("l1", false, false))))
      val pPredictions1 = Seq(
        new PortionPredictions(0.5,
          Seq(new Prediction("l1", true, true), new Prediction("l1", false, false))),
        new PortionPredictions(1.0,
          Seq(new Prediction("l1", true, true), new Prediction("l1", false, false))))
      val actual = trainer.createResultsForAggregation(Seq(
        (fold0, Stream(pPredictions0)),
        (fold1, Stream(pPredictions1)
      )))
      val expected = Seq(
        ResultTuple("l1",0,0.5,true,true),
        ResultTuple("l1",0,0.5,false,false),
        ResultTuple("l1",0,1.0,true,true),
        ResultTuple("l1",0,1.0,false,false),
        ResultTuple("l1",1,0.5,true,true),
        ResultTuple("l1",1,0.5,false,false),
        ResultTuple("l1",1,1.0,true,true),
        ResultTuple("l1",1,1.0,false,false))

      actual shouldBe expected
    }
  }

  describe("create gold labels tests") {
    val trainer = new LearningCurveTrainerBuilder().build(engine)
    it("creates labels as expected") {
      implicit val formats = org.json4s.DefaultFormats
      val actual = trainer.createGoldLabels(parse(
        """{ "content": "Who drives a chevy malibu? Would you recommend it?",
           "metadata": { "iso_639_1": "en" },
           "annotations": [
           { "label": { "name": "aaa" }, "isPositive": true },
           { "label": { "name": "bbb" }, "isPositive": false },
            ] }
        """).extract[JObject])
      val expected = Map("aaa" -> true, "bbb" -> false)
      actual shouldBe expected
    }
  }

//  describe("integration test") {
//    val inFile : String = "test_data/multiple_points.json"
//    val inFilePath = getClass.getClassLoader.getResource(inFile).getPath()
//    val configFile : String = "test_data/pipeline_config.json"
//    val configFilePath = getClass.getClassLoader.getResource(configFile).getPath()
//    val labelFile : String = "test_data/english_social_sentiment/label_rule_config.json"
//    val labelFilePath = getClass.getClassLoader.getResource(labelFile).getPath()
//    implicit val formats = org.json4s.DefaultFormats
//    val docs = () => Source.fromFile(inFilePath).getLines.map(line => parse(line).extract[JObject])
//    val labelsAndRules = parse(Source.fromFile(labelFilePath).reader()).extract[JObject]
//    val config = parse(Source.fromFile(configFilePath).reader()).extract[JObject]
//
//    it("works with k-class trainer") {
//      val kclass = new KClass1FPBuilder(new KClassDataFrameGeneratorBuilder(),
//        new SimpleLogisticRegressionBuilder(2),
//        true)
//      val lcTrainerBuilder = new LearningCurveTrainerBuilder(kclass, 2, Array(0.5, 1.0), 1L)
//      val lcTrainer = lcTrainerBuilder.build(engine)
//      val alloy = lcTrainer.trainAlloy("t1", docs, labelsAndRules, Some(config))
//      val summaries = alloy match {
//        case a: BaseAlloy[Classification] with HasTrainingSummary => a.getTrainingSummaries
//        case o => None
//      }
//      summaries
//    }
//
//    it("works with multi-class trainer") {
//      val kclass = new MultiClass1FPBuilder(new MultiClassDataFrameGeneratorBuilder(),
//        new MultiClassLRFurnaceBuilder(2))
//      val lcTrainerBuilder = new LearningCurveTrainerBuilder(kclass, 2, Array(0.5, 1.0), 1L)
//      val lcTrainer = lcTrainerBuilder.build(engine)
//      val alloy = lcTrainer.trainAlloy("t1", docs, labelsAndRules, Some(config))
//      val summaries = alloy match {
//        case a: BaseAlloy[Classification] with HasTrainingSummary => a.getTrainingSummaries
//        case o => None
//      }
//      summaries
//    }
//  }
}
