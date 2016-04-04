package com.idibon.ml.train.alloy

import com.idibon.ml.alloy.{BaseAlloy, HasTrainingSummary}
import com.idibon.ml.common.EmbeddedEngine
import com.idibon.ml.feature.Buildable
import com.idibon.ml.predict.{Span, Classification}
import com.idibon.ml.predict.ml.TrainingSummary
import com.idibon.ml.predict.ml.metrics._
import com.idibon.ml.train.alloy.evaluation.{Granularity, AlloyEvaluator}
import org.scalatest._

/**
  * Tests the Learning Curve Alloy Forge
  */
class LearningCurveAlloyForgeSpec extends FunSpec
  with Matchers with BeforeAndAfter with ParallelTestExecution with BeforeAndAfterAll {

  val engine = new EmbeddedEngine

  /**
    * Helper method to create a sequence of label float metrics of a single metric type.
    *
    * @param metricType
    * @return
    */
  def createLabelMetricSequence(metricType: MetricTypes) = {
    Seq(
      (0.5, new LabelFloatMetric(metricType, MetricClass.Binary, "L1", 1.0f)),
      (1.0, new LabelFloatMetric(metricType, MetricClass.Binary, "L1", 1.0f)),
      (0.6, new LabelFloatMetric(metricType, MetricClass.Binary, "L2", 1.0f)),
      (0.9, new LabelFloatMetric(metricType, MetricClass.Binary, "L2", 1.0f))
    )
  }

  describe("create learning curve metrics tests") {

    val forge = new LearningCurveAlloyForge(engine, "test", Seq(), null, 2, Array(0.5), 1L)

    it("throws illegal state exception with bad metric") {
      intercept[IllegalStateException]{
        forge.createLearningCurveMetrics(Map(MetricTypes.WeightedPrecision -> Seq()))
      }
    }
    it("can create LearningCurveLabelPrecision from LabelPrecision") {
      val actual = forge.createLearningCurveMetrics(Map(
        MetricTypes.LabelPrecision -> createLabelMetricSequence(MetricTypes.LabelPrecision)))
      val expected = Seq(
        new LabelPointsMetric(MetricTypes.LearningCurveLabelPrecision,
          MetricClass.Alloy, "L1", Seq((0.5f, 1.0f), (1.0f, 1.0f))),
        new LabelPointsMetric(MetricTypes.LearningCurveLabelPrecision,
          MetricClass.Alloy, "L2", Seq((0.6f, 1.0f), (0.9f, 1.0f))))
      actual.sortBy(x => x.asInstanceOf[LabelPointsMetric].label) shouldBe expected
    }
    it("can create LearningCurveLabelF1 from LabelF1") {
      val actual = forge.createLearningCurveMetrics(Map(
        MetricTypes.LabelF1 -> createLabelMetricSequence(MetricTypes.LabelF1)))
      val expected = Seq(
        new LabelPointsMetric(MetricTypes.LearningCurveLabelF1,
          MetricClass.Alloy, "L1", Seq((0.5f, 1.0f), (1.0f, 1.0f))),
        new LabelPointsMetric(MetricTypes.LearningCurveLabelF1,
          MetricClass.Alloy, "L2", Seq((0.6f, 1.0f), (0.9f, 1.0f))))
      actual.sortBy(x => x.asInstanceOf[LabelPointsMetric].label) shouldBe expected
    }
    it("can create LearningCurveLabelRecall from LabelRecall") {
      val actual = forge.createLearningCurveMetrics(Map(
        MetricTypes.LabelRecall -> createLabelMetricSequence(MetricTypes.LabelRecall)))
      val expected = Seq(
        new LabelPointsMetric(MetricTypes.LearningCurveLabelRecall,
          MetricClass.Alloy, "L1", Seq((0.5f, 1.0f), (1.0f, 1.0f))),
        new LabelPointsMetric(MetricTypes.LearningCurveLabelRecall,
          MetricClass.Alloy, "L2", Seq((0.6f, 1.0f), (0.9f, 1.0f))))
      actual.sortBy(x => x.asInstanceOf[LabelPointsMetric].label) shouldBe expected
    }
    it("can create LearningCurveF1 from F1") {
      val actual = forge.createLearningCurveMetrics(Map(
        MetricTypes.F1 -> Seq(
          (0.5, new FloatMetric(MetricTypes.F1, MetricClass.Binary, 0.5f)),
          (0.6, new FloatMetric(MetricTypes.F1, MetricClass.Binary, 0.6f)),
          (0.9, new FloatMetric(MetricTypes.F1, MetricClass.Binary, 0.7f)),
          (1.0, new FloatMetric(MetricTypes.F1, MetricClass.Binary, 1.0f))
        )))
      val expected = Seq(
        new PointsMetric(MetricTypes.LearningCurveF1,
          MetricClass.Alloy, Seq((0.5f, 0.5f), (0.6f, 0.6f), (0.9f, 0.7f), (1.0f, 1.0f))))
      actual.sortBy(x => x.asInstanceOf[LabelPointsMetric].label) shouldBe expected
    }
  }

  describe("createLabelPointsMetrics tests") {
    it("creates label points metrics") {
      val forge = new LearningCurveAlloyForge(engine, "test", Seq(), null, 2, Array(0.5), 1L)
      val actual = forge.createLabelPointsMetrics(
        MetricTypes.LearningCurveLabelF1,
        MetricClass.Alloy,
        createLabelMetricSequence(MetricTypes.LabelF1))
      val expected = Seq(
        new LabelPointsMetric(MetricTypes.LearningCurveLabelF1,
          MetricClass.Alloy, "L1", Seq((0.5f, 1.0f), (1.0f, 1.0f))),
        new LabelPointsMetric(MetricTypes.LearningCurveLabelF1,
          MetricClass.Alloy, "L2", Seq((0.6f, 1.0f), (0.9f, 1.0f))))
      actual.sortBy(x => x.label) shouldBe expected
    }
  }

  describe("transformAndFilterToWantedMetrics tests"){
    val forge = new LearningCurveAlloyForge(engine, "test", Seq(), null, 2, Array(0.5), 1L)

    /**
      * Helper method to create a metric.
      *
      * @param mt
      * @param mc
      * @return
      */
    def createMetric(mt: MetricTypes, mc: MetricClass.Value): Metric with Buildable[_, _] = {
      mt match {
        case m if m == MetricTypes.F1 => new FloatMetric(mt, mc, 0.4f)
        case m if m == MetricTypes.LabelPrecision => new LabelFloatMetric(mt, mc, "a", 0.4f)
        case m if m == MetricTypes.LabelRecall => new LabelFloatMetric(mt, mc, "a", 0.4f)
        case m if m == MetricTypes.LabelF1 => new LabelFloatMetric(mt, mc, "a", 0.4f)
        case m if m == MetricTypes.Precision => new FloatMetric(mt, mc, 0.4f)
      }
    }
    /**
      * Helper method to create some metrics.
      *
      * @param metricTypes
      * @param metricClasses
      * @return
      */
    def createSomeMetrics(metricTypes: Seq[MetricTypes],
                          metricClasses: Seq[MetricClass.Value]): Seq[Metric with Buildable[_, _]] = {
        metricTypes.flatMap(mt => {
          metricClasses.map(mc => {
            createMetric(mt, mc)
          })
        })
    }

    it("filters correct metrics") {
      val metrics = createSomeMetrics(
        Seq(MetricTypes.Precision, MetricTypes.F1, MetricTypes.LabelRecall,
          MetricTypes.LabelF1, MetricTypes.LabelPrecision),
        Seq(MetricClass.Alloy)
      )
      val ts = new TrainingSummary("0.5", metrics)
      val actual = forge.transformAndFilterToWantedMetrics(Seq((0.5, ts)))
      val expected: Map[MetricTypes, Seq[(Double, Metric with Buildable[_, _])]] = Map(
        MetricTypes.F1 ->
          Seq((0.5, createSomeMetrics(Seq(MetricTypes.F1), Seq(MetricClass.Alloy)).head)),
        MetricTypes.LabelRecall ->
          Seq((0.5, createSomeMetrics(Seq(MetricTypes.LabelRecall), Seq(MetricClass.Alloy)).head)),
        MetricTypes.LabelF1 ->
          Seq((0.5, createSomeMetrics(Seq(MetricTypes.LabelF1), Seq(MetricClass.Alloy)).head)),
        MetricTypes.LabelPrecision ->
          Seq((0.5, createSomeMetrics(Seq(MetricTypes.LabelPrecision), Seq(MetricClass.Alloy)).head))
      )
      actual.size shouldBe expected.size
      actual.foreach({case (key, value) =>
        value shouldBe expected(key)
      })
    }
    it("filters correct metric class"){
      val metrics1 = createSomeMetrics(
        Seq(MetricTypes.F1, MetricTypes.LabelRecall,
          MetricTypes.LabelF1, MetricTypes.LabelPrecision),
        Seq(MetricClass.Binary))
      val metrics2 = createSomeMetrics(
        Seq(MetricTypes.LabelF1, MetricTypes.LabelPrecision),
        Seq(MetricClass.Alloy))
      val ts1 = new TrainingSummary("0.5", metrics1)
      val ts2 = new TrainingSummary("0.6", metrics2)
      val actual = forge.transformAndFilterToWantedMetrics(Seq((0.5, ts1), (0.6, ts2)))
      val expected: Map[MetricTypes, Seq[(Double, Metric with Buildable[_, _])]] = Map(
        MetricTypes.LabelF1 ->
          Seq((0.6, createSomeMetrics(Seq(MetricTypes.LabelF1), Seq(MetricClass.Alloy)).head)),
        MetricTypes.LabelPrecision ->
          Seq((0.6, createSomeMetrics(Seq(MetricTypes.LabelPrecision), Seq(MetricClass.Alloy)).head))
      )
      actual.size shouldBe expected.size
      actual.foreach({case (key, value) =>
        value shouldBe expected(key)
      })
    }
    it("groups metric types properly") {
      val metrics1 = createSomeMetrics(
        Seq(MetricTypes.LabelF1),
        Seq(MetricClass.Alloy))
      val metrics2 = createSomeMetrics(
        Seq(MetricTypes.LabelF1),
        Seq(MetricClass.Alloy))
      val metrics3 = createSomeMetrics(
        Seq(MetricTypes.LabelF1),
        Seq(MetricClass.Alloy))
      val ts1 = new TrainingSummary("0.5", metrics1)
      val ts2 = new TrainingSummary("0.6", metrics2)
      val ts3 = new TrainingSummary("0.7", metrics3)
      val actual = forge.transformAndFilterToWantedMetrics(
        Seq((0.5, ts1),
          (0.6, ts2),
          (0.7, ts3)))
      val expected: Map[MetricTypes, Seq[(Double, Metric with Buildable[_, _])]] = Map(
        MetricTypes.LabelF1 ->
          Seq(
            (0.5, createSomeMetrics(Seq(MetricTypes.LabelF1), Seq(MetricClass.Alloy)).head),
            (0.6, createSomeMetrics(Seq(MetricTypes.LabelF1), Seq(MetricClass.Alloy)).head),
            (0.7, createSomeMetrics(Seq(MetricTypes.LabelF1), Seq(MetricClass.Alloy)).head)
          )
      )
      actual.size shouldBe expected.size
      actual.foreach({case (key, value) =>
        value shouldBe expected(key)
      })
    }
  }

  describe("getting portion summaries from alloy tests") {
    val forge = new LearningCurveAlloyForge[Span](engine, "test", Seq(), null, 2, Array(0.5), 1L)

    def createSummaries(name1: String, name2: String) = {
      Seq(
        new TrainingSummary(name1,
          Seq[Metric with Buildable[_, _]](
            new LabelFloatMetric(MetricTypes.LabelF1, MetricClass.Alloy, "L1", 1.0f),
            new PropertyMetric(MetricTypes.Notes, MetricClass.Alloy,
              Seq((AlloyEvaluator.GRANULARITY, Granularity.Document.toString))))),
        new TrainingSummary(name2,
          Seq[Metric with Buildable[_, _]](
            new LabelFloatMetric(MetricTypes.LabelPrecision, MetricClass.Alloy, "L1", 1.0f),
            new PropertyMetric(MetricTypes.Notes, MetricClass.Alloy,
              Seq((AlloyEvaluator.GRANULARITY, Granularity.Document.toString)))))
      )
    }
    it("ignores non cross validation summaries") {
      val summaries1 = createSummaries("1", "2")
      val alloy1 = new BaseAlloy[Span]("name", Seq(), Map()) with HasTrainingSummary {
        override def getTrainingSummaries: Option[Seq[TrainingSummary]] = {
          Some(summaries1)
        }
      }
      forge.getXValPortionSummaries(Seq((0.5, alloy1))) shouldBe Seq()
    }

    it("gets all training summaries") {
      val summaries1 = createSummaries(
        s"1${CrossValidatingAlloyTrainer.SUFFIX}", s"2${CrossValidatingAlloyTrainer.SUFFIX}")
      val alloy1 = new BaseAlloy[Span]("name", Seq(), Map()) with HasTrainingSummary {
        override def getTrainingSummaries: Option[Seq[TrainingSummary]] = {
          Some(summaries1)
        }
      }
      val summaries2 = createSummaries(
        s"3${CrossValidatingAlloyTrainer.SUFFIX}", s"4${CrossValidatingAlloyTrainer.SUFFIX}")
      val alloy2 = new BaseAlloy[Span]("name", Seq(), Map()) with HasTrainingSummary {
        override def getTrainingSummaries: Option[Seq[TrainingSummary]] = {
          Some(summaries2)
        }
      }
      val alloy3 = new BaseAlloy[Span]("name", Seq(), Map()) with HasTrainingSummary {}
      val actual = forge.getXValPortionSummaries(Seq(
        (0.5, alloy1),
        (0.6, alloy2),
        (0.7, alloy3)
      ))
      val expected = Seq(
        (0.5, summaries1(0)),
        (0.5, summaries1(1)),
        (0.6, summaries2(0)),
        (0.6, summaries2(1))
      )
      actual.sortBy(x => (x._1, x._2.identifier)) shouldBe expected
    }
  }
}
