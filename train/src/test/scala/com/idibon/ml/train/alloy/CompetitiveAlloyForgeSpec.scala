package com.idibon.ml.train.alloy

import com.idibon.ml.alloy.{Alloy, BaseAlloy, HasTrainingSummary}
import com.idibon.ml.common
import com.idibon.ml.common.EmbeddedEngine
import com.idibon.ml.feature.{Builder, Buildable}
import com.idibon.ml.predict.{PredictResult, Span, Label, Classification}
import com.idibon.ml.predict.ml.TrainingSummary
import com.idibon.ml.predict.ml.metrics._
import com.idibon.ml.train.TrainOptions
import com.idibon.ml.train.alloy.evaluation.{Granularity, AlloyEvaluator}
import org.json4s.JsonAST.JObject
import org.json4s._
import org.json4s.native.JsonMethods._
import org.scalatest._

import scala.io.Source

/**
  * Tests the Competitive Alloy Forge
  */
class CompetitiveAlloyForgeSpec extends FunSpec
  with Matchers with BeforeAndAfter with ParallelTestExecution with BeforeAndAfterAll {

  val engine = new EmbeddedEngine

  /**
    * Helper method to make dummy alloys with just a training summary.
    *
    * @param name
    * @param trainingSummary
    * @return
    */
  def createAlloy[T <: PredictResult with Buildable[T, Builder[T]]](
    name: String, trainingSummary: Seq[TrainingSummary]) = {
    new BaseAlloy[T](name, Seq(), Map()) with HasTrainingSummary {
      override def getTrainingSummaries = {
        Some(trainingSummary)
      }
    }
  }

  describe("Find Max F1 tests") {
    it("works finding maxf1 without any granularity") {
      val trainer = new CompetitiveAlloyForge[Span](engine,
        "test", Seq(), Map("0" -> new DummyCompetingForge(), "1" -> new DummyCompetingForge()),
        2, 1L)
      val averagedResults = Seq[BaseAlloy[Span] with HasTrainingSummary](
        createAlloy[Span]("0",
        Seq(new TrainingSummary(s"0${CrossValidatingAlloyTrainer.SUFFIX}", Seq(
          new FloatMetric(MetricTypes.F1, MetricClass.Binary, 1.0f),
          new FloatMetric(MetricTypes.Precision, MetricClass.Binary, 1.0f),
          new FloatMetric(MetricTypes.Recall, MetricClass.Binary, 1.0f)
        )))),
        createAlloy("1",
        Seq(new TrainingSummary(s"1${CrossValidatingAlloyTrainer.SUFFIX}", Seq(
          new FloatMetric(MetricTypes.F1, MetricClass.Binary, 0.99f),
          new FloatMetric(MetricTypes.Precision, MetricClass.Binary, 0.99f),
          new FloatMetric(MetricTypes.Recall, MetricClass.Binary, 0.99f)
        ))))
      )
      val actual = trainer.findMaxF1(averagedResults)
      val expected = averagedResults(0)
      actual shouldBe expected
    }

    it("ignores training summaries that aren't averages") {
      val trainer = new CompetitiveAlloyForge[Span](engine,
        "test", Seq(), Map("0" -> new DummyCompetingForge(), "1" -> new DummyCompetingForge()),
        2, 1L)
      val averagedResults = Seq[BaseAlloy[Span] with HasTrainingSummary](
        createAlloy[Span]("0",
          Seq(new TrainingSummary(s"0-NOTPROPERSUFFIX", Seq(
            new FloatMetric(MetricTypes.F1, MetricClass.Binary, 1.0f),
            new FloatMetric(MetricTypes.Precision, MetricClass.Binary, 1.0f),
            new FloatMetric(MetricTypes.Recall, MetricClass.Binary, 1.0f)
          )),
          new TrainingSummary(s"0${CrossValidatingAlloyTrainer.SUFFIX}", Seq(
            new FloatMetric(MetricTypes.F1, MetricClass.Binary, 0.5f),
            new FloatMetric(MetricTypes.Precision, MetricClass.Binary, 0.4f),
            new FloatMetric(MetricTypes.Recall, MetricClass.Binary, 0.3f)
          )))),
        createAlloy("1",
          Seq(new TrainingSummary(s"1${CrossValidatingAlloyTrainer.SUFFIX}", Seq(
            new FloatMetric(MetricTypes.F1, MetricClass.Binary, 0.99f),
            new FloatMetric(MetricTypes.Precision, MetricClass.Binary, 0.99f),
            new FloatMetric(MetricTypes.Recall, MetricClass.Binary, 0.99f)
          ))))
      )
      val actual = trainer.findMaxF1(averagedResults)
      val expected = averagedResults(1)
      actual shouldBe expected
    }

    it("works finding maxf1 with granularity") {
      val trainer = new CompetitiveAlloyForge[Span](engine,
        "test", Seq(), Map("0" -> new DummyCompetingForge(), "1" -> new DummyCompetingForge()),
        2, 1L)
      val averagedResults = Seq[BaseAlloy[Span] with HasTrainingSummary](
        createAlloy[Span]("0",
          Seq(new TrainingSummary(s"0${CrossValidatingAlloyTrainer.SUFFIX}",
            Seq[Metric with Buildable[_, _]](
              new FloatMetric(MetricTypes.F1, MetricClass.Binary, 0.50f),
              new FloatMetric(MetricTypes.Precision, MetricClass.Binary, 0.50f),
              new FloatMetric(MetricTypes.Recall, MetricClass.Binary, 0.50f),
              new PropertyMetric(MetricTypes.Notes, MetricClass.Binary,
                Seq((AlloyEvaluator.GRANULARITY, Granularity.Token.toString)))
            )))),
        createAlloy("1",
          Seq(new TrainingSummary(s"1${CrossValidatingAlloyTrainer.SUFFIX}",
            Seq[Metric with Buildable[_, _]](
              new FloatMetric(MetricTypes.F1, MetricClass.Binary, 0.99f),
              new FloatMetric(MetricTypes.Precision, MetricClass.Binary, 0.99f),
              new FloatMetric(MetricTypes.Recall, MetricClass.Binary, 0.99f),
              new PropertyMetric(MetricTypes.Notes, MetricClass.Binary,
                Seq((AlloyEvaluator.GRANULARITY, "filter me!")))
            ))))
      )
      val actual = trainer.findMaxF1(averagedResults)
      val expected = averagedResults(0)
      actual shouldBe expected
    }
  }
}

class DummyCompetingForge extends AlloyForge[Span] {
  override val name: String = "dummy"

  /**
    * Synchronous method that creates the alloy.
    *
    * @param options
    * @param evaluator
    * @return
    */
  override def doForge(options: TrainOptions, evaluator: AlloyEvaluator): Alloy[Span] = ???

  /**
    * Returns the appropriate evaluator for this alloy.
    *
    * @param engine
    * @param taskType
    * @return
    */
  override def getEvaluator(engine: common.Engine, taskType: String): AlloyEvaluator = ???

  override val labels: Seq[Label] = Seq()
}
