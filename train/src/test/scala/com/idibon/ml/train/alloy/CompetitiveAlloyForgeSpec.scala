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

import scala.concurrent.Await
import scala.concurrent.duration.Duration
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

  describe("filterToAppropriateSummaries tests") {
    it("handles empty sequence") {
      CompetitiveAlloyForge.filterToAppropriateSummaries(Seq()) shouldBe Seq()
    }
    it("handles filtering non-cross validation summaries & notes not existing") {
      val summaries = Seq(
        new TrainingSummary(s"0${CrossValidatingAlloyTrainer.SUFFIX}", Seq()),
        new TrainingSummary(s"0-NOTPROPERSUFFIX", Seq())
      )
      val expected = Seq(summaries(0))
      CompetitiveAlloyForge.filterToAppropriateSummaries(summaries) shouldBe expected
    }
    it("handles filtering on notes when they exist") {
      val summaries = Seq(
        new TrainingSummary(s"0${CrossValidatingAlloyTrainer.SUFFIX}",
          Seq(new PropertyMetric(MetricTypes.Notes, MetricClass.Binary,
            Seq((AlloyEvaluator.GRANULARITY, Granularity.Token.toString))))),
        new TrainingSummary(s"1${CrossValidatingAlloyTrainer.SUFFIX}",
          Seq(new PropertyMetric(MetricTypes.Notes, MetricClass.Binary,
            Seq((AlloyEvaluator.GRANULARITY, "filter me!"))
          )))
      )
      val expected = Seq(summaries(0))
      CompetitiveAlloyForge.filterToAppropriateSummaries(summaries) shouldBe expected
    }
  }

  describe("Find Max F1 tests") {
    it("works finding maxf1 without any granularity") {
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
      val actual = CompetitiveAlloyForge.findMaxF1[Span](averagedResults)
      val expected = averagedResults(0)
      actual shouldBe expected
    }

    it("ignores training summaries that aren't averages") {
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
      val actual = CompetitiveAlloyForge.findMaxF1[Span](averagedResults)
      val expected = averagedResults(1)
      actual shouldBe expected
    }

    it("works finding maxf1 with granularity") {
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
      val actual = CompetitiveAlloyForge.findMaxF1[Span](averagedResults)
      val expected = averagedResults(0)
      actual shouldBe expected
    }

    it("works when no F1 or macroF1 exists") {
      val averagedResults = Seq[BaseAlloy[Span] with HasTrainingSummary](
        createAlloy[Span]("0",
          Seq(new TrainingSummary(s"0${CrossValidatingAlloyTrainer.SUFFIX}",
            Seq[Metric with Buildable[_, _]](
              new FloatMetric(MetricTypes.Precision, MetricClass.Binary, 0.50f),
              new FloatMetric(MetricTypes.Recall, MetricClass.Binary, 0.50f),
              new PropertyMetric(MetricTypes.Notes, MetricClass.Binary,
                Seq((AlloyEvaluator.GRANULARITY, Granularity.Token.toString)))
            )))),
        createAlloy("1",
          Seq(new TrainingSummary(s"1${CrossValidatingAlloyTrainer.SUFFIX}",
            Seq[Metric with Buildable[_, _]](
              new FloatMetric(MetricTypes.MacroF1, MetricClass.Binary, 0.99f),
              new FloatMetric(MetricTypes.Precision, MetricClass.Binary, 0.99f),
              new FloatMetric(MetricTypes.Recall, MetricClass.Binary, 0.99f),
              new PropertyMetric(MetricTypes.Notes, MetricClass.Binary,
                Seq((AlloyEvaluator.GRANULARITY, Granularity.Document.toString)))
            ))))
      )
      val actual = CompetitiveAlloyForge.findMaxF1[Span](averagedResults)
      val expected = averagedResults(1)
      actual shouldBe expected
    }
  }

  describe("integration test") {
    it("works") {
      def createLabels(alloyConfig: TestAlloyConfiguration): Seq[Label] = {
        alloyConfig.uuid_to_label.map({ case (uuid, name) =>
          new Label(uuid, name)
        }).toSeq
      }
      def createTrainOptions(alloyConfig: TestAlloyConfiguration, data: Option[() => Iterator[JObject]]) = {
        val options = TrainOptions()
        data.map(f => options.addDocuments(f()))
        alloyConfig.rules.nonEmpty match {
          case true => options.addRules(alloyConfig.rules.map(r => (r.label, r.expression, r.weight)))
          case false => ()
        }
        options
      }

      implicit val formats = DefaultFormats
      val configFile : String = "test_data/trainer_configs/test_chain_ner_competitive_config.json"
      val trainingFile : String = "test_data/ner_data/ner_data_small.json"
      val labelsAndRulesFile : String = "test_data/ner_data/ner_config.json"
      val engine = new EmbeddedEngine
      val trainerName: String = "RichardSimmonsIII"

      val configFileStream = new java.io.InputStreamReader(getClass.getClassLoader.getResourceAsStream(configFile))
      val trainerConfigJObject = parse(configFileStream).extract[JObject]
      configFileStream.close

      // Import the labels & rules from file
      val line = Source.fromFile(getClass.getClassLoader.getResource(labelsAndRulesFile).getPath())
        .getLines().foldLeft(new StringBuilder())((bld, jsn) => bld.append(jsn)).mkString
      val labelsAndRulesJObject = parse(line).extract[JObject]

      val alloyConfig = labelsAndRulesJObject.extract[TestAlloyConfiguration]
      val forgeConfig = trainerConfigJObject.extract[TestForgeConfiguration]
      val labels = createLabels(alloyConfig)
      val data = () => {
        // training data
        Source.fromFile(getClass.getClassLoader.getResource(trainingFile).getPath())
          .getLines.map(line => parse(line).extract[JObject])
      }
      val options = createTrainOptions(alloyConfig, Some(data))

      // Create the trainer
      val forge = AlloyForge[Span](
        engine, forgeConfig.forgeName, trainerName, labels.toSeq, forgeConfig.forgeConfig)

      forge shouldBe an[AlloyForge[Span]]
      forge shouldBe an[CompetitiveAlloyForge[Span]]

      // train the alloy
      val alloy = Await.result(
        forge.forge(options.build(labels), forge.getEvaluator(engine, "extraction.bio_ner")),
        Duration.Inf)

      alloy shouldBe an[Alloy[Span]]
      val summaries = alloy match {
        case o: BaseAlloy[Span] with HasTrainingSummary => o.getTrainingSummaries
      }
      summaries.get.size shouldBe 6
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

/** Schema for for label_and_rules (i.e., task / alloy) configuration file
  *
  * @param task_type type of task being trained (classification vs extraction)
  * @param uuid_to_label mapping of UUID strings to the (current) label name
  * @param rules opaque rules data
  */
case class TestAlloyConfiguration(
                               task_type: String,
                               uuid_to_label: Map[String, String],
                               rules: Array[TestRule])

case class TestRule(label: String, expression: String, weight: Float)


/** Schema for JSON training configuration file */
case class TestForgeConfiguration(forgeName: String = "",
                              forgeConfig: JObject = null,
                              trainerConfig: JObject = null,
                              configVersion: String = "0.0.1")
