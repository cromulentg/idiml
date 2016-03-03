package com.idibon.ml.train.alloy

import com.idibon.ml.alloy.{Alloy, BaseAlloy, HasTrainingSummary}
import com.idibon.ml.common.EmbeddedEngine
import com.idibon.ml.predict.Classification
import com.idibon.ml.predict.ml.TrainingSummary
import com.idibon.ml.predict.ml.metrics._
import org.json4s.JsonAST.JObject
import org.json4s._
import org.json4s.native.JsonMethods._
import org.scalatest._

import scala.io.Source

/**
  * Tests the Competitive Alloy Trainer
  */
class CompetitiveAlloyTrainerSpec extends FunSpec
  with Matchers with BeforeAndAfter with ParallelTestExecution with BeforeAndAfterAll {

  val engine = new EmbeddedEngine

  /**
    * Helper method to make dummy alloys with just a training summary.
    * @param name
    * @param trainingSummary
    * @return
    */
  def createAlloy(name: String, trainingSummary: Seq[TrainingSummary]) = {
    new BaseAlloy[Classification](name, Seq(), Map()) with HasTrainingSummary {
      override def getTrainingSummaries = {
        Some(trainingSummary)
      }
    }
  }

  describe("Find Max F1 tests") {
    it("works as intended") {
      val trainer = new CompetitiveAlloyTrainerBuilder().build(engine)
      val averagedResults = Seq[BaseAlloy[Classification] with HasTrainingSummary](
        createAlloy("0",
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
      val trainer = new CompetitiveAlloyTrainerBuilder().build(engine)
      val averagedResults = Seq[BaseAlloy[Classification] with HasTrainingSummary](
        createAlloy("0",
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
  }

  describe("integration test -- tests abstract class too"){
    implicit val formats = DefaultFormats
    val configFile : String = "test_data/trainer_configs/test_competitive_alloy_xval_config.json"
    val trainingFile : String = "test_data/english_social_sentiment/training_small.json"
    val labelsAndRulesFile : String = "test_data/english_social_sentiment/label_rule_config_small.json"
    val predictionsFile : String = "test_data/english_social_sentiment/predictions_small.json"
    val engine = new EmbeddedEngine
    val trainerName: String = "RichardSimmonsII"

    val configFileStream = new java.io.InputStreamReader(getClass.getClassLoader.getResourceAsStream(configFile))
    val trainerConfigJObject = parse(configFileStream).extract[JObject]
    configFileStream.close

    // Import the labels & rules from file
    val line = Source.fromFile(getClass.getClassLoader.getResource(labelsAndRulesFile).getPath())
      .getLines().foldLeft(new StringBuilder())((bld, jsn) => bld.append(jsn)).mkString
    val labelsAndRulesJObject = parse(line).extract[JObject]

    // Create the trainer
    val trainer = AlloyFactory.getTrainer(engine, (trainerConfigJObject \ "trainerConfig").extract[JObject])

    trainer shouldBe an[AlloyTrainer]
    trainer shouldBe an[CompetitiveAlloyTrainer]

    // Train the alloy
    val alloy = trainer.trainAlloy(
      trainerName,
      () => {
        // training data
        Source.fromFile(getClass.getClassLoader.getResource(trainingFile).getPath())
          .getLines.map(line => parse(line).extract[JObject])
      },
      labelsAndRulesJObject,
      Some(trainerConfigJObject)
    )

    alloy shouldBe an[Alloy[_]]
    val summaries = alloy match {
      case o: BaseAlloy[Classification] with HasTrainingSummary => o.getTrainingSummaries
    }
    summaries.get.size shouldBe 3
  }
}
