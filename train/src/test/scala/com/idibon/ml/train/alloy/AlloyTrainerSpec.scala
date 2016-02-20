package com.idibon.ml.train.alloy

import com.idibon.ml.alloy.Alloy
import com.idibon.ml.common.EmbeddedEngine
import com.idibon.ml.predict.{PredictOptionsBuilder, Classification}
import com.typesafe.scalalogging.StrictLogging
import org.json4s._
import org.json4s.native.JsonMethods._
import org.scalatest._
import org.scalatest.Matchers._

import scala.io.Source

/**
  * Tests the Alloy Trainers
  */
class AlloyTrainerSpec extends FunSpec
  with Matchers with BeforeAndAfter with ParallelTestExecution with BeforeAndAfterAll with StrictLogging {

  override def beforeAll = {
    super.beforeAll
  }

  override def afterAll = {
    super.afterAll
  }

  // org.json4s
  implicit val formats = DefaultFormats

  val configFile : String = "test_data/trainer_configs/base_kbinary_xval_with_fp_config.json"
  val trainingFile : String = "test_data/english_social_sentiment/training_small.json"
  val labelsAndRulesFile : String = "test_data/english_social_sentiment/label_rule_config.json"
  val predictionsFile : String = "test_data/english_social_sentiment/predictions_small.json"
  val engine = new EmbeddedEngine
  var trainer : AlloyTrainer = _
  val trainerName: String = "RichardSimmons"
  var trainerConfigJObject: JObject = _
  var labelsAndRulesJObject: JObject = _
  var alloy : Alloy[Classification] = _

  describe("MultiClass1FP tests") {
    //TODO:
  }

  describe("KClass1FP tests") {
    //TODO:
  }

  describe("KClassKFP tests") {
    before {
      // Import pipeline config from file
      val configFileStream = new java.io.InputStreamReader(getClass.getClassLoader.getResourceAsStream(configFile))
      trainerConfigJObject = parse(configFileStream).extract[JObject]
      configFileStream.close

      // Import the labels & rules from file
      val line = Source.fromFile(getClass.getClassLoader.getResource(labelsAndRulesFile).getPath())
        .getLines().foldLeft(new StringBuilder())((bld, jsn) => bld.append(jsn)).mkString
      labelsAndRulesJObject = parse(line).extract[JObject]

      // Create the trainer
      trainer = AlloyFactory.getTrainer(engine, (trainerConfigJObject \ "trainerConfig").extract[JObject])

      trainer shouldBe an[AlloyTrainer]

      // Train the alloy
      alloy = trainer.trainAlloy(
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

    }

    it("makes predictions") {
      val minPrediction : Float = 0.0f
      val maxPrediction : Float = 1.0f

      // Test some predictions
      Source.fromFile(getClass.getClassLoader.getResource(predictionsFile).getPath())
        .getLines.toStream.par
        .foreach(line => {
          val document = parse(line).extract[JObject]
          val builder = new PredictOptionsBuilder().showSignificantFeatures(0.1f)

          val result = alloy.predict(document, builder.build)

          result.toArray.foreach(r => {
            logger.debug(s"result: $r, " + (document \ "content").extract[JValue])

            // Expect the prediction to be reasonable
            r.asInstanceOf[Classification].probability should (be > minPrediction and be < maxPrediction)
          })
        })
    }
  }
}
