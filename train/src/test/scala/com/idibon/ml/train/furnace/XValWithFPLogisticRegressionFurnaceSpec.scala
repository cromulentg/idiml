package com.idibon.ml.train.furnace

import com.idibon.ml.alloy.Alloy
import com.idibon.ml.common.EmbeddedEngine
import com.idibon.ml.predict.{PredictOptionsBuilder, PredictOptions, Document, Classification}
import com.idibon.ml.train.alloy.{AlloyTrainer, AlloyFactory}
import com.typesafe.scalalogging.StrictLogging
import org.json4s._
import org.json4s.native.JsonMethods._
import org.scalatest._

import scala.io.Source
import scala.util.{Success, Try}


/**
  * Tests XValWithFPLogisticRegression
  */
class XValWithFPLogisticRegressionFurnaceSpec extends FunSpec
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
  val annotationsFile : String = "test_data/annotations.json"
  val labelsAndRulesFile : String = "test_data/simple_language/label_rule_config.json"
  val predictionsFile : String = "test_data/predictions.json"
  val engine = new EmbeddedEngine
  var trainer : AlloyTrainer = _
  val trainerName: String = "RichardSimmons"
  var trainerConfigJObject: JObject = _
  var labelsAndRulesJObject: JObject = _
  var alloy : Alloy[Classification] = _

  /** Sets up the test object, spark context, & feature pipeline */
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
  }


  // Alloy test
  describe("Alloy tests") {
    it("trains an alloy") {
      alloy = trainer.trainAlloy(
        trainerName,
        () => {
          // training data
          Source.fromFile(getClass.getClassLoader.getResource(annotationsFile).getPath())
            .getLines.map(line => parse(line).extract[JObject])
        },
        labelsAndRulesJObject,
        Some(trainerConfigJObject)
      )

      alloy shouldBe an[Alloy[_]]


      // Test some predictions
      Source.fromFile(getClass.getClassLoader.getResource(predictionsFile).getPath())
        .getLines.toStream.par
        .foreach(line => {
          val document = parse(line).extract[JObject]
          //val builder = new PredictOptionsBuilder().showSignificantFeatures(0.1f)
          val builder = new PredictOptionsBuilder()

          val result = alloy.predict(document, builder.build)
          logger.debug(s"document: $document")

          result.toArray.foreach(r => logger.debug(s"result: " + r))
        })
    }
  }

}



