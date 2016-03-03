package com.idibon.ml.train.alloy

import org.json4s.JsonAST.JObject
import org.json4s._
import org.json4s.native.JsonMethods._
import org.scalatest._

import scala.io.Source

/**
  * Tests the K Fold Data Set Creator
  */
class KFoldDataSetCreatorSpec extends FunSpec
  with Matchers with BeforeAndAfter with ParallelTestExecution with BeforeAndAfterAll {

  describe("creating the fold streams tests") {

    val trainer = new TestTrainer()
    val inFile : String = "test_data/multiple_points.json"
    val inFilePath = getClass.getClassLoader.getResource(inFile).getPath()
    implicit val formats = org.json4s.DefaultFormats
    val docs = () => Source.fromFile(inFilePath).getLines.map(line => parse(line).extract[JObject])

    it("creates hold out and streams and folds properly") {
      val actual = trainer.createFoldDataSets(docs, 2, Array(0.5, 1.0), 1L)
      actual.size shouldBe 2
      actual(0).fold shouldBe 0
      actual(1).fold shouldBe 1
      actual.foreach(fold => {
        fold.holdout.size shouldBe 5
        fold.trainingStreams.size shouldBe 2
        fold.trainingStreams(0).portion shouldBe 0.5
        fold.trainingStreams(0).stream.size shouldBe 2
        fold.trainingStreams(1).portion shouldBe 1.0
        fold.trainingStreams(1).stream.size shouldBe 5
      })
    }

    it("creates the right createValidationFoldMapping"){
      val actual = trainer.createValidationFoldMapping(docs().toStream, 3, 1L)
      actual shouldBe Array[Short](0, 0, 1, 2, 1, 2, 0, 0, 1, 2)
    }

    it("getKthItems correctly") {
      val actual = trainer.getKthItems(docs().toStream, Stream(0,1,0,1,0,1,0,1,0,1), 1)
      actual.size shouldBe 5
      implicit val formats = org.json4s.DefaultFormats
      (actual.head \ "name").extract[String] shouldBe "1"
      (actual.reverse.head \ "name").extract[String] shouldBe "9"
    }

    it("getAllButKthItems correctly") {
      val actual = trainer.getAllButKthItems(docs().toStream, Stream(0,1,0,1,0,1,0,1,0,1), 0)
      actual.size shouldBe 5
      implicit val formats = org.json4s.DefaultFormats
      (actual.head \ "name").extract[String] shouldBe "1"
      (actual.reverse.head \ "name").extract[String] shouldBe "9"
    }

    it("getPortion correctly") {
      val actual = trainer.getPortion(docs().toStream, 0.4)
      actual.size shouldBe 4
    }

    it("creates training data set objects correctly") {
      val actual = trainer.createFoldDataSets(docs, 2, 0.5, 1L, Map())
      actual.size shouldBe 2
      actual(0).info.fold shouldBe 0
      actual(0).info.portion shouldBe 0.5
      actual(1).info.fold shouldBe 1
      actual(1).info.portion shouldBe 0.5
      actual.foreach(ds => {
        ds.test().size shouldBe 5
        ds.train().size shouldBe 2
      })
    }
  }
}

case class TestTrainer() extends KFoldDataSetCreator {}
