package com.idibon.ml.train.furnace

import scala.util.{Success, Failure}
import scala.concurrent.Await

import com.idibon.ml.alloy._
import com.idibon.ml.feature._
import com.idibon.ml.predict._
import com.idibon.ml.train.TrainOptions
import com.idibon.ml.common.EmbeddedEngine

import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.native.JsonMethods.parse

import org.scalatest.{Matchers, FunSpec}

class ChainNERFurnaceSpec extends FunSpec with Matchers {

  val sequenceConfig = (
    ("transforms" -> List(
      (("name" -> "contentType") ~
        ("class" -> "com.idibon.ml.feature.contenttype.ContentTypeDetector")),
      (("name" -> "lang") ~
        ("class" -> "com.idibon.ml.feature.language.LanguageDetector")),
      (("name" -> "content") ~
        ("class" -> "com.idibon.ml.feature.ContentExtractor")),
      (("name" -> "tokenizer") ~
        ("class" -> "com.idibon.ml.feature.tokenizer.ChainTokenTransformer") ~
        ("config" -> ("accept" -> List("Word")))))) ~
    ("pipeline" -> List(
      (("name" -> "$output") ~ ("inputs" -> List("tokenizer"))),
      (("name" -> "tokenizer") ~ ("inputs" -> List("content", "lang", "contentType"))),
      (("name" -> "content") ~ ("inputs" -> List("$document"))),
      (("name" -> "lang") ~ ("inputs" -> List("$document", "contentType"))),
      (("name" -> "contentType") ~ ("inputs" -> List("$document"))))))

  val extractorConfig = (
    ("transforms" -> List(
      (("name" -> "contentType") ~
        ("class" -> "com.idibon.ml.feature.contenttype.ContentTypeDetector")),
      (("name" -> "lang") ~
        ("class" -> "com.idibon.ml.feature.language.LanguageDetector")),
      (("name" -> "words") ~
        ("class" -> "com.idibon.ml.feature.bagofwords.ChainBagOfWords") ~
        ("config" -> ("transform" -> "ToLower"))),
      (("name" -> "lift") ~
        ("class" -> "com.idibon.ml.feature.ChainLiftTransformer")),
      (("name" -> "index") ~
        ("class" -> "com.idibon.ml.feature.indexer.ChainIndexTransformer")))) ~
    ("pipeline" -> List(
      (("name" -> "contentType") ~ ("inputs" -> List("$document"))),
      (("name" -> "lang") ~ ("inputs" -> List("$document", "contentType"))),
      (("name" -> "words") ~ ("inputs" -> List("$sequence", "lang"))),
      (("name" -> "lift") ~ ("inputs" -> List("words"))),
      (("name" -> "index") ~ ("inputs" -> List("lift"))),
      (("name" -> "$output") ~ ("inputs" -> List("index"))))))

  describe("initialization") {

    it("should load furnaces from configuration data") {
      val config = (("sequenceGenerator" -> sequenceConfig) ~
        ("featureExtractor" -> extractorConfig))

      ChainNERFurnace(new EmbeddedEngine, "spec", config) shouldBe a [Furnace2[_]]
    }

    it("should use custom seed values") {
      val eval = new scala.util.Random(17)
      val config = (("sequenceGenerator" -> sequenceConfig) ~
        ("featureExtractor" -> extractorConfig) ~
        ("seed" -> 17))
      val furnace = ChainNERFurnace(new EmbeddedEngine, "spec", config)
      furnace.prng.nextInt() shouldBe eval.nextInt()
    }
  }

  describe("train") {
    val documents = List(
      """{"content":"CRFs are complicated","annotations":[{"label":{"name":"abbreviation"},"isPositive":true,"offset":0,"length":4}]}""",
      """{"content":"Training better not be FUBAR","annotations":[{"label":{"name":"abbreviation"},"isPositive":true,"offset":23,"length":5}]}""")

    it("should train a predict model") {
      val config = (("sequenceGenerator" -> sequenceConfig) ~
        ("featureExtractor" -> extractorConfig))
      val furnace = ChainNERFurnace(new EmbeddedEngine, "spec", config)

      val training = TrainOptions()
        .addDocuments(documents.map(d => parse(d).asInstanceOf[JObject]))
        .withMaxTrainTime(5.0)
        .build()

      val model = Await.result(furnace.train(training), training.maxTrainTime)
      model shouldBe a [crf.BIOModel]
      val prediction = model.predict(
        Document.document(parse(documents.head).asInstanceOf[JObject]),
        PredictOptions.DEFAULT)
      prediction should have length 1
      prediction.head shouldBe a [Span]
      prediction.head.label shouldBe "abbreviation"
      prediction.head.probability should be (0.65f +- 0.05f)
      prediction.head.asInstanceOf[Span].offset shouldBe 0
      prediction.head.asInstanceOf[Span].length shouldBe 4
    }
  }
}
