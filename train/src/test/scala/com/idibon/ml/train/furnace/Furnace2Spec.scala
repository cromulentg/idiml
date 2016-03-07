package com.idibon.ml.train.furnace

import com.idibon.ml.predict._
import com.idibon.ml.common.EmbeddedEngine

import org.scalatest.{Matchers, FunSpec}
import org.json4s.JsonDSL._
import org.json4s._

class Furnace2Spec extends FunSpec with Matchers {

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

  val furnaceConfig = (("sequenceGenerator" -> sequenceConfig) ~
    ("featureExtractor" -> extractorConfig))

  describe("registry") {
    it("should raise an exception for invalid result types") {
      intercept[NoSuchElementException] {
        Furnace2[NotARealResult](null, "ChainNERFurnace", "new furnace", null)
      }
      intercept[NoSuchElementException] {
        Furnace2[Classification](new EmbeddedEngine, "ChainNERFurnace",
          "new furnace", furnaceConfig)
      }
    }

    it("should return the correct furance") {
      val furnace = Furnace2[Span](new EmbeddedEngine, "ChainNERFurnace",
        "new furnace", furnaceConfig)
      furnace.name shouldBe "new furnace"
      furnace shouldBe a [ChainNERFurnace]
    }
  }
}

class NotARealResult extends PredictResult {
  def label = ""
  def probability = 0.0f
  def matchCount = 0
  def flags = 0
}
