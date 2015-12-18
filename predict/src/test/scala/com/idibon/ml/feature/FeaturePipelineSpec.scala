package com.idibon.ml.feature
import com.idibon.ml.feature.tokenizer.{Token,TokenTransformer}

import scala.util.Random

import org.scalatest.{Matchers, FunSpec}
import org.json4s.{JObject,JString}

class FeaturePipelineSpec extends FunSpec with Matchers {

  describe("isValidBinding") {
    val consumer = Class
      .forName("com.idibon.ml.feature.TokenConsumer")
      .newInstance.asInstanceOf[FeatureTransformer]
    val tokenizer = Class
      .forName("com.idibon.ml.feature.tokenizer.TokenTransformer")
      .newInstance.asInstanceOf[FeatureTransformer]
    val generator = Class
      .forName("com.idibon.ml.feature.DocumentExtractor")
      .newInstance.asInstanceOf[FeatureTransformer]

    it("should return true on valid bindings") {
      FeaturePipeline.isValidBinding(consumer, List(tokenizer)) shouldBe true
      FeaturePipeline.isValidBinding(tokenizer, List(generator)) shouldBe true
      /* this is interpreted / indistinguishable from a variadic parameter of
       * type Feature[JObject]*, so the empty list is 'ok'. */
      FeaturePipeline.isValidBinding(tokenizer, List()) shouldBe true
    }

    it("should return false on invalid bindings") {
      FeaturePipeline.isValidBinding(tokenizer, List(consumer)) shouldBe false
      FeaturePipeline.isValidBinding(tokenizer, List(generator, generator)) shouldBe false
      FeaturePipeline.isValidBinding(consumer, List(tokenizer, generator)) shouldBe false
    }
  }

  describe("buildDependencyChain") {
    it("should handle an empty list correctly") {
      val chain = FeaturePipeline.sortDependencies(List[PipelineEntry]())
      chain shouldBe empty
    }

    it("should sort simple pipelines in dependency order") {
      val chain = FeaturePipeline.sortDependencies(
        Random.shuffle(List(
          new PipelineEntry("tokenizer", List("$document")),
          new PipelineEntry("ngrammer", List("tokenizer")),
          new PipelineEntry("$output", List("tokenizer", "ngrammer"))))
      )
      chain shouldBe List(List("tokenizer"), List("ngrammer"), List("$output"))
    }

    it("should support diamond graphs") {
      val chain = FeaturePipeline.sortDependencies(
        Random.shuffle(List(
          new PipelineEntry("tokenizer", List("$document")),
          new PipelineEntry("metadata", List("$document")),
          new PipelineEntry("strip-punctuation", List("tokenizer")),
          new PipelineEntry("n-grams", List("strip-punctuation")),
          new PipelineEntry("word-vectors", List("n-grams")),
          new PipelineEntry("$output", List("word-vectors", "metadata"))))
      )
      chain shouldBe List(
        List("tokenizer", "metadata"),
        List("strip-punctuation"),
        List("n-grams"),
        List("word-vectors"),
        List("$output")
      )
    }

    it("should raise an exception on cyclic graphs") {
      intercept[IllegalArgumentException] {
        FeaturePipeline.sortDependencies(
          Random.shuffle(List(
            new PipelineEntry("tokenizer", List("$document")),
            new PipelineEntry("n-grams", List("tokenizer", "word-vectors")),
            new PipelineEntry("word-vectors", List("n-grams")),
            new PipelineEntry("$output", List("word-vectors"))))
        )
      }

      intercept[IllegalArgumentException] {
        FeaturePipeline.sortDependencies(
          Random.shuffle(List(
            new PipelineEntry("tokenizer", List("$document")),
            new PipelineEntry("n-grams", List("tokenizer", "n-grams")),
            new PipelineEntry("$output", List("n-grams"))))
        )
      }
    }

    it("should raise an exception on non-existent transforms") {
      intercept[IllegalArgumentException] {
        FeaturePipeline.sortDependencies(
          Random.shuffle(List(
            new PipelineEntry("n-grams", List("tokenizer")),
            new PipelineEntry("$output", List("n-grams"))))
        )
      }
    }
  }
}

private [this] class TokenConsumer extends FeatureTransformer {
  def apply(tokens: Seq[Feature[Token]]) = tokens
}

private [this] class DocumentExtractor extends FeatureTransformer {
  def apply(document: JObject): Seq[StringFeature] = {
    List(StringFeature((document \ "content").asInstanceOf[JString].s))
  }
}
