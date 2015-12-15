package com.idibon.ml.feature
import org.scalatest.{Matchers, FunSpec}
import scala.util.Random

class FeaturePipelineSpec extends FunSpec with Matchers {

  describe("buildDependencyChain") {
    val pipeline = new FeaturePipeline

    it("should handle an empty list correctly") {
      val chain = pipeline.sortDependencies(List[PipelineEntry]())
      chain shouldBe empty
    }

    it("should sort simple pipelines in dependency order") {
      val chain = pipeline.sortDependencies(
        Random.shuffle(List(
          new PipelineEntry("tokenizer", List("$document")),
          new PipelineEntry("ngrammer", List("tokenizer")),
          new PipelineEntry("$output", List("tokenizer", "ngrammer"))))
      )
      chain shouldBe List(List("tokenizer"), List("ngrammer"), List("$output"))
    }

    it("should support diamond graphs") {
      val chain = pipeline.sortDependencies(
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
        pipeline.sortDependencies(
          Random.shuffle(List(
            new PipelineEntry("tokenizer", List("$document")),
            new PipelineEntry("n-grams", List("tokenizer", "word-vectors")),
            new PipelineEntry("word-vectors", List("n-grams")),
            new PipelineEntry("$output", List("word-vectors"))))
        )
      }

      intercept[IllegalArgumentException] {
        pipeline.sortDependencies(
          Random.shuffle(List(
            new PipelineEntry("tokenizer", List("$document")),
            new PipelineEntry("n-grams", List("tokenizer", "n-grams")),
            new PipelineEntry("$output", List("n-grams"))))
        )
      }
    }

    it("should raise an exception on non-existent transforms") {
      intercept[IllegalArgumentException] {
        pipeline.sortDependencies(
          Random.shuffle(List(
            new PipelineEntry("n-grams", List("tokenizer")),
            new PipelineEntry("$output", List("n-grams"))))
        )
      }
    }
  }
}
