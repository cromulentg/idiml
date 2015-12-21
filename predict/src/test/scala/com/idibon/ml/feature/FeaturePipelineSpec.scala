package com.idibon.ml.feature
import com.idibon.ml.feature.tokenizer.{Token,TokenTransformer}

import scala.util.Random
import scala.collection.mutable.{HashMap => MutableMap}

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.scalatest.{Matchers, FunSpec}
import org.json4s.{JObject, JString, JDouble, JField}

class FeaturePipelineSpec extends FunSpec with Matchers {

  describe("bindGraph") {

    it("should raise an exception if $output is incomplete") {
      val transforms = Map[String, FeatureTransformer]()
      val pipeline = List(new PipelineEntry("$output", List()))

      intercept[IllegalArgumentException] {
        FeaturePipeline.bindGraph(transforms, pipeline)
      }
    }

    it("should raise an exception if a transformer doesn't implement apply")(pending)

    it("should raise an exception if $output is undefined")(pending)

    it("should raise an exception if a pipeline stage is missing")(pending)

    it("should raise an exception if a reserved name is used")(pending)

    it("should raise an exception if the transformer uses currying") {
      val transforms = Map(
        "curriedExtractor" -> new CurriedExtractor,
        "contentExtractor" -> new DocumentExtractor
      )
      val pipeline = List(
        new PipelineEntry("$output", List("curriedExtractor")),
        new PipelineEntry("contentExtractor", List("$document")),
        new PipelineEntry("curriedExtractor", List("$document", "contentExtractor"))
      )
      intercept[UnsupportedOperationException] {
        FeaturePipeline.bindGraph(transforms, pipeline)
      }
    }

    it("should treat $document as a typeOf[JObject]") {
      val transforms = Map(
        "contentExtractor" -> new DocumentExtractor,
        "metadataVector" -> new MetadataNumberExtractor,
        "featureVector" -> new FeatureVectors
      )
      val pipeline = List(
        new PipelineEntry("$output", List("featureVector", "metadataVector")),
        new PipelineEntry("contentExtractor", List("$document")),
        new PipelineEntry("metadataVector", List("$document")),
        new PipelineEntry("featureVector", List("contentExtractor"))
      )

      val graph = FeaturePipeline.bindGraph(transforms, pipeline)
      graph.size shouldBe 3

      val intermediates = MutableMap[String, Any]()
      intermediates.put("$document", JObject(List(
        JField("content", JString("A document!")),
        JField("metadata", JObject(List(
          JField("number", JDouble(3.14159265)))))
      )))

      for (stage <- graph; transform <- stage.transforms) {
        intermediates.put(transform.name, transform.transform(intermediates))
      }

      intermediates.get("$output") shouldBe
        Some(List(Vectors.dense(11.0), Vectors.dense(3.14159265)))
    }

    it("should support variadic arguments in pipelines") {
      val transforms = Map(
        "contentExtractor" -> new DocumentExtractor,
        "concatenator" -> new VectorConcatenator,
        "metadataVector" -> new MetadataNumberExtractor,
        "featureVector" -> new FeatureVectors
      )

      val pipelines = List(
        (Vectors.dense(3.14159265, 11.0), List(
          new PipelineEntry("$output", List("concatenator")),
          new PipelineEntry("concatenator", List("metadataVector", "featureVector")),
          new PipelineEntry("contentExtractor", List("$document")),
          new PipelineEntry("metadataVector", List("$document")),
          new PipelineEntry("featureVector", List("contentExtractor")))),
        (Vectors.dense(11.0, 3.14159265), List(
          // exactly the same, but the order of concatenation is reversed
          new PipelineEntry("$output", List("concatenator")),
          new PipelineEntry("concatenator", List("featureVector", "metadataVector")),
          new PipelineEntry("contentExtractor", List("$document")),
          new PipelineEntry("metadataVector", List("$document")),
          new PipelineEntry("featureVector", List("contentExtractor"))))
      )

      pipelines.foreach({ case (expected, pipeline) => {
        val graph = FeaturePipeline.bindGraph(transforms, pipeline)
        graph.size shouldBe 4

        val intermediates = MutableMap[String, Any]()
        intermediates.put("$document", JObject(List(
          JField("content", JString("A document!")),
          JField("metadata", JObject(List(
            JField("number", JDouble(3.14159265)))))
        )))

        for (stage <- graph; transform <- stage.transforms) {
          intermediates.put(transform.name, transform.transform(intermediates))
        }

        intermediates.get("$output") shouldBe Some(List(expected))
      }})
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

private [this] class VectorConcatenator extends FeatureTransformer {
  def apply(inputs: Vector*): Vector = {
    Vectors.dense(inputs.foldLeft(Array[Double]())(_ ++ _.toArray))
  }
}

private [this] class CurriedExtractor extends FeatureTransformer {
  def apply(document: JObject)(tokens: Seq[Feature[String]]): Vector = {
    Vectors.dense(1.0)
  }
}

private [this] class FeatureVectors extends FeatureTransformer {
  def apply(content: Seq[Feature[String]]): Vector = {
    Vectors.dense(content.map(_.get.length.toDouble).toArray)
  }
}

private [this] class MetadataNumberExtractor extends FeatureTransformer {
  def apply(document: JObject): Vector = {
    Vectors.dense((document \ "metadata" \ "number").asInstanceOf[JDouble].num)
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
