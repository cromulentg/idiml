package com.idibon.ml.train.datagenerator.crf

import scala.language.implicitConversions

import com.idibon.ml.feature._
import com.idibon.ml.predict.crf._
import com.idibon.ml.train.datagenerator.json._

import org.apache.spark.mllib.linalg.Vectors
import org.json4s.{Extraction, JObject}

import org.scalatest.{Matchers, FunSpec, BeforeAndAfter}

class BIOTaggerSpec extends FunSpec with Matchers with BIOTagger with BeforeAndAfter {

  var sequenceGenerator: SequenceGenerator = null
  var featureExtractor: ChainPipeline = null

  def buildSequenceGenerator = (SequenceGeneratorBuilder("foo")
    += ("contentType", new contenttype.ContentTypeDetector, Seq("$document"))
    += ("lang", new language.LanguageDetector, Seq("$document", "contentType"))
    += ("content", new ContentExtractor, Seq("$document"))
    += ("tokenizer", new tokenizer.ChainTokenTransformer(Seq(tokenizer.Tag.Word)),
      Seq("content", "lang", "contentType"))
    := ("tokenizer"))

  def buildChainPipeline = (ChainPipelineBuilder("foo")
    += ("contentType", new contenttype.ContentTypeDetector, Seq("$document"))
    += ("lang", new language.LanguageDetector, Seq("$document", "contentType"))
    += ("words", new bagofwords.ChainBagOfWords(bagofwords.CaseTransform.ToLower),
      Seq("$sequence", "lang"))
    += ("lift", new ChainLiftTransformer(), Seq("words"))
    += ("index", new indexer.ChainIndexTransformer(), Seq("lift"))
    := ("index"))

  before {
    sequenceGenerator = buildSequenceGenerator
    featureExtractor = buildChainPipeline
  }

  implicit def toJson(doc: Document): JObject = {
    implicit val formats = org.json4s.DefaultFormats
    Extraction.decompose(doc).asInstanceOf[JObject]
  }

  def prime(documents: JObject*) {
    documents.foreach(doc => featureExtractor(doc, sequenceGenerator(doc)))
    sequenceGenerator = sequenceGenerator.freeze
    featureExtractor = featureExtractor.freeze
  }

  it("should raise an exception if a non-span annotation exists") {
    val doc = Document("a document", List(Annotation("foo")))
    intercept[IllegalArgumentException] { tag(doc) }
  }

  it("should treat a token that wholly contains an annotation as BEGIN") {
    val doc = Document("a document", List(Annotation("foo", true, 4, 3)))
    prime(doc)
    tag(doc) shouldBe Seq(BIOTag("O") -> Vectors.dense(1.0, 0.0),
      BIOTag("Bfoo") -> Vectors.dense(0.0, 1.0))
  }

  it("should treat subsequent tokens partially within the annotation as INSIDE") {
    val doc = Document("z document", List(Annotation("foo", true, 0, 3)))
    prime(doc)
    tag(doc) shouldBe Seq(BIOTag("Bfoo") -> Vectors.dense(1.0, 0.0),
      BIOTag("Ifoo") -> Vectors.dense(0.0, 1.0))
  }

  it("should detect the first token within the annotation as BEGIN") {
    val doc = Document("a document", List(Annotation("foo", true, 1, 9)))
    prime(doc)
    tag(doc) shouldBe Seq(BIOTag("O") -> Vectors.dense(1.0, 0.0),
      BIOTag("Bfoo") -> Vectors.dense(0.0, 1.0))
  }

  it("should ignore negative and zero-length annotations") {
    val doc = Document("a document", List(Annotation("foo", false, 0, 1),
      Annotation("foo", true, 0, 1), Annotation("foo", true, 4, -4),
      Annotation("foo", true, 3, 0)))
    prime(doc)
    tag(doc) shouldBe Seq(BIOTag("Bfoo") -> Vectors.dense(1.0, 0.0),
      BIOTag("O") -> Vectors.dense(0.0, 1.0))
  }

  it("should raise an exception if annotations overlap") {
    val doc = Document("a document", List(Annotation("foo", true, 0, 1),
      Annotation("foo", true, 0, 4)))
    prime(doc)
    intercept[AssertionError] { tag(doc) }
  }

  it("should skip over annotations swallowed by long tokens") {
    val doc = Document("a document a", List(Annotation("foo", true, 2, 3),
      Annotation("foo", true, 5, 1), Annotation("foo", true, 6, 1),
      Annotation("bar", true, 11, 10)))
    prime(doc)
    tag(doc) shouldBe Seq(BIOTag("O") -> Vectors.dense(1.0, 0.0),
      BIOTag("Bfoo") -> Vectors.dense(0.0, 1.0),
      BIOTag("Bbar") -> Vectors.dense(1.0, 0.0))
  }

  it("should support multiple spans") {
    val doc = Document("the quick brown fox jumped over the lazy dog",
      List(Annotation("np", true, 4, 15), Annotation("v", true, 20, 6),
        Annotation("np", true, 36, 8)))
    prime(doc)
    tag(doc) shouldBe(Seq(
      BIOTag("O") -> Vectors.dense(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      BIOTag("Bnp") -> Vectors.dense(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      BIOTag("Inp") -> Vectors.dense(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      BIOTag("Inp") -> Vectors.dense(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
      BIOTag("Bv") -> Vectors.dense(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
      BIOTag("O") -> Vectors.dense(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
      BIOTag("O") -> Vectors.dense(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      BIOTag("Bnp") -> Vectors.dense(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
      BIOTag("Inp") -> Vectors.dense(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)))
  }
}
