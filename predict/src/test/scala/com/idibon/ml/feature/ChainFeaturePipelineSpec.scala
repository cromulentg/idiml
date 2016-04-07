package com.idibon.ml.feature

import org.scalatest.{FunSpec, Matchers}
import org.json4s.JsonDSL._
import org.apache.spark.mllib.linalg.Vectors

class ChainFeaturePipelineSpec extends FunSpec with Matchers {

  def buildSequenceGenerator = (SequenceGeneratorBuilder("foo")
    += ("contentType", new contenttype.ContentTypeDetector(false), Seq("$document"))
    += ("lang", new language.LanguageDetector, Seq("$document", "contentType"))
    += ("content", new ContentExtractor, Seq("$document"))
    += ("tokenizer", new tokenizer.ChainTokenTransformer(Seq(tokenizer.Tag.Word)),
      Seq("content", "lang", "contentType"))
    := ("tokenizer"))

  def buildChainPipeline = (ChainPipelineBuilder("foo")
    += ("contentType", new contenttype.ContentTypeDetector(false), Seq("$document"))
    += ("lang", new language.LanguageDetector, Seq("$document", "contentType"))
    += ("words", new bagofwords.ChainBagOfWords(bagofwords.CaseTransform.ToLower),
      Seq("$sequence", "lang"))
    += ("shapes", new wordshapes.ChainWordShapesTransformer, Seq("$sequence"))
    += ("lift", new ChainLiftTransformer(), Seq("shapes", "words"))
    += ("index", new indexer.ChainIndexTransformer(), Seq("lift"))
    := ("index"))



  it("should generate chains of vectors") {
    var sequencer = buildSequenceGenerator
    var extractor = buildChainPipeline

    /* 8 total features: Word(hello), Shape(cc), Word(This), Shape(Ccc),
     * Word(is), Word(a), Shape(c), Word(sequence) */
    val document = (("content" -> "hello, This is a sequence") ~ ("name" -> "foo"))
    extractor(document, sequencer(document))
    extractor = extractor.freeze
    sequencer = sequencer.freeze
    val features = extractor(document, sequencer(document))

    features shouldBe Chain(Vectors.dense(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      Vectors.dense(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
      Vectors.dense(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
      Vectors.dense(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
      Vectors.dense(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
  }
}
