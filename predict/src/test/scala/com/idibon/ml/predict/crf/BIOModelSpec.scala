package com.idibon.ml.predict.crf

import com.idibon.ml.feature.tokenizer.{Tag, Token}

import scala.util.Random

import com.idibon.ml.feature._
import com.idibon.ml.predict._
import com.idibon.ml.common._
import com.idibon.ml.alloy._

import org.apache.spark.mllib.linalg.Vectors
import org.json4s.JsonDSL._

import org.scalatest.{Matchers, FunSpec}

class BIOModelSpec extends FunSpec with Matchers {

  it("should save and load correctly") {
    def $(label: String, v0: Double, vec: Double*) = (label -> Vectors.dense(v0, vec: _*))
    val documents = Seq(
      Seq($("O", 0.0, 0.0, 0.0, 0.0, 1.0),
        $("B0", 0.0, 0.0, 0.0, 1.0, 0.0),
        $("I0", 0.0, 0.0, 1.0, 0.0, 0.0),
        $("I0", 0.0, 0.0, 0.0, 0.0, 1.0),
        $("I0", 0.0, 0.0, 0.0, 0.0, 1.0)),
      Seq($("O", 0.0, 0.0, 0.0, 0.0, 1.0),
        $("O", 0.0, 0.0, 1.0, 0.0, 0.0),
        $("B0", 0.0, 0.0, 0.0, 1.0, 0.0),
        $("I0", 0.0, 1.0, 1.0, 0.0, 0.0),
        $("O", 1.0, 0.0, 0.0, 0.0, 0.0)))
    val model = new FactorieCRF(5) with TrainableFactorieModel
    model.train(documents map model.observe, new Random)

    val sequencer = (SequenceGeneratorBuilder("foo")
      += ("contentType", new contenttype.ContentTypeDetector, Seq("$document"))
      += ("lang", new language.LanguageDetector, Seq("$document", "contentType"))
      += ("content", new ContentExtractor, Seq("$document"))
      += ("tokenizer", new tokenizer.ChainTokenTransformer(Seq(tokenizer.Tag.Word)),
        Seq("content", "lang", "contentType"))
      := ("tokenizer"))

    val extractor = (ChainPipelineBuilder("foo")
      += ("contentType", new contenttype.ContentTypeDetector, Seq("$document"))
      += ("lang", new language.LanguageDetector, Seq("$document", "contentType"))
      += ("words", new bagofwords.ChainBagOfWords(bagofwords.CaseTransform.ToLower),
        Seq("$sequence", "lang"))
      += ("lift", new ChainLiftTransformer(), Seq("words"))
      += ("index", new indexer.ChainIndexTransformer(), Seq("lift"))
      := ("index"))

    val doc = ("content" -> "there are 5 features here")
    extractor(doc, sequencer(doc))
    val bio = new BIOModel(model, sequencer.freeze(), extractor.freeze())

    val archive = collection.mutable.HashMap[String, Array[Byte]]()
    val json = bio.save(new MemoryAlloyWriter(archive))

    val loaded = (new BIOModelLoader).load(new EmbeddedEngine,
      Some(new MemoryAlloyReader(archive.toMap)), json)

    val confidence = model.predict(Seq(Vectors.dense(0.0, 0.0, 0.0, 1.0, 0.0))).head._2

    loaded.predict(Document.document("content" -> "features"),
      (new PredictOptionsBuilder().showTokens().showTokenTags().build())) shouldBe
      Seq(Span("0", confidence.toFloat, 0, 0, 8,
      Seq(Token("features",Tag.Word,0,8)), Seq(BIOType.BEGIN)))
  }
}
