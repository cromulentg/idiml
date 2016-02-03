package com.idibon.ml.predict.ml

import java.io._

import com.idibon.ml.alloy.{IntentAlloy, JarAlloy}
import com.idibon.ml.feature.indexer.IndexTransformer
import com.idibon.ml.feature.language.LanguageDetector
import com.idibon.ml.feature.tokenizer.TokenTransformer
import com.idibon.ml.feature.{ContentExtractor, FeaturePipeline, FeaturePipelineBuilder}
import com.idibon.ml.predict.ensemble.GangModel
import com.idibon.ml.predict.{PredictOptionsBuilder}
import org.apache.spark.mllib.classification.IdibonSparkMLLIBLRWrapper
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.json4s.JsonDSL._
import org.json4s._
import org.scalatest.{BeforeAndAfter, FunSpec, Matchers, ParallelTestExecution}

/**
  * Class to test IdibonMultiClassLRModel & related.
  */
class IdibonMultiClassLRModelSpec extends FunSpec
with Matchers with BeforeAndAfter with ParallelTestExecution {

  val pipeline: FeaturePipeline = (FeaturePipelineBuilder.named("StefansPipeline")
    += (FeaturePipelineBuilder.entry("convertToIndex", new IndexTransformer, "convertToTokens"))
    += (FeaturePipelineBuilder.entry("convertToTokens", new TokenTransformer, "contentExtractor", "language"))
    += (FeaturePipelineBuilder.entry("language", new LanguageDetector, "$document"))
    += (FeaturePipelineBuilder.entry("contentExtractor", new ContentExtractor, "$document"))
    := ("convertToIndex"))

  val text: String = "Everybody loves replacing hadoop with spark because it's much faster. a b d"
  val doc: JObject = ( "content" -> text )
  val fp = pipeline.prime(List(doc))

  var tempFilename = ""
  before {
    tempFilename = ""
  }

  after {
    try {
      new File(tempFilename).delete()
    } catch {
      case ioe: IOException => None
    }
  }

  describe("Save & load integration test") {
    it("saves, loads and predicts as expected") {
      val intercept = -1.123
      val coefficients = fp(doc).toDense
      val label: String = "alabel"
      // do hacky thing where model coefficients are document coefficients -- just to make
      // sure it all works
      val model = new IdibonMultiClassLRModel(
        Map("alabel" -> 1, "!alabel" -> 0),
        new IdibonSparkMLLIBLRWrapper(coefficients, intercept, coefficients.size, 2), fp)
      val labelToModel = Map((GangModel.MULTI_CLASS_LABEL -> model))
      val alloy = new JarAlloy(labelToModel, Map[String, String]())
      tempFilename = "save123.jar"
      alloy.save(tempFilename)
      // let's make sure to delete the file on exit
      val jarFile = new File(tempFilename)
      jarFile.exists() shouldBe true
      // get alloy back & predict on it.
      val resurrectedAlloy = JarAlloy.load(null, tempFilename)
      val options = new PredictOptionsBuilder().build()
      val result1 = alloy.predict(doc, options).get(label).getTopResult()
      val result2 = resurrectedAlloy.predict(doc, options).get(label).getTopResult()
      result2.matchCount shouldBe result1.matchCount
      result2.probability shouldBe result1.probability
    }
  }

  describe("Loads as intended") {
    it("Throws exception on unhandled version") {
      intercept[IOException] {
        new IdibonMultiClassLRModelLoader().load(null,
          null,  Some(JObject(List[JField](
            ("version", JString("0.0.0")), ("label", JString("alabel"))))))
      }
    }
  }

  describe("Saves as intended") {
    it("returns config as expected") {
      val alloy = new IntentAlloy()
      val intercept = -1.123
      val coefficients = fp(doc).toDense
      val label: String = "alabel"
      val model = new IdibonMultiClassLRModel(
        Map("alabel" -> 1, "!alabel" -> 0),
        new IdibonSparkMLLIBLRWrapper(coefficients, intercept, coefficients.size, 2), fp)
      val config = model.save(alloy.writer())
      implicit val formats = DefaultFormats
      val version = (config.get \ "version" ).extract[String]
      version shouldEqual "0.0.1"
      val featureMeta = (config.get \ "feature-meta").extract[JObject]
      featureMeta should not be JNothing
      featureMeta should not be JNull
    }
  }

  describe("Tests predict") {
    it("returns appropriate probability") {
      val intercept = -1.123
      val coefficients = Vectors.sparse(26, Array(0, 1, 2), Array(0.54, 0.83, 0.2)).toDense
      val model = new IdibonMultiClassLRModel(
        Map("alabel" -> 1, "!alabel" -> 0),
        new IdibonSparkMLLIBLRWrapper(coefficients, intercept, coefficients.size, 2), fp)
      val result = model.predict(doc, new PredictOptionsBuilder().build())
      result.getTopResult().getLabel() shouldBe "alabel"
      result.getTopResult().probability shouldBe 0.60992575f
      result.getTopResult().matchCount shouldBe 1
      val secondResult = result.getAllResults().get(1)
      secondResult.getLabel() shouldBe "!alabel"
      secondResult.probability shouldBe 0.39007428f
      secondResult.matchCount shouldBe 1
    }
    it("should return significant features") {
      val intercept = -0.123
      val coefficients = Vectors.sparse(26, Array(0, 1, 2), Array(0.1, -1.23, 0.2)).toDense
      val model = new IdibonMultiClassLRModel(
        Map("alabel" -> 1, "blabel" -> 0),
        new IdibonSparkMLLIBLRWrapper(coefficients, intercept, coefficients.size, 2), fp)
      val result = model.predict(doc, new PredictOptionsBuilder()
        .showSignificantFeatures(0.75f).build()).getTopResult()
      result.getLabel() shouldBe "blabel"
      result.probability shouldBe 0.7413506f
      result.matchCount shouldBe 1
      result.significantFeatures shouldBe List(("token- ", 0.7946197f))
    }
  }

  describe("Test writeCodecLibSVM") {
    it("test sparse vector"){
      val out: ByteArrayOutputStream = new ByteArrayOutputStream()
      val intercept = -1.123
      val coefficients = Vectors.sparse(
        10, Array[Int](1, 5, 9), Array[Double](0.234, 1.23, 234234.234)).toDense
      IdibonMultiClassLRModel.writeCodecLibSVM(Map("alabel" -> 1, "!alabel" -> 0),
        new WrappedByteArrayOutputStream(out), intercept, coefficients, coefficients.size)
      out.toByteArray() shouldEqual Array[Byte](2, 6, 97, 108, 97, 98, 101, 108, 1, 7, 33, 97, 108,
        97, 98, 101, 108, 0, 10, -65, -15, -9, -50, -39, 22, -121, 43, 10, 10, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 63, -51, -13, -74, 69, -95, -54, -63, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0,
        0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 5, 63, -13, -82, 20, 122, -31, 71, -82, 6, 0, 0, 0, 0,
        0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 9, 65, 12, -105, -47,
        -33, 59, 100, 90)
      out.close()
    }

    it("test empty sparse vector"){
      val out: ByteArrayOutputStream = new ByteArrayOutputStream()
      val intercept = 0.50
      val coefficients = Vectors.sparse(0, Array[Int](), Array[Double]())
      IdibonMultiClassLRModel.writeCodecLibSVM(Map("alabel" -> 1, "!alabel" -> 0),
        new WrappedByteArrayOutputStream(out), intercept, coefficients, coefficients.size)
      out.toByteArray() shouldEqual Array[Byte](2, 6, 97, 108, 97, 98, 101, 108, 1, 7, 33, 97, 108,
        97, 98, 101, 108, 0, 0, 63, -32, 0, 0, 0, 0, 0, 0, 0, 0)
      out.close()
    }
  }

  describe("Test readCodecLibSVM") {
    it("works on sparse vector") {
      val input = new ByteArrayInputStream(Array[Byte](2, 6, 97, 108, 97, 98, 101, 108, 1, 7, 33, 97, 108,
        97, 98, 101, 108, 0, 10, -65, -15, -9, -50, -39, 22, -121, 43, 10, 10, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 63, -51, -13, -74, 69, -95, -54, -63, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0,
        0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 5, 63, -13, -82, 20, 122, -31, 71, -82, 6, 0, 0, 0, 0,
        0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 9, 65, 12, -105, -47,
        -33, 59, 100, 90))
      val (intercept: Double,
      coefficients: Vector,
      labelToInt: Map[String, Int],
      numFeatures: Int) = IdibonMultiClassLRModel.readCodecLibSVM(new WrappedByteArrayInputStream(input))
      intercept shouldEqual -1.123
      coefficients.toSparse shouldEqual  Vectors.sparse(
        10, Array[Int](1, 5, 9), Array[Double](0.234, 1.23, 234234.234))
      labelToInt shouldEqual Map("alabel" -> 1, "!alabel" -> 0)
      numFeatures shouldEqual 10
    }

    it("works on empty sparse vector") {
      val input = new ByteArrayInputStream(Array[Byte](2, 6, 97, 108, 97, 98, 101, 108, 1, 7, 33, 97, 108,
        97, 98, 101, 108, 0, 0, 63, -32, 0, 0, 0, 0, 0, 0, 0, 0))
      val (intercept: Double,
      coefficients: Vector,
      labelToInt: Map[String, Int],
      numFeatures: Int) = IdibonMultiClassLRModel.readCodecLibSVM(new WrappedByteArrayInputStream(input))
      intercept shouldEqual 0.50
      coefficients.toSparse shouldEqual Vectors.sparse(0, Array[Int](), Array[Double]())
      labelToInt shouldEqual Map("alabel" -> 1, "!alabel" -> 0)
      numFeatures shouldEqual 0
    }
  }
}



