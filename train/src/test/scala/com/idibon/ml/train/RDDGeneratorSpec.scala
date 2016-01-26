package com.idibon.ml.train

import scala.io.Source
import com.idibon.ml.feature.{FeaturePipeline, ContentExtractor, FeaturePipelineBuilder}
import com.idibon.ml.feature.language.LanguageDetector
import com.idibon.ml.feature.indexer.IndexTransformer
import com.idibon.ml.feature.tokenizer.TokenTransformer
import com.idibon.ml.common.EmbeddedEngine
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.scalatest.{BeforeAndAfterAll, BeforeAndAfter, Matchers, FunSpec}
import org.json4s.JObject
import org.json4s.native.JsonMethods.parse

/** Verifies the functionality of RDDGenerator
  *
  */
class RDDGeneratorSpec extends FunSpec with Matchers
  with BeforeAndAfter with BeforeAndAfterAll {

  override def beforeAll = {
    super.beforeAll
  }

  override def afterAll = {
    super.afterAll
  }

  val inFile : String = "test_data/labeled_points.json"
  var pipeline : FeaturePipeline = _
  val engine = new EmbeddedEngine

  /** Sets up the test object, spark context, & feature pipeline */
  before {
    // Define a pipeline that generates feature vectors
    pipeline = (FeaturePipelineBuilder.named("IntentPipeline")
      += (FeaturePipelineBuilder.entry("convertToIndex", new IndexTransformer, "convertToTokens"))
      += (FeaturePipelineBuilder.entry("convertToTokens", new TokenTransformer, "contentExtractor", "languageDetector"))
      += (FeaturePipelineBuilder.entry("languageDetector", new LanguageDetector, "$document"))
      += (FeaturePipelineBuilder.entry("contentExtractor", new ContentExtractor, "$document"))
      := ("convertToIndex"))
  }

  describe("RDDGenerator") {
    it("should generate LabeledPoint RDD's correctly") {
      implicit val formats = org.json4s.DefaultFormats

      val inFilePath = getClass.getClassLoader.getResource(inFile).getPath()
      val (training, fp) = RDDGenerator.getLabeledPointRDDs(engine, pipeline, () => {
        Source.fromFile(inFilePath)
          .getLines.map(line => parse(line).extract[JObject])
      })

      training.size shouldBe 1

      val rdd = training("Intent to Buy")
      rdd shouldBe an[RDD[_]]

      val labeled_point_result = LabeledPoint(1.0, Vectors.sparse(19, Array(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18), Array(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0)))
      rdd.collect().head shouldBe labeled_point_result
    }
  }
}
