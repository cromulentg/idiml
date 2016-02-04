package com.idibon.ml.train.datagenerator

import com.idibon.ml.common.EmbeddedEngine
import com.idibon.ml.feature.indexer.IndexTransformer
import com.idibon.ml.feature.language.LanguageDetector
import com.idibon.ml.feature.tokenizer.TokenTransformer
import com.idibon.ml.feature.{ContentExtractor, FeaturePipeline, FeaturePipelineBuilder}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.json4s.JObject
import org.json4s.native.JsonMethods.parse
import org.scalatest.{BeforeAndAfter, BeforeAndAfterAll, FunSpec, Matchers}

import scala.io.Source

/** Verifies the functionality of KClassDataFrameGenerator
  *
  */
class KClassDataFrameGeneratorSpec extends FunSpec with Matchers
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
      += (FeaturePipelineBuilder.entry("convertToIndex", new IndexTransformer(0), "convertToTokens"))
      += (FeaturePipelineBuilder.entry("convertToTokens", new TokenTransformer, "contentExtractor", "languageDetector"))
      += (FeaturePipelineBuilder.entry("languageDetector", new LanguageDetector, "$document"))
      += (FeaturePipelineBuilder.entry("contentExtractor", new ContentExtractor, "$document"))
      := ("convertToIndex"))
  }

  describe("KClassDataFrameGenerator") {
    it("should generate LabeledPoint RDD's correctly") {
      implicit val formats = org.json4s.DefaultFormats

      val inFilePath = getClass.getClassLoader.getResource(inFile).getPath()
      val primedPipeline = pipeline.prime(
        Source.fromFile(inFilePath).getLines.map(line => parse(line).extract[JObject]))
      val gen = new KClassDataFrameGenerator()
      val baseTraining = gen.createPerLabelLPs(primedPipeline, () => {
        Source.fromFile(inFilePath)
          .getLines.map(line => parse(line).extract[JObject])
      })
      val labeledPoint = LabeledPoint(1.0,
        Vectors.sparse(
          19,
          Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18),
          Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)))
      // very base training data
      val expected = Map("Intent to Buy" -> List(labeledPoint))
      baseTraining shouldBe expected
      // very conversion into RDD
      val training = gen.createPerLabelRDDs(engine, baseTraining)

      training.size shouldBe 1

      val rdd = training("Intent to Buy")
      rdd shouldBe an[RDD[_]]

      rdd.collect().head shouldBe labeledPoint
    }

    it("should generate data frames correctly") {
      implicit val formats = org.json4s.DefaultFormats
      val labeledPoint = LabeledPoint(1.0,
        Vectors.sparse(
          19,
          Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18),
          Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)))

      val inFilePath = getClass.getClassLoader.getResource(inFile).getPath()
      val primedPipeline = pipeline.prime(
        Source.fromFile(inFilePath).getLines.map(line => parse(line).extract[JObject]))

      val dataFrameMap = new KClassDataFrameGenerator().getLabeledPointData(engine, primedPipeline, () => {
        Source.fromFile(inFilePath)
          .getLines.map(line => parse(line).extract[JObject])
      })
      val sqlContext = new org.apache.spark.sql.SQLContext(engine.sparkContext)
      dataFrameMap.get.foreach({ case (label, dataframe) => {
        label shouldBe "Intent to Buy"
        val results = dataframe.collect()
        results === Array(labeledPoint)
      }})
    }
  }
}
