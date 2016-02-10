package com.idibon.ml.train.furnace

import com.idibon.ml.common.EmbeddedEngine
import com.idibon.ml.feature.indexer.IndexTransformer
import com.idibon.ml.feature.language.LanguageDetector
import com.idibon.ml.feature.tokenizer.TokenTransformer
import com.idibon.ml.feature.contenttype.ContentTypeDetector
import com.idibon.ml.feature.{ContentExtractor, FeaturePipelineBuilder, FeaturePipeline}
import com.idibon.ml.train.datagenerator.{KClassDataFrameGeneratorBuilder, KClassDataFrameGenerator}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.json4s._
import org.json4s.native.JsonMethods._
import org.scalatest._

import scala.io.Source

/**
  * Tests SimpleLogisticRegression
  */
class SimpleLogisticRegressionSpec extends FunSpec
  with Matchers with BeforeAndAfter with ParallelTestExecution with BeforeAndAfterAll {

  override def beforeAll = {
    super.beforeAll
  }

  override def afterAll = {
    super.afterAll
  }

  val inFile : String = "test_data/multiclass_labeled_points.json"
  var pipeline : FeaturePipeline = _
  val engine = new EmbeddedEngine

  /** Sets up the test object, spark context, & feature pipeline */
  before {
    // Define a pipeline that generates feature vectors
    pipeline = (FeaturePipelineBuilder.named("IntentPipeline")
      += (FeaturePipelineBuilder.entry("convertToIndex", new IndexTransformer(0), "convertToTokens"))
      += (FeaturePipelineBuilder.entry("convertToTokens", new TokenTransformer, "contentExtractor", "languageDetector", "contentDetector"))
      += (FeaturePipelineBuilder.entry("languageDetector", new LanguageDetector, "$document", "contentDetector"))
      += (FeaturePipelineBuilder.entry("contentExtractor", new ContentExtractor, "$document"))
      += (FeaturePipelineBuilder.entry("contentDetector", new ContentTypeDetector, "$document"))
      := ("convertToIndex"))
  }

  describe("featurize data tests") {
    it("produces correct labelled point data map") {
      implicit val formats = org.json4s.DefaultFormats
      val inFilePath = getClass.getClassLoader.getResource(inFile).getPath()
      val primedPipeline = pipeline.prime(
        Source.fromFile(inFilePath).getLines.map(line => parse(line).extract[JObject]))
      val simple = new SimpleLogisticRegressionFurnaceBuilder().build(engine)
      val features = simple.featurizeData(() => {
        Source.fromFile(inFilePath)
          .getLines.map(line => parse(line).extract[JObject])
      }, new KClassDataFrameGeneratorBuilder().build(), List(primedPipeline)).head
      features.get.size shouldBe 2
      features.get.keys.toList shouldBe List("Intent to Buy", "Monkey")
      val itbpoints = features.get("Intent to Buy").collect()
      itbpoints.size shouldBe 2
      itbpoints(0) === Array(LabeledPoint(1.0,
        Vectors.sparse(29,
          Array(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18),
          Array(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0))))
      itbpoints(1) === Array(LabeledPoint(0.0,
        Vectors.sparse(29,
          Array(19,20,21,22,23,24,25,26,27,28),
          Array(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0))))
      val mpoints = features.get("Monkey").collect()
      mpoints.size shouldBe 1
      mpoints(0) === Array(LabeledPoint(1.0,
        Vectors.sparse(29,
          Array(19,20,21,22,23,24,25,26,27,28),
          Array(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0))))
    }
  }

  describe("fit tests") {
    //TODO: Once we stabalize on parameter ingestion.
  }

  describe("get estimator") {
    //TODO: once we stabalize on parameter ingestion.
  }
}


/**
  * Tests XValLogisticRegression
  */
class XValLogisticRegressionSpec extends FunSpec
  with Matchers with BeforeAndAfter with ParallelTestExecution with BeforeAndAfterAll {

  override def beforeAll = {
    super.beforeAll
  }

  override def afterAll = {
    super.afterAll
  }

  val inFile : String = "test_data/multiclass_labeled_points.json"
  var pipeline : FeaturePipeline = _
  val engine = new EmbeddedEngine

  /** Sets up the test object, spark context, & feature pipeline */
  before {
    // Define a pipeline that generates feature vectors
    pipeline = (FeaturePipelineBuilder.named("IntentPipeline")
      += (FeaturePipelineBuilder.entry("convertToIndex", new IndexTransformer(0), "convertToTokens"))
      += (FeaturePipelineBuilder.entry("convertToTokens", new TokenTransformer, "contentExtractor", "languageDetector", "contentDetector"))
      += (FeaturePipelineBuilder.entry("languageDetector", new LanguageDetector, "$document", "contentDetector"))
      += (FeaturePipelineBuilder.entry("contentExtractor", new ContentExtractor, "$document"))
      += (FeaturePipelineBuilder.entry("contentDetector", new ContentTypeDetector, "$document"))
      := ("convertToIndex"))
  }

  describe("featurize data tests") {
    it("produces correct labelled point data map") {
      implicit val formats = org.json4s.DefaultFormats
      val inFilePath = getClass.getClassLoader.getResource(inFile).getPath()
      val primedPipeline = pipeline.prime(
        Source.fromFile(inFilePath).getLines.map(line => parse(line).extract[JObject]))
      val xval = new XValLogisticRegressionFurnaceBuilder().build(engine)
      val features = xval.featurizeData(() => {
        Source.fromFile(inFilePath)
          .getLines.map(line => parse(line).extract[JObject])
      }, new KClassDataFrameGeneratorBuilder().build(), List(primedPipeline)).head
      features.get.size shouldBe 2
      features.get.keys.toList shouldBe List("Intent to Buy", "Monkey")
      val itbpoints = features.get("Intent to Buy").collect()
      itbpoints.size shouldBe 2
      itbpoints(0) === Array(LabeledPoint(1.0,
        Vectors.sparse(29,
          Array(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18),
          Array(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0))))
      itbpoints(1) === Array(LabeledPoint(0.0,
        Vectors.sparse(29,
          Array(19,20,21,22,23,24,25,26,27,28),
          Array(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0))))
      val mpoints = features.get("Monkey").collect()
      mpoints.size shouldBe 1
      mpoints(0) === Array(LabeledPoint(1.0,
        Vectors.sparse(29,
          Array(19,20,21,22,23,24,25,26,27,28),
          Array(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0))))
    }
  }

  describe("fit tests") {
    //TODO: Once we stabilize on parameter ingestion.
  }

  describe("get estimator") {
    //TODO: once we stabilize on parameter ingestion.
  }
}


/**
  * Tests HoldOutSetLogisticRegressionFurnace
  */
class HoldOutSetLogisticRegressionFurnaceSpec extends FunSpec
  with Matchers with BeforeAndAfter with ParallelTestExecution with BeforeAndAfterAll {

  override def beforeAll = {
    super.beforeAll
  }

  override def afterAll = {
    super.afterAll
  }

  val inFile : String = "test_data/multiclass_labeled_points.json"
  var pipeline : FeaturePipeline = _
  val engine = new EmbeddedEngine

  /** Sets up the test object, spark context, & feature pipeline */
  before {
    // Define a pipeline that generates feature vectors
    pipeline = (FeaturePipelineBuilder.named("IntentPipeline")
      += (FeaturePipelineBuilder.entry("convertToIndex", new IndexTransformer(0), "convertToTokens"))
      += (FeaturePipelineBuilder.entry("convertToTokens", new TokenTransformer, "contentExtractor", "languageDetector", "contentDetector"))
      += (FeaturePipelineBuilder.entry("languageDetector", new LanguageDetector, "$document", "contentDetector"))
      += (FeaturePipelineBuilder.entry("contentExtractor", new ContentExtractor, "$document"))
      += (FeaturePipelineBuilder.entry("contentDetector", new ContentTypeDetector, "$document"))
      := ("convertToIndex"))
  }

  describe("featurize data tests") {
    it("produces correct labelled point data map") {
      implicit val formats = org.json4s.DefaultFormats
      val inFilePath = getClass.getClassLoader.getResource(inFile).getPath()
      val primedPipeline = pipeline.prime(
        Source.fromFile(inFilePath).getLines.map(line => parse(line).extract[JObject]))
      val xval = new HoldOutSetLogisticRegressionFurnaceBuilder().build(engine)
      val features = xval.featurizeData(() => {
        Source.fromFile(inFilePath)
          .getLines.map(line => parse(line).extract[JObject])
      }, new KClassDataFrameGeneratorBuilder().build(), List(primedPipeline)).head
      features.get.size shouldBe 2
      features.get.keys.toList shouldBe List("Intent to Buy", "Monkey")
      val itbpoints = features.get("Intent to Buy").collect()
      itbpoints.size shouldBe 2
      itbpoints(0) === Array(LabeledPoint(1.0,
        Vectors.sparse(29,
          Array(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18),
          Array(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0))))
      itbpoints(1) === Array(LabeledPoint(0.0,
        Vectors.sparse(29,
          Array(19,20,21,22,23,24,25,26,27,28),
          Array(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0))))
      val mpoints = features.get("Monkey").collect()
      mpoints.size shouldBe 1
      mpoints(0) === Array(LabeledPoint(1.0,
        Vectors.sparse(29,
          Array(19,20,21,22,23,24,25,26,27,28),
          Array(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0))))
    }
  }

  describe("fit tests") {
    //TODO: Once we stabalize on parameter ingestion.
  }

  describe("get estimator") {
    //TODO: once we stabalize on parameter ingestion.
  }
}
