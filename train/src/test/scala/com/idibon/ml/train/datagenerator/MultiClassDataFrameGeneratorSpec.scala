package com.idibon.ml.train.datagenerator

import com.idibon.ml.common.EmbeddedEngine
import com.idibon.ml.feature.indexer.IndexTransformer
import com.idibon.ml.feature.language.LanguageDetector
import com.idibon.ml.feature.tokenizer.TokenTransformer
import com.idibon.ml.feature.contenttype.ContentTypeDetector
import com.idibon.ml.feature.{ContentExtractor, FeaturePipeline, FeaturePipelineBuilder}
import com.idibon.ml.train.alloy.MultiClass
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.json4s.JObject
import org.json4s.native.JsonMethods.parse
import org.scalatest.{BeforeAndAfter, BeforeAndAfterAll, FunSpec, Matchers}

import scala.io.Source

/** Verifies the functionality of MultiClassDataFrameGenerator
  *
  */
class MultiClassDataFrameGeneratorSpec extends FunSpec with Matchers
  with BeforeAndAfter with BeforeAndAfterAll {

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
      += (FeaturePipelineBuilder.entry("convertToIndex", new IndexTransformer, "convertToTokens"))
      += (FeaturePipelineBuilder.entry("convertToTokens", new TokenTransformer, "contentExtractor", "languageDetector", "contentDetector"))
      += (FeaturePipelineBuilder.entry("languageDetector", new LanguageDetector, "$document", "contentDetector"))
      += (FeaturePipelineBuilder.entry("contentExtractor", new ContentExtractor, "$document"))
      += (FeaturePipelineBuilder.entry("contentDetector", new ContentTypeDetector, "$document"))
      := ("convertToIndex"))
  }

  describe("MultiClassDataFrameGenerator") {

    val dp1 = LabeledPoint(0,
      Vectors.sparse(29,
        Array(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18),
        Array(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0)))
    val dp2 = LabeledPoint(1,
      Vectors.sparse(29,
        Array(19,20,21,22,23,24,25,26,27,28),
        Array(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0)))
    val itbLP = LabeledPoint(0.0, Vectors.zeros(0))
    val monkeyLP = LabeledPoint(1.0, Vectors.zeros(0))
    val expectedTrainingData = Map("Intent to Buy" -> List(itbLP),
                       "Monkey" -> List(monkeyLP),
                        MultiClass.MODEL_KEY -> List(dp1, dp2))

    it("should generate LabeledPoint lists correctly") {
      implicit val formats = org.json4s.DefaultFormats

      val inFilePath = getClass.getClassLoader.getResource(inFile).getPath()
      val primedPipeline = pipeline.prime(
        Source.fromFile(inFilePath).getLines.map(line => parse(line).extract[JObject]))
      val gen = new MultiClassDataFrameGeneratorBuilder().build()
      val baseTraining = gen.createPerLabelLPs(primedPipeline, () => {
        Source.fromFile(inFilePath)
          .getLines.map(line => parse(line).extract[JObject])
      })
      baseTraining shouldBe expectedTrainingData
    }

    it("should convert labeled point lists to RDDs correctly") {
      val gen = new MultiClassDataFrameGeneratorBuilder().build()
      // very conversion into RDD
      val training = gen.createPerLabelRDDs(engine, expectedTrainingData)
      training.size shouldBe 3
      List(("Intent to Buy", Array(itbLP)), ("Monkey", Array(monkeyLP)), (MultiClass.MODEL_KEY, Array(dp1, dp2)))
        .foreach({case (label, dps) => {
        val rdd = training(label)
        rdd shouldBe an[RDD[_]]
        rdd.collect() shouldBe dps
      }})
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

      val dataFrameMap = new MultiClassDataFrameGeneratorBuilder().build().getLabeledPointData(engine, primedPipeline, () => {
        Source.fromFile(inFilePath)
          .getLines.map(line => parse(line).extract[JObject])
      })
      val sqlContext = new org.apache.spark.sql.SQLContext(engine.sparkContext)
      dataFrameMap.get.foreach({ case (label, dataframe) => {
        val results = dataframe.collect()
        results === expectedTrainingData(label).toArray
      }})
    }
  }
}
