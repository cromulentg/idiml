package com.idibon.ml.predict.ml.metrics

import com.idibon.ml.common.EmbeddedEngine
import com.idibon.ml.feature.contenttype.ContentTypeDetector
import com.idibon.ml.feature.indexer.IndexTransformer
import com.idibon.ml.feature.language.LanguageDetector
import com.idibon.ml.feature.tokenizer.TokenTransformer
import com.idibon.ml.feature.{ContentExtractor, FeaturePipeline, FeaturePipelineBuilder}
import com.idibon.ml.predict.ml.TrainingSummary
import com.idibon.ml.predict.{Classification, PredictOptions, Document, PredictModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.json4s.JObject
import org.json4s.native.JsonMethods.parse
import org.scalatest.{BeforeAndAfter, BeforeAndAfterAll, FunSpec, Matchers}

import scala.io.Source

/** Verifies the functionality of KClassDataFrameGenerator
  *
  */
class MetricHelperSpec extends FunSpec with Matchers
  with BeforeAndAfter with BeforeAndAfterAll {

  val inFile : String = "fixtures/labeled_points.json"
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

  describe("createPositiveLPs tests") {
    it("should generate correct createPositiveLPs for consumption") {
      implicit val formats = org.json4s.DefaultFormats
      val inFilePath = getClass.getClassLoader.getResource(inFile).getPath()
      val primedPipeline = pipeline.prime(
        Source.fromFile(inFilePath).getLines.map(line => parse(line).extract[JObject]))
      val gen = new Dummy with MetricHelper()
      val (dToL, points) = gen.createPositiveLPs(primedPipeline, () => {
        Source.fromFile(inFilePath)
          .getLines.map(line => parse(line).extract[JObject])
      })
      dToL shouldBe Map(0.0 -> "Intent to Buy")
      val vect = Vectors.sparse(
        19,
        Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18),
        Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
      points.head shouldBe (List(0.0), vect)

    }
  }

  describe("getModelThresholds tests") {
    it("should get model thresholds if they exist") {
      val gen = new Dummy with MetricHelper()
      val models = List(("test",
        DummyModel(Some(Seq(
          new TrainingSummary("test",
            Seq(new FloatMetric(MetricTypes.BestF1Threshold, MetricClass.Binary, 0.6f))))))))
      val actual = gen.getModelThresholds(models)
      actual shouldBe Map("test" -> 0.6f)
    }
    it("should not fail if thresholds don't exist") {
      val gen = new Dummy with MetricHelper()
      val models = List(("test", new DummyModel()), ("test2", new DummyModel()))
      val actual = gen.getModelThresholds(models)
      actual shouldBe Map("test" -> 0.0f, "test2" -> 0.0f)
    }
  }
}

case class Dummy()

case class DummyModel(training: Option[Seq[TrainingSummary]] = None) extends PredictModel[Classification] {
  val reifiedType = classOf[DummyModel]
  override def predict(document: Document, options: PredictOptions): Seq[Classification] = ???
  override def getFeaturesUsed(): Vector = ???
  override def getEvaluationMetric(): Double = ???
  override def getTrainingSummary(): Option[Seq[TrainingSummary]] = training
}
