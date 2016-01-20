package com.idibon.ml.train

import com.idibon.ml.feature.{FeaturePipeline, ContentExtractor, FeaturePipelineBuilder}
import com.idibon.ml.feature.indexer.IndexTransformer
import com.idibon.ml.feature.tokenizer.TokenTransformer
import com.idibon.ml.test.Spark
import com.idibon.ml.test.VerifyLogging
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.scalatest.{BeforeAndAfterAll, BeforeAndAfter, Matchers, FunSpec}

/** Verifies the functionality of RDDGenerator
  *
  */
class RDDGeneratorSpec extends FunSpec with Matchers
  with BeforeAndAfter with BeforeAndAfterAll with VerifyLogging {

  override val loggerName = classOf[RDDGenerator].getName

  override def beforeAll = {
    super.beforeAll
    initializeLogging
  }

  override def afterAll = {
    shutdownLogging
    super.afterAll
  }

  val inFile : String = "test_data/labeled_points.json"
  var generator : RDDGenerator = _
  var pipeline : FeaturePipeline = _

  /** Sets up the test object, spark context, & feature pipeline */
  before {
    generator = new RDDGenerator()
    generator shouldBe an[RDDGenerator]

    // Define a pipeline that generates feature vectors
    pipeline = (FeaturePipelineBuilder.named("IntentPipeline")
      += (FeaturePipelineBuilder.entry("convertToIndex", new IndexTransformer, "convertToTokens"))
      += (FeaturePipelineBuilder.entry("convertToTokens", new TokenTransformer, "contentExtractor"))
      += (FeaturePipelineBuilder.entry("contentExtractor", new ContentExtractor, "$document"))
      := ("convertToIndex"))
  }

  after {
    // reset the logged messages after every test
    resetLog
  }

  describe("RDDGenerator") {
    it("should generate LabeledPoint RDD's correctly") {
      val inFilePath = getClass.getClassLoader.getResource(inFile).getPath()
      val training = generator.getLabeledPointRDDs(Spark.sc, inFilePath, pipeline)
      training.isDefined shouldBe true

      training.size shouldBe 1

      val rdd = training.get("Intent to Buy")
      rdd shouldBe an[RDD[_]]

      val labeled_point_result = LabeledPoint(1.0, Vectors.sparse(19, Array(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18), Array(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0)))
      rdd.collect().head shouldBe labeled_point_result
    }
  }
}
