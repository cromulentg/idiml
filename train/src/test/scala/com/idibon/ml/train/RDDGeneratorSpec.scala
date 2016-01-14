package com.idibon.ml.train

import com.idibon.ml.feature.{FeaturePipeline, ContentExtractor, FeaturePipelineBuilder}
import com.idibon.ml.feature.indexer.IndexTransformer
import com.idibon.ml.feature.tokenizer.TokenTransformer
import com.idibon.ml.test.VerifyLogging
import java.io._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
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

  var generator : RDDGenerator = _
  var conf : SparkConf = _
  var sc : SparkContext = _
  var pipeline : FeaturePipeline = _
  var infilePath : String = _
  var fileContent : String = _
  var file : File = _

  /** Sets up the test object, spark context, & feature pipeline */
  before {
    generator = new RDDGenerator()
    generator shouldBe an[RDDGenerator]

    // Instantiate the Spark environment
    conf = new SparkConf().setAppName("idiml").setMaster("local[8]").set("spark.driver.host", "localhost")
    sc = new SparkContext(conf)

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

  /** Generates a temporary file for creating labeled points from
    *
    * @param fileContent: the set of document/annotation objects in json format. Generated from idibin script
    *                     bin/open_source_integration/export_training_to_idiml.rb
   */
  def create_sample_file(fileContent: String) = {
    // Create a file to create RDDs from
    infilePath = "/tmp/RDDGeneratorSpec.txt"

    file = new File(infilePath)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(fileContent)
    bw.close()
  }

  describe("RDDGenerator") {
    it("should generate LabeledPoint RDD's correctly") {
      create_sample_file("{ \"content\":\"Who drives a chevy malibu? Would you recommend it?\", \"metadata\": { \"iso_639_1\":\"en\"}, \"annotations\": [{ \"label\": { \"name\":\"Intent to Buy\"}, \"isPositive\":true}]}\n")

      val training = generator.getLabeledPointRDDs(sc, infilePath, pipeline)
      training.isDefined shouldBe true

      training.size shouldBe 1

      val rdd = training.get("Intent to Buy")
      rdd shouldBe an[RDD[_]]

      val labeled_point_result = LabeledPoint(1.0, Vectors.sparse(19, Array(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18), Array(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0)))
      rdd.collect().head shouldBe labeled_point_result

      // Remove the temporary file we generated
      file.delete()
    }

  }
}
