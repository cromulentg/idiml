package com.idibon.ml.train

import com.idibon.ml.alloy.{Codec,IntentAlloy}
import com.idibon.ml.feature.{DocumentExtractor, FeaturePipeline, FeaturePipelineBuilder}
import com.idibon.ml.feature.indexer.IndexTransformer
import com.idibon.ml.feature.tokenizer.TokenTransformer
import com.idibon.ml.predict.Engine
import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.{SparkContext, SparkConf}
import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.native.JsonMethods.{compact, parse, render}

/** EmbeddedEngine
  *
  * Performs training, given a set of documents and annotations.
  *
  *
  * */
class EmbeddedEngine extends Engine {

  def start() = {
    start("/tmp/idiml.txt")
  }

  def start(filename: String) = {

    // Instantiate the Spark environment
    val conf = new SparkConf().setAppName("idiml").setMaster("local[8]").set("spark.driver.host", "localhost")
    val sc = new SparkContext(conf)

    // Define a pipeline that generates feature vectors
    val pipeline = (FeaturePipelineBuilder.named("IntentPipeline")
      += (FeaturePipelineBuilder.entry("convertToIndex", new IndexTransformer, "convertToTokens"))
      += (FeaturePipelineBuilder.entry("convertToTokens", new TokenTransformer, "contentExtractor"))
      += (FeaturePipelineBuilder.entry("contentExtractor", new DocumentExtractor, "$document"))
      := ("convertToIndex"))

    val labels = new RDDGenerator(sc, filename, pipeline).getLabeledPointRDDs

    // TODO: use the labeledPoints to do some training

    val alloy = new IntentAlloy()

    // Save the pipeline definition
    val test = pipeline.save(alloy.writer.within("IntentPipeline"))
    val alloyMetaJson: JObject = ("IntentPipeline" -> test)
    val writer = alloy.writer.within("IntentPipeline").resource("config.json")
    Codec.String.write(writer, compact(render(alloyMetaJson)))

    // Load the pipeline definition again
    val reader = alloy.reader.within("IntentPipeline").resource("config.json")
    val config = Codec.String.read(reader)
    val newPipelineConfig: JObject = (parse(config) \ "IntentPipeline").asInstanceOf[JObject]
    val newPipeline2 = (new FeaturePipeline).load(alloy.reader().within("IntentPipeline"), Some(newPipelineConfig))

  }
}

