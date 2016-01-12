package com.idibon.ml.predict

import com.idibon.ml.alloy.IntentAlloy
import com.idibon.ml.feature.bagofwords.BagOfWordsTransformer
import com.idibon.ml.feature.indexer.IndexTransformer
import com.idibon.ml.feature.tokenizer.TokenTransformer
import com.idibon.ml.feature.{FeatureTransformer, FeaturePipelineBuilder, FeaturePipeline, StringFeature}
import com.idibon.ml.predict.ml.IdibonLogisticRegressionModel
import org.apache.spark.ml.classification.IdibonSparkLogisticRegressionModelWrapper
import org.apache.spark.mllib.feature
import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.native.JsonMethods.{compact, parse, render}


/**
  * Toy engine class that stitches together Idibon's feature pipeline and Spark's LR
  * to perform a prediction.
  */
class EmbeddedEngine extends Engine {

  /**
    * Very crude POC. This will change as we add more to the code base.
    */
  def start() = {
    println("Called from Scala")
    // create pipeline manually
    val pipeline: FeaturePipeline = (FeaturePipelineBuilder.named("StefansPipeline")
      += (FeaturePipelineBuilder.entry("convertToIndex", new IndexTransformer, "bagOfWords"))
      += (FeaturePipelineBuilder.entry("bagOfWords", new BagOfWordsTransformer, "convertToTokens"))
      += (FeaturePipelineBuilder.entry("convertToTokens", new TokenTransformer, "contentExtractor"))
      += (FeaturePipelineBuilder.entry("contentExtractor", new DocumentExtractor, "$document"))
      := ("convertToIndex"))
    // get some data.
    val text: String = "Everybody loves replacing hadoop with spark because it's much faster. a b d"
    val doc: JObject = ( "content" -> text )
    // create feature vector
    val featureVector = pipeline(doc).head
    println(featureVector)

    // load some stock model
    val weights = featureVector  // Just using these as a placeholder.
    val intercept = 0.0
    val sparkLRModel = new IdibonSparkLogisticRegressionModelWrapper("myModel", weights, intercept)
    val idibonModel = new IdibonLogisticRegressionModel()
    idibonModel.lrm = sparkLRModel

    println(sparkLRModel.predictProbability(featureVector))
    println(idibonModel.predict(featureVector, new PredictOptionsBuilder().build()).toString)
  }

}

/**
  * Remove this class after Michelle's commit.
  */
class DocumentExtractor extends FeatureTransformer {
  def apply(document: JObject): Seq[StringFeature] = {
    List(StringFeature((document \ "content").asInstanceOf[JString].s))
  }
}
