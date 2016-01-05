package com.idibon.ml.predict

import com.idibon.ml.alloy.IntentAlloy
import com.idibon.ml.feature.bagofwords.BagOfWordsTransformer
import com.idibon.ml.feature.tokenizer.TokenTransformer
import com.idibon.ml.feature.{FeaturePipeline, StringFeature}
import com.idibon.ml.predict.ml.IdibonLogisticRegressionModel
import org.apache.spark.ml.classification.IdibonSparkLogisticRegressionModelWrapper
import org.apache.spark.mllib.feature
import org.json4s._
import org.json4s.native.JsonMethods._


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

    // get some data.
    val text: String = "Everybody loves replacing hadoop with spark because it's much faster. a b d"
    val rawFeature = new StringFeature(text)
    // do some featurization
    val tokenizer = new TokenTransformer()
    val tokenized = tokenizer.apply(Seq[StringFeature](rawFeature))
    println(tokenized)
    val bOWer = new BagOfWordsTransformer()
    val bagOfWords = bOWer.apply(tokenized)
    println(bagOfWords)
    // convert to indexes
    val hashingTF = new feature.HashingTF()
    val hashedIndexes = hashingTF.transform(bagOfWords)
    println(hashedIndexes)

    // load some stock model
    val weights = hashedIndexes  // Just using these as a placeholder.
    val intercept = 0.0
    val sparkLRModel = new IdibonSparkLogisticRegressionModelWrapper("myModel", weights, intercept)
    val idibonModel = new IdibonLogisticRegressionModel()
    idibonModel.lrm = sparkLRModel

    println(sparkLRModel.predictProbability(hashedIndexes))
    println(idibonModel.predict(hashedIndexes, false, 0.0).toString)
  }

  /**
    * Next iteration of the above. WIP. Need Gary's branch...
    * @return
    */
  def realisticExample() = {
    val json = parse(
      """{
"transforms":[
  {"name":"A","class":"com.idibon.ml.feature.tokenizer.TokenTransformer"},
  {"name":"B","class":"com.idibon.ml.feature.bagofwords.BagOfWordsTransformer"},
  {"name":"C","class":"com.idibon.ml.feature.indexer.IndexTransformer"}],
"pipeline":[
  {"name":"A","inputs":["$document"]},
  {"name":"B","inputs":["A"]},
  {"name":"C","inputs":["B"]},
  {"name":"$output","inputs":["C"]}]}""").asInstanceOf[JObject]
    val pipeline = new FeaturePipeline()
    val reader = new IntentAlloy() //TODO create pipeline via "builder"
    pipeline.load(reader.reader(), Some(json.asInstanceOf[JObject]))
  }

}
