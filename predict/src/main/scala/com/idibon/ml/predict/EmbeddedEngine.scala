package com.idibon.ml.predict

import com.idibon.ml.feature.bagofwords.BagOfWordsTransformer
import com.idibon.ml.feature.tokenizer.TokenTransformer
import com.idibon.ml.feature.StringFeature
import com.idibon.ml.predict.ml.IdibonLogisticRegressionModel
import org.apache.spark.ml.classification.IdibonSparkLogisticRegressionModelWrapper
import org.apache.spark.mllib.feature

/**
  * Toy engine class that stitches together Idibon's feature pipeline and Spark's LR
  * to perform a prediction.
  */
class EmbeddedEngine extends Engine {

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
    println(idibonModel.predict(hashedIndexes, false).toString)

  }
}
