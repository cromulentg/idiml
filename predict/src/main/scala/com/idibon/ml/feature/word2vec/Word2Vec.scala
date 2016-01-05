package com.idibon.ml.feature.word2vec

import com.idibon.ml.feature.FeatureTransformer

import org.apache.spark._
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.linalg._

/**
  * Word2Vec feature for creating vector representations from sequences of strings
  *
  * @param sc Spark context object
  * @param path path to pre-built spark word2vec model directory
  */

class Word2VecTransformer(val sc: SparkContext, val path: String) extends FeatureTransformer {
  private val model = Word2VecModel.load(sc, path)
  private val vectors = model.getVectors
  private val (_, firstVector) = vectors.head
  private val vectorSize = firstVector.size

  /**
    * Transform a sequence of strings to a vector representing the sequence. The transform
    * is performed by averaging all word vectors in the sequence
    *
    * This is based on the transform method in org.apache.spark.ml.feature.Word2Vec. That method
    * takes a DataFrame as input and has been adapted here to take a sequence of tokens.
    *
    * @param words a sequence of strings
    * @return average of vectors for all words in the sequence
    */
  def apply(words: Seq[String]): Vector = {
    val sum = Vectors.zeros(vectorSize)

    if (words.size == 0) {
      Vectors.sparse(vectorSize, Array.empty[Int], Array.empty[Double])
    } else {
      val sum = Vectors.zeros(vectorSize)
      words.foreach { word =>
        IdibonBLAS.axpy(1.0, model.transform(word), sum)
      }
      IdibonBLAS.scal(1.0 / words.size, sum)
      sum
    }
  }
}
