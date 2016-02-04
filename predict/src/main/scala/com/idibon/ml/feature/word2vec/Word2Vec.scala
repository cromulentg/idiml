package com.idibon.ml.feature.word2vec

import com.idibon.ml.alloy.Alloy
import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}
import com.idibon.ml.feature._

import org.apache.spark._
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.linalg._

import org.json4s._

/**
  * Word2Vec feature for creating vector representations from sequences of strings
  *
  * @param sc SparkContext object
  * @param model Word2VecModel model object
  * @param path path to directory where the model is stored (a String)
  */
class Word2VecTransformer(val sc: SparkContext, val model: Word2VecModel, val path: String) extends FeatureTransformer
  with Archivable[Word2VecTransformer,Word2VecTransformerLoader] {

  val vectors = model.getVectors
  private val (_, firstVector) = vectors.head
  val vectorSize = firstVector.size

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

  /**
    * Saves the path to the model object in a JObject. (This path is the
    * only info needed to reload the Word2VecTransformer.)
    *
    * @param writer destination within Alloy for any resources that
    *   must be preserved for this object to be reloadable
    * @return Some[JObject] of configuration data that must be preserved
    *   to reload the object. None if no configuration is needed
    */
  def save(writer: Alloy.Writer): Option[JObject] = {
    Some(JObject(JField("path", JString(path))))
  }

}

class Word2VecTransformerLoader extends ArchiveLoader[Word2VecTransformer] {

  /**
    * Loads a Word2VecTransformer from a path pointing to a saved Word2VecModel
    *
    * @param engine implementation of the Engine trait
    * @param reader location within Alloy for loading any resources
    *   previous preserved by a call to
    * @param config archived configuration data returned by a previous
    * @return a Word2VecTransformer
    */
  def load(engine: Engine, reader: Option[Alloy.Reader], config: Option[JObject]): Word2VecTransformer = {
    implicit val formats = DefaultFormats
    val path = (config.get \ "path").extract[String]
    val model = Word2VecModel.load(engine.sparkContext, path)
    val transformer = new Word2VecTransformer(engine.sparkContext, model, path)
    transformer
  }

}
