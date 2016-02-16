package com.idibon.ml.feature.word2vec

import java.util

import com.idibon.ml.alloy.Alloy
import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}
import com.idibon.ml.feature._

import org.apache.spark._
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.linalg._

import org.json4s._

import scala.collection.JavaConverters._

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
    * OOV words return a vector of zeros.
    *
    * @param words a sequence of strings
    * @return average of vectors for all words in the sequence
    */
  def apply(words: Seq[String]): Vector = {
    words.foldLeft(Vectors.zeros(vectorSize))({ case (accum, word) => {
      Try({
        IdibonBLAS.axpy(1.0, model.transform(word), accum)
        accum
      }).getOrElse(accum)
    }})
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
    * Loads a Word2VecTransformer from a path pointing to a saved Spark Word2VecModel or
    * a gzipped binary file output by the original Word2Vec C implementation
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
    val modelType = (config.get \ "type").extract[String]

    val model = modelType match {
      case "spark" => Word2VecModel.load(engine.sparkContext, path)
      case "bin" => {
        val reader = new Word2VecBinReader
        val binFilePath = path
        val javaWord2VecMap: util.LinkedHashMap[String, Array[Float]] = reader.parseBinFile(binFilePath)
        val scalaWord2VecMap = javaWord2VecMap.asScala.toMap
        new Word2VecModel(scalaWord2VecMap)
      }
      case _ => {
        throw new IllegalArgumentException("Invalid model type string ' " + modelType +
          "'. Currently only 'spark' and 'bin' are allowed");
      }
    }

    new Word2VecTransformer(engine.sparkContext, model, path)
  }
}

