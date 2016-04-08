package com.idibon.ml.feature.indexer

import com.idibon.ml.alloy.Alloy.Reader
import com.idibon.ml.feature._
import com.idibon.ml.alloy.Alloy
import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}

import org.json4s.{JInt, JObject}
import org.json4s.JsonDSL._
import org.apache.spark.mllib.linalg.{SparseVector, Vector, Vectors}
import com.typesafe.scalalogging.Logger

/** Base class for index transforms
  *
  * Index transform use a Vocabulary index to convert a sequence of features
  * into a vector of counts.
  */
abstract class AbstractIndexTransformer(vocabulary: Vocabulary)
    extends FeatureTransformer with TerminableTransformer {

  /** Saves the transform state to an alloy
    *
    * @param writer alloy output interface
    * @return configuration JSON
    */
  def save(writer: Alloy.Writer): Option[JObject] = {
    val fos = new FeatureOutputStream(
      writer.resource(AbstractIndexTransformer.INDEX_RESOURCE_NAME))

    try {
      vocabulary.save(fos)
    } finally {
      fos.close()
    }

    Some("minimumObservations" -> vocabulary.minimumObservations)
  }

  /** Number of dimensions in generated vectors */
  def numDimensions = Some(vocabulary.size)

  /** Removes unneeded features from the vocabulary to save space
    *
    * @param pred predicate function returning true for pruned features
    */
  def prune(pred: (Int) => Boolean) = vocabulary.prune(pred)

  /** Returns the feature based on its assigned dimension
    *
    * @param i dimension to invert
    * @return feature assigned at the dimension, if assigned
    */
  def getFeatureByIndex(i: Int): Option[Feature[_]] = vocabulary.invert(i)

  /** Converts a list of features to a vector of counts
    *
    * @param features list of features
    * @return vector of feature counts
    */
  protected def toVector(features: Seq[Feature[_]]): SparseVector = {
    /* convert all of the features to indices, and sort by ascending
     * index to construct a list of unique indices and the number
     * of times each index appears, to generate a SparseVector of
     * feature counts */
    val indices = features.map(f => vocabulary(f)).sorted.toArray
    val dimensions = vocabulary.size

    /* skip over OOV features at the start, and return a zero vector
     * if every feature is OOV */
    var i = indices.indexWhere(_ > Vocabulary.OOV, 0)
    if (i == -1) return Vectors.sparse(dimensions, Array(), Array()).asInstanceOf[SparseVector]

    val uniques = new Array[Int](indices.length - i)
    val counts = new Array[Double](indices.length - i)

    var active = 0
    while (i > -1) {
      /* find the next feature transition in the non-uniq'd list, and
       * set the count to be the distance from the current location to
       * the next unique feature */
      val next = indices.indexWhere(_ != indices(i), i)
      uniques(active) = indices(i)
      val tf = if (next > -1) { next - i } else { indices.length - i }
      counts(active) = tf
      active += 1
      i = next
    }

    Vectors.sparse(vocabulary.size, uniques.slice(0, active), counts.slice(0, active))
      .asInstanceOf[SparseVector]
  }

  protected def logger: Logger
}

/** Base class for index transform loaders */
abstract class AbstractIndexTransformLoader[T] extends ArchiveLoader[T] {

  /** Creates a new instance of the transform with the provided vocabulary */
  protected def newTransform(v: Vocabulary): T

  def load(engine: Engine, reader: Option[Alloy.Reader],
      config: Option[JObject]): T = {

    val vocabulary: Vocabulary = loadVocabulary(reader, config)
    newTransform(vocabulary)
  }

  /**
    * Loads the vocabulary if there is one to load, else returns a mutable vocabulary.
    *
    * @param reader location within Alloy for loading any resources
    *   previous preserved by a call to save
    * @param config archived configuration data returned by a previous call to save
    * @return a vocabulary that maps feature -> dimension index
    */
  def loadVocabulary(reader: Option[Reader], config: Option[JObject]): Vocabulary = {
    implicit val formats = org.json4s.DefaultFormats
    val observations = config.map(_ \ "minimumObservations")
      .collect({case j: JInt => j})
      .map(_.num.intValue())
      .getOrElse(0)
    val vocabulary = reader match {
      case None => {
        // when no reader exists, create an empty, mutable vocabulary
        new MutableVocabulary
      }
      case Some(reader) => {
        // otherwise, reload the vocabulary from the alloy
        val fis = new FeatureInputStream(
          reader.resource(AbstractIndexTransformer.INDEX_RESOURCE_NAME))
        try {
          Vocabulary.load(fis)
        } finally {
          fis.close()
        }
      }
    }
    vocabulary.minimumObservations = observations
    vocabulary
  }
}

object AbstractIndexTransformer {
  val INDEX_RESOURCE_NAME = "featureIndex"
  val IDF_RESOURCE_NAME = "idfFeatureIndex"
}
