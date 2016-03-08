package com.idibon.ml.feature.indexer

import scala.collection.mutable.ArrayBuffer

import com.idibon.ml.feature.{Chain, Feature, Freezable}
import com.idibon.ml.common.Archivable
import com.typesafe.scalalogging.StrictLogging

import org.apache.spark.mllib.linalg.Vector

/** IndexTransformer for feature chains
  *
  * Generates a chain of feature vectors, where each link is the vector
  * representation of a list of features. Used for sequence classification.
  */
class ChainIndexTransformer(vocabulary: Vocabulary)
    extends AbstractIndexTransformer(vocabulary)
    with Archivable[ChainIndexTransformer, ChainIndexTransformLoader]
    with Freezable[ChainIndexTransformer] with StrictLogging {

  def this() { this(new MutableVocabulary()) }

  def apply(chains: Chain[Seq[Feature[_]]]*): Chain[Vector] = {
    val joined = if (chains.length == 1) chains.head else join(chains)
    joined.map(features => toVector(features.value))
  }

  /** Performs a link-wise concatenation of feature lists across chains
    *
    * Given multiple chains (which must have the same number of links),
    * produces a single chain where each link includes all of the features
    * from each input chain
    */
  def join(chains: Seq[Chain[Seq[Feature[_]]]]) = {
    val joined = new ArrayBuffer[Seq[Feature[_]]](chains.head.size)
    chains.head.foreach(joined += _.value)
    chains.tail.foreach(chain => {
      require(chain.size == joined.size, "Incompatible chain lengths!")
      chain.toIterable.zipWithIndex.foreach({ case (link, index) => {
      joined(index) = joined(index) ++: link.value
      }})
    })
    Chain(joined)
  }

  def freeze(): ChainIndexTransformer =
    new ChainIndexTransformer(vocabulary.freeze)
}

/** Paired loader class for IndexTransformer */
class ChainIndexTransformLoader
    extends AbstractIndexTransformLoader[ChainIndexTransformer] {
  protected def newTransform(v: Vocabulary) = new ChainIndexTransformer(v)
}
