package com.idibon.ml.feature.indexer

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

  def apply(chain: Chain[Seq[Feature[_]]]): Chain[Vector] =
    chain.map(features => toVector(features.value))

  def freeze(): ChainIndexTransformer =
    new ChainIndexTransformer(vocabulary.freeze)
}

/** Paired loader class for IndexTransformer */
class ChainIndexTransformLoader
    extends AbstractIndexTransformLoader[ChainIndexTransformer] {
  protected def newTransform(v: Vocabulary) = new ChainIndexTransformer(v)
}
