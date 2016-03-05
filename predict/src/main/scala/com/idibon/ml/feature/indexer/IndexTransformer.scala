package com.idibon.ml.feature.indexer

import com.idibon.ml.feature.{Feature, Freezable}
import com.idibon.ml.common.Archivable
import com.typesafe.scalalogging.StrictLogging

import org.apache.spark.mllib.linalg.Vector

/** FeatureTransformer that collapses converts features to a Vector */
class IndexTransformer(private[indexer] val vocabulary: Vocabulary)
    extends AbstractIndexTransformer(vocabulary)
    with Archivable[IndexTransformer, IndexTransformLoader]
    with Freezable[IndexTransformer] with StrictLogging {

  def this() { this(new MutableVocabulary()) }

  def apply(features: Seq[Feature[_]]*): Vector = toVector(features.flatten)

  def freeze(): IndexTransformer = new IndexTransformer(vocabulary.freeze)
}

/** Paired loader class for IndexTransformer */
class IndexTransformLoader extends AbstractIndexTransformLoader[IndexTransformer] {
  protected def newTransform(v: Vocabulary) = new IndexTransformer(v)
}
