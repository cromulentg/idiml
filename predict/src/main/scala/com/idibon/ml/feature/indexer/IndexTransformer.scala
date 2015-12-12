import com.idibon.ml.feature.{Archivable,Feature,FeatureTransformer}
import org.apache.spark.mllib.linalg.{IdibonBLAS, Vector, Vectors}
import scala.reflect.runtime.universe._

package com.idibon.ml.feature.indexer {

/** Internal implementation of the indexer FeatureTransformer */

  /**
    * A FeatureTransformer for indices that:
    *   - maintains a map Feature to Int of size V, where V is the total size of the known feature vocabulary.
    *     This defines a unique index i for every feature in V
    *   - implements apply by mapping all input features to their indices (nb: not all input features will be in the
    *     vocabulary)
    *   - creates V-dimensional SparseVector for each input feature in the vocabulary, where each vector is defined a
    *     vector[j] = { 1 if map[feature] == j, else 0 }
    *   - sums the vectors to produce the returned vector
    */
  class IndexTransformer extends FeatureTransformer[Index] {

    def input = IndexTransformer.input

    def options = None

    var featureIndex = scala.collection.mutable.Map[Feature[_], Int]()

    /** This function performs a lookup on the provided feature. It returns the unique index associated with the
      * feature. If the feature has not been seen before, it is added to the map and assigned a new index.
      *
      * @param feature
      * @return unique index
      */
    def lookupOrAddToFeatureIndex(feature: Feature[_]): Int = {
      if (featureIndex.isEmpty) {
        featureIndex(feature) = 0
      }
      else if (!featureIndex.contains(feature)) {
        featureIndex(feature) = featureIndex.valuesIterator.max + 1
      }

      featureIndex(feature)
    }

    /**
      * This function maps all features to unique indexes. It returns the size of the feature vocabulary.
      *
      * @param features
      * @return unique index
      */
    def createFeatureIndex(features: Seq[Feature[_]]): Int = {

      val uniqueIndexes = features.map(t => lookupOrAddToFeatureIndex(t))

      uniqueIndexes.max + 1
    }

    /**
      * This function creates the feature index map from all provided features. It creates a vector for each feature
      * and sums them all together. It returns this consolidated feature vector.
      *
      * @param features
      * @return feature vector representing all provided features
      */
    def getFeatureVector(features: Seq[Feature[_]]): Vector = {
      // Create our indexes and find out how large the return vector should be
      val vocabSize = createFeatureIndex(features)

      // BLAS only supports adding to a dense vector, so let's instantiate one full of zeroes
      val featureVector = Vectors.dense(Array.fill[Double](vocabSize)(0))

      features.map(f => {
                val index = lookupOrAddToFeatureIndex(f)
                val singleFeature = Vectors.sparse(vocabSize, Seq((index, 1.0)))
                IdibonBLAS.axpy(1, singleFeature, featureVector) })

      featureVector.toSparse
    }

    def apply(inputFeatures: scala.collection.immutable.Map[String, Seq[Feature[_]]]): Seq[Index] = {
      // extract input features
      val features: Seq[Feature[_]] = inputFeatures("features").map(f => f.getAs[Feature[_]])

      if (features.length < 1)
        Seq()
      else {
        Seq(new Index(getFeatureVector(features)))
      }
    }
  }

  private[indexer] object IndexTransformer {
    lazy val input = scala.collection.immutable.Map("features" -> typeOf[Feature[_]])
  }

}
