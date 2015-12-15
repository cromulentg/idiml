import com.idibon.ml.feature.{Archivable,Feature,FeatureTransformer}
import org.apache.spark.mllib.linalg.{IdibonBLAS, Vector, Vectors}
import scala.reflect.runtime.universe._

package com.idibon.ml.feature.indexer {

import com.idibon.ml.alloy.{Codec, Alloy}
import org.json4s._

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
    *   
    *   @author Michelle Casbon <michelle@idibon.com>
    */
  class IndexTransformer extends FeatureTransformer with Archivable {

    private[indexer] var featureIndex = scala.collection.mutable.Map[Feature[_], Int]()

    def getFeatureIndex = featureIndex

    def save(writer: Alloy.Writer) = {
      val dos = writer.resource("featureIndex")

      // Save the dimensionality of the featureIndex map so we know how many times to call Codec.read() at load time
      Codec.VLuint.write(dos, featureIndex.size)

      // Store each key/value pair in sequence
      // Since we don't know the concrete type of the key, store the class name, then have the feature save itself
      featureIndex.foreach{
        case (key, value) => {
          Codec.String.write(dos, key.getClass.getName)
          key.save(dos)
          Codec.VLuint.write(dos, value)
        }
      }

      // No config to return
      None
    }

    def load(reader: Alloy.Reader, config: Option[JObject]): this.type = {
      val dis = reader.resource("featureIndex")

      featureIndex = scala.collection.mutable.Map[Feature[_], Int]()

      // Retrieve the number of elements in the featureIndex map
      val size = Codec.VLuint.read(dis)

      1 to size foreach { _ =>
        // Retrieve the class name
        val featureClass = Codec.String.read(dis)
        // Retrieve the feature
        // To instantiate the saved feature, it must have a parameterless constructor defined. Since we will
        // immediately load the saved values, default constructor values are fine
        var key : Feature[_] = Class.forName(featureClass).newInstance().asInstanceOf[Feature[_]]

        // Load feature contents
        key.load(dis)
        // Load the index value for this feature
        val value = Codec.VLuint.read(dis)

        featureIndex(key) = value
      }

      this
    }

    /** This function performs a lookup on the provided feature. It returns the unique index associated with the
      * feature. If the feature has not been seen before, it is added to the map and assigned a new index.
      *
      * @param feature
      * @return unique index
      */
    private[indexer] def lookupOrAddToFeatureIndex(feature: Feature[_]): Int = {
      if (featureIndex.isEmpty) {
        featureIndex(feature) = 0
      }
      else if (!featureIndex.contains(feature)) {
        featureIndex(feature) = featureIndex.size
      }

      featureIndex(feature)
    }

    /**
      * This function maps all features to unique indexes. It returns the size of the feature vocabulary.
      *
      * @param features
      * @return unique index
      */
    private[indexer] def createFeatureIndex(features: Seq[Feature[_]]): Int = {

      val allIndexes = features.map(t => lookupOrAddToFeatureIndex(t))

      allIndexes.max + 1
    }

    /**
      * This function creates the feature index map from all provided features. It creates a vector for each feature
      * and sums them all together. It returns this consolidated feature vector.
      *
      * @param features
      * @return feature vector representing all provided features
      */
    private[indexer] def getFeatureVector(features: Seq[Feature[_]]): Vector = {
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

    def apply(features: Seq[Feature[_]]): Vector = {
      if (features.length < 1)
        Vectors.zeros(0)
      else {
        getFeatureVector(features)
      }
    }
  }
}
