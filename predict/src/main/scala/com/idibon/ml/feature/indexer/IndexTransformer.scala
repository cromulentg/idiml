import com.idibon.ml.feature._
import org.apache.spark.mllib.linalg.{IdibonBLAS, Vector, Vectors}
import com.typesafe.scalalogging.StrictLogging

package com.idibon.ml.feature.indexer {

import com.idibon.ml.alloy.{Alloy, Codec}
import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}
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
    *   @author Stefan Krawczyk <stefan@idibon.com>
    *
    * @param minimumObservations the minimum number of times a feature must be
    *   observed before including it in the vocabulary
    * @param frozen whether the index size is frozen and can be added to.
    * @param frozenSize the frozen size - needed incase the index is pruned
    *   once frozen.
    */
  class IndexTransformer(private[indexer] val vocabulary: Vocabulary)
      extends FeatureTransformer
      with Archivable[IndexTransformer, IndexTransformLoader]
      with TerminableTransformer
      with StrictLogging {

    def this() {
      this(new MutableVocabulary())
    }

    def save(writer: Alloy.Writer) = {
      val fos = new FeatureOutputStream(
        writer.resource(IndexTransformer.INDEX_RESOURCE_NAME))

      try {
        vocabulary.save(fos)
      } finally {
        fos.close()
      }

      Some(JObject(List(JField("minimumObservations",
        JInt(vocabulary.minimumObservations)))))
    }

    /**
      * This function maps all features to unique indexes, or to a reserved
      * out-of-vocabulary value.
      *
      * It returns the size of the feature vocabulary and the associated index for
      * each input feature.
      *
      * @param features a list of features to transform
      * @return the current vocabulary size and the index of each feature
      */
    private[indexer] def mapFeaturesToIndices(features: Seq[Feature[_]]): (Int, Seq[Int]) = {
      val indices = features.map(f => vocabulary(f))
      (vocabulary.size, indices)
    }

    /**
      * This function creates the feature index map from all provided features.
      *
      * To be memory efficient and not create an array the size of the vocab, it:
      *  1) maps all the features to their index value and sorts them. O(n + n log(n))
      *  2) creates two arrays that will store the index and count pairs O(n)
      *  3) goes through the sorted list of indexes and sticks the appropriate values
      *       in the two arrays. O(n)
      *  4) it then creates a sparse vector, chopping off any trailing zeros in the two arrays. O(n)
      *
      * So for a total complexity of O(n log(n)) + 4 O(n) and space complexity of 5 O(n) where
      * <i>n</i> is the number of features passed in.
      *
      * @param features
      * @return feature vector representing all provided features
      */
    private[indexer] def getFeatureVector(features: Seq[Feature[_]]): Vector = {
      // Create our indexes and find out how large in theory the return vector should be
      val (vocabSize, indices) = mapFeaturesToIndices(features)
      // Map features to indexes & sort since they need to be in ascending order
      val indexValues = indices.sorted.toArray
      // Preallocate the arrays needed using their maximum possible size.
      val newIndexes = Array.fill[Int](indexValues.length)(0)
      val newValues = Array.fill[Double](indexValues.length)(0.0)
      var lastIndex = 0
      // skip OOV features
      var i = indexValues.indexWhere(_ > Vocabulary.OOV, 0)
      logger.debug(s"OOV/Pruned features were $i from ${indexValues.length}")
      var j = 0
      // traverse the sorted array of feature index values -- if i == -1 then we're all OOV so skip.
      while(i > -1 && i < indexValues.length) {
        // find next index whose value doesn't equal the current index value. j could equal -1.
        j = indexValues.indexWhere(_ != indexValues(i), i)
        // now we store the current index value
        newIndexes(lastIndex) = indexValues(i)
        // and how many times it occurred -- j could be -1 if we reached the end without a result
        newValues(lastIndex) = if (j > -1) { j - i } else {indexValues.length - i}
        // and we increment where we are in the output array
        lastIndex += 1
        // now move i to where ever j ended up else skip to the end
        i = if (j > -1) j else { indexValues.length }
      }
      // slicing allows us to make sure we chop off any trailing zeros
      Vectors.sparse(vocabSize, newIndexes.slice(0, lastIndex), newValues.slice(0, lastIndex))
    }

    def apply(features: Seq[Feature[_]]*): Vector = {
      val allFeatures = features.flatten
      if (allFeatures.length < 1)
        Vectors.zeros(vocabulary.size).toSparse
      else
        getFeatureVector(allFeatures)
    }

    def numDimensions = Some(vocabulary.size)

    def prune(transform: (Int) => Boolean) = vocabulary.prune(transform)

    /** Retrieves a feature from its index
      *
      * This can be used to implement model debugging features and
      * significant features. If no feature exists for the provided
      * index, returns None.
      */
    def getFeatureByIndex(i: Int): Option[Feature[_]] = vocabulary.invert(i)

    def freeze() = vocabulary.freeze
  }

  /** Paired loader class for IndexTransformer */
  class IndexTransformLoader extends ArchiveLoader[IndexTransformer] {

    /** Loads the IndexTransformer from an Alloy */
    def load(engine: Engine, reader: Option[Alloy.Reader],
      config: Option[JObject]): IndexTransformer = {

      val observations = config.map(_ \ "minimumObservations")
        .map(_.asInstanceOf[JInt].num.intValue)
        .getOrElse(0)

      val vocabulary = reader match {
        case None => {
          // when no reader exists, create an empty, mutable vocabulary
          new MutableVocabulary
        }
        case Some(reader) => {
          // otherwise, reload the vocabulary from the alloy
          val fis = new FeatureInputStream(
            reader.resource(IndexTransformer.INDEX_RESOURCE_NAME))
          try {
            Vocabulary.load(fis)
          } finally {
            fis.close()
          }
        }
      }

      vocabulary.minimumObservations = observations
      new IndexTransformer(vocabulary)
    }
  }

  private[this] object IndexTransformer {
    val INDEX_RESOURCE_NAME = "featureIndex"
  }
}
