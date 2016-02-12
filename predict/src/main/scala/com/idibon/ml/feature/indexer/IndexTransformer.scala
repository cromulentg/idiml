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
  class IndexTransformer(minimumObservations: Int,
    private var frozen: Boolean = false,
    private var frozenSize: Int = 0) extends FeatureTransformer
      with Archivable[IndexTransformer, IndexTransformLoader]
      with TerminableTransformer
      with StrictLogging {

    logger.trace(s"event=new;minimumObservations=$minimumObservations,frozen=$frozen,frozenSize=$frozenSize")

    private[indexer] val featureIndex = scala.collection.mutable.Map[Feature[_], Int]()
    private[indexer] val observations = scala.collection.mutable.Map[Feature[_], Int]()
    private[indexer] val inverseIndex = scala.collection.mutable.Map[Int, Feature[_]]()

    def getFeatureIndex = featureIndex

    def save(writer: Alloy.Writer) = {
      val fos = new FeatureOutputStream(
        writer.resource(IndexTransformer.INDEX_RESOURCE_NAME))
      try {
        // write boolean for frozen
        fos.writeBoolean(frozen)
        /* save the dimensionality of the feature index, so that if frozen,
         * we can create properly sized vectors */
        Codec.VLuint.write(fos, frozenSize)
        /* sort the features in order of increasing index, so that we can
         * delta-encode the indices in the file to take advantage of space
         * savings from the VLuint data type. if any non-buildable features
         * are in the index, log an error since these can't be saved */
        val vocabulary = featureIndex.toSeq
          .filter(_._1.isInstanceOf[Buildable[_, _]])
          .map({ case (f, i) => (f.asInstanceOf[Feature[_] with Buildable[_, _]], i) })
          .sortWith(_._2 < _._2)

        val unsaveable = featureIndex.keys
          .filter(!_.isInstanceOf[Buildable[_, _]])
          .map(_.getClass).toList
          .distinct

        if (!unsaveable.isEmpty)
          logger.error(s"Unable to save features: ${unsaveable.mkString(", ")}")

        /* Save the size of the vocabulary so we know how many items to read
         * at load time */
        Codec.VLuint.write(fos, vocabulary.size)
        /* store the difference in index value between adjacent indices in the
         * vocabulary, to bias stored values to smaller values that take better
         * advantage of the VLuint encoding */
        var lastIndex = 0
        vocabulary.foreach({ case (feature, index) => {
          fos.writeFeature(feature)
          Codec.VLuint.write(fos, index - lastIndex)
          lastIndex = index
        }})
      } finally {
        fos.close()
      }

      Some(JObject(List(JField("minimumObservations", JInt(minimumObservations)))))
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

      if (frozen) {
        (numDimensions.get, features.map(f => featureIndex.getOrElse(f, -1)))
      } else {
        // perform all possibly-mutative operations inside a critical section
        featureIndex.synchronized {
          /* NB: if a feature transitions from "too few observations" to
           * "just enough observations" within this loop, we may return
           * different indices for the same feature in the sequence. this
           * should be fine, since observation thresholding is a priming-time
           * operation, anyway */
          val indices = features.map(f => {
            if (!featureIndex.contains(f)) {
              val observed = observations.get(f).getOrElse(0) + 1
              if (observed >= minimumObservations) {
                /* the feature has appeared enough times to add to the vocabulary.
                 * look deep within its soul and assign it a name based on the order
                 * in which it joined */
                featureIndex += (f -> featureIndex.size)
                // no need to remember how many times it's been observed now
                observations.remove(f)
              } else {
                observations += (f -> observed)
              }
            }
            featureIndex.getOrElse(f, -1)
          })
          (numDimensions.get, indices)
        }
      }
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
      var i = indexValues.indexWhere(_ > -1, 0)
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
        Vectors.zeros(numDimensions.get).toSparse
      else
        getFeatureVector(allFeatures)
    }

    def numDimensions = if (frozen) Some(frozenSize) else Some(featureIndex.size)

    def prune(transform: (Int) => Boolean): Unit = {
      featureIndex.map(x => {
        if (transform(x._2)) featureIndex.remove(x._1)
      })
    }

    /** Retrieves a feature from its index
      *
      * This can be used to implement model debugging features and
      * significant features. If no feature exists for the provided
      * index, returns None.
      */
    def getFeatureByIndex(i: Int): Option[Feature[_]] = {
      if (inverseIndex.isEmpty)
        featureIndex.find(_._2 == i).map(_._1)
      else
        inverseIndex.get(i)
    }

    def freeze(): Unit = {
      if (!frozen) {
        featureIndex.synchronized {
          frozenSize = featureIndex.size
          observations.clear
          frozen = true
        }
      }
    }
  }

  /** Paired loader class for IndexTransformer */
  class IndexTransformLoader extends ArchiveLoader[IndexTransformer] {

    /** Loads the IndexTransformer from an Alloy */
    def load(engine: Engine, reader: Option[Alloy.Reader], config: Option[JObject]): IndexTransformer = {
      val observations = config.map(_ \ "minimumObservations")
        .map(_.asInstanceOf[JInt].num.intValue)
        .getOrElse(0)

      reader match {
        case None => new IndexTransformer(observations)
        case Some(reader) => {
          val fis = new FeatureInputStream(
            reader.resource(IndexTransformer.INDEX_RESOURCE_NAME))

          // read boolean for frozen
          val frozen = fis.readBoolean()
          // read int for frozen size
          val frozenSize = Codec.VLuint.read(fis)

          // Retrieve the number of elements in the featureIndex map
          val size = Codec.VLuint.read(fis)

          val transformer = new IndexTransformer(observations, frozen, frozenSize)

          var indexValue = 0
          1 to size foreach { _ =>
            val feature = fis.readFeature
            val delta = Codec.VLuint.read(fis)
            indexValue += delta
            transformer.featureIndex += (feature -> indexValue)
            transformer.inverseIndex += (indexValue -> feature)
          }
          transformer
        }
      }
    }
  }

  private[this] object IndexTransformer {
    val INDEX_RESOURCE_NAME = "featureIndex"
  }
}
