package com.idibon.ml.feature.indexer

import com.idibon.ml.alloy.Codec
import com.idibon.ml.feature.{Freezable, FeatureInputStream, FeatureOutputStream}

import scala.collection.mutable.ArrayBuffer

/**
  * Trait to hold behaviors related to calculating Inverse Document Frequency (IDF).
  *
  * See http://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html.
  *
  * IDF helps to give a "high" value to things that occur rarely across the entire corpus,
  * and a "low" value to things that occur frequently across the corpus.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>" on 4/8/16.
  */
trait IDFCalculator extends Freezable[IDFCalculator]{
  /** The minimum number of times a feature must be observed in individual documents.
    *
    * If the IDFCalculator is not frozen, any feature observed at least
    * minimumDocumentObservations times may have a non-zero IDF value. Features
    * observed fewer than this threshold are treated as having value 0.0.
    */
  def minimumDocumentObservations: Int
  def minimumDocumentObservations_=(i: Int): Unit

  /** Increments the total number of unique documents seen **/
  def incrementTotalDocCount(): Unit

  /** Method to increment seen counts of a bunch of feature dimensions
    *
    * @param dimensions
    */
  def incrementSeenCount(dimensions: Array[Int]): Unit

  /** Method to calculate inverse document frequency for a dimension (feature).
    *
    * @param dimension the feature to get IDF for
    * @return IDF value or 0.0 if it was pruned/OOV.
    */
  def inverseDocumentFrequency(dimension: Int): Double

  /** Method to calculate inverse document frequency dimensions (features).
    *
    * @param dimensions the features to get IDF values for
    * @return IDF values, where 0.0 means it was pruned/OOV.
    */
  def inverseDocumentFrequency(dimensions: Seq[Int]): Seq[Double]
  def freeze: IDFCalculator
  def prune(predicate: (Int) => Boolean): Unit
  def save(os: FeatureOutputStream): Unit
  def size: Int
}

object IDFCalculator {
  /** Loads an IDF calculator from a feature input stream.
    *
    * Uses the boolean to determine whether to create an
    * immutable or mutable IDFCalculator.
    *
    * @param is stream to read from.
    * @return an IDF Calculator.
    */
  def load(is: FeatureInputStream): IDFCalculator = {
    val isFrozen = is.readBoolean()
    if (isFrozen) {
      val numEntries = Codec.VLuint.read(is)
      val idfValues = (0 until numEntries).map(_ => {
        Codec.VLuint.read(is) -> is.readDouble()
      }).toMap
      new ImmutableIDFCalculator(idfValues)
    } else {
      val numDocs = Codec.VLuint.read(is)
      val numDimensions = Codec.VLuint.read(is)
      val counter = new ArrayBuffer[Int](numDimensions)
      (0 until numDimensions).foreach(i => counter += Codec.VLuint.read(is))
      new MutableIDFCalculator(counter, numDocs)
    }
  }

  /** Saves a MutableIDFCalculator to a stream.
    *
    * @param os stream to write to.
    * @param numDocs number of documents in the corpus.
    * @param counter number of occurrences for each dimension across the corpus.
    */
  def save(os: FeatureOutputStream,
           numDocs: Int,
           counter: ArrayBuffer[Int]): Unit = {
    os.writeBoolean(false)
    Codec.VLuint.write(os, numDocs)
    Codec.VLuint.write(os, counter.size)
    counter.foreach({count =>
      Codec.VLuint.write(os, count)
    })
  }

  /** Saves an ImmutableIDFCalculator to a stream.
    *
    * @param os stream to write to.
    * @param idfValues map of dimension -> idf value.
    */
  def save(os: FeatureOutputStream,
           idfValues: Map[Int, Double]): Unit = {
    os.writeBoolean(true)
    Codec.VLuint.write(os, idfValues.size)
    idfValues.foreach({case (dimension, idfValue) =>
      Codec.VLuint.write(os, dimension)
      os.writeDouble(idfValue)
    })
  }

  /**
    * Computes the inverse document frequency for each term (dimension).
    * IDF_t = log ( N / df_t) ==> log(N) - log(df_t)
    *
    * @return dimension -> idf value.
    */
  def computeIDFs(numDocs: Int,
                  counts: IndexedSeq[Int],
                  minDocCount: Int): Map[Int, Double] = {
    val numerator = Math.log(numDocs)
    val minFilterValue = Math.max(minDocCount, 1)
    counts.zipWithIndex
      .filter({case (count, dimension) => count >= minFilterValue})
      .map({ case (count, dimension) => dimension -> (numerator - Math.log(count))})
      .toMap
  }
}

/**
  * Immutable Version that just has already precomputed IDF values for feature dimensions.
  *
  * @param idfValues dimension -> idf value.
  */
case class ImmutableIDFCalculator(idfValues: Map[Int, Double]) extends IDFCalculator {

  def inverseDocumentFrequency(dimension: Int): Double = {
    idfValues.getOrElse(dimension, 0.0)
  }

  def inverseDocumentFrequency(dimension: Seq[Int]): Seq[Double] = {
    dimension.map(inverseDocumentFrequency(_))
  }
  def save(os: FeatureOutputStream): Unit = {
    IDFCalculator.save(os, idfValues)
  }
  def incrementTotalDocCount(): Unit = {}
  def incrementSeenCount(dimensions: Array[Int]): Unit = {}
  def freeze: IDFCalculator = this
  def prune(predicate: (Int) => Boolean): Unit = {}
  def size: Int = idfValues.size
  override def minimumDocumentObservations: Int = 0
  override def minimumDocumentObservations_=(i: Int): Unit = {}
}

/**
  * Mutable version that handles counting, growing and pruning related to
  * being able to calculate IDF values for a dimension.
  */
case class MutableIDFCalculator(private[indexer] val counter: ArrayBuffer[Int] = new ArrayBuffer[Int],
                                var numDocs: Int = 0) extends IDFCalculator {
  override var minimumDocumentObservations = 0
  var frozen: Boolean = false

  def size: Int = counter.size

  /**
    * Increments the total number of documents seen.
    */
  def incrementTotalDocCount(): Unit = {
    if (!frozen) {
      numDocs += 1
    }
  }

  /**
    * Calculates the IDF value of passed in dimension.
    *
    * Uses the current state to calculate the value.
    *
    * NOTE: not thread safe.
    *
    * @param dimension the feature to get IDF for
    * @return IDF value or 0.0 if it was pruned/OOV.
    */
  def inverseDocumentFrequency(dimension: Int): Double = {
    if (dimension < counter.size && counter(dimension) > 0) {
      Math.log(numDocs) - Math.log(counter(dimension))
    } else {
      0.0
    }
  }

  /**
    * Calculates the IDF values of the passed in dimensions.
    *
    * Uses the current state to calculate the value.
    *
    * NOTE: not thread safe.
    *
    * @param dimensions the features to get IDF values for
    * @return IDF values, where 0.0 means it was pruned/OOV.
    */
  def inverseDocumentFrequency(dimensions: Seq[Int]): Seq[Double] = {
    dimensions.map(d => inverseDocumentFrequency(d))
  }

  /**
    * Increments the document term count for passed in dimension.
    *
    * @param dimension dimension to increment document term counts for.
    */
  def incrementSeenCount(dimension: Int): Unit = {
    if (!frozen) {
      this.synchronized {
        if (dimension < counter.size) {
          counter(dimension) += 1
        } else {
          assert(dimension == counter.length, "Why are we trying to append some wild dimension value?")
          counter += 1
        }
      }
    }
  }

  /**
    * Increments the document term counts for passed in dimensions.
    *
    * @param dimensions dimensions to increment document term counts for.
    */
  def incrementSeenCount(dimensions: Array[Int]): Unit = {
    dimensions.foreach(d => incrementSeenCount(d))
  }

  /**
    * Creates an immutable IDF calculator from this mutable one.
    *
    * @return Immutable IDFCalculator
    */
  def freeze: IDFCalculator = {
    this.synchronized {
      val m = new MutableIDFCalculator(counter.clone(), numDocs)
      m.frozen = true
      m
    }
  }

  /** Zeros all document counts for features where the predicate returns true.
    *
    * @param predicate predicate function returning true for dimensions which
    *   should be pruned
    */
  def prune(predicate: (Int) => Boolean): Unit = {
    this.synchronized {
      counter.indices.foreach(i => {
        if (predicate(i)) counter(i) = 0
      })
    }
  }

  /**
    * Saves Mutable IDF Calculator to a stream.
    *
    * @param os stream to write to.
    */
  def save(os: FeatureOutputStream): Unit = {
    this.synchronized {
      if (frozen) {
        IDFCalculator.save(os, IDFCalculator.computeIDFs(numDocs, counter, minimumDocumentObservations))
      } else {
        IDFCalculator.save(os, numDocs, counter)
      }
    }
  }
}
