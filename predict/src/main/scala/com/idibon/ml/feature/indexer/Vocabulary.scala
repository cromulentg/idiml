package com.idibon.ml.feature.indexer

import scala.collection.mutable

import com.idibon.ml.feature._
import com.idibon.ml.alloy.Codec

import com.typesafe.scalalogging.StrictLogging

/** Maps a set of known Features to indexed dimensions within a domain
  *
  * Out-of-domain Features are mapped to the special Vocabulary.OOV
  * value. Some implementations support dynamic assigment of dimensions
  * to of out-of-domain Features.
  */
private[indexer] trait Vocabulary
    extends Function1[Feature[_], Int]
    with Freezable[Vocabulary] {
  /** Returns the dimension index for the feature, if assigned
    *
    * If the feature is known to the vocabulary, this method returns
    * the dimension index of the feature. If not, returns the special
    * value Vocabulary.OOV
    *
    * @param f feature
    * @return dimension index, or Vocabulary.OOV
    */
  def apply(f: Feature[_]): Int

  /** Given a dimension index, returns the corresponding feature
    *
    * If the dimension is assigned, returns the feature that has that
    * index. If the dimension is unassigned or invalid, returns None
    *
    * @param i dimension index
    * @return Some(Feature) if the index is assigned, else None
    */
  def invert(i: Int): Option[Feature[_]]

  /** Saves the Vocabulary to an output stream
    *
    * @param os output stream
    */
  def save(os: FeatureOutputStream): Unit

  /** Removes Features from the vocabulary using a predicate function
    *
    * Removes any feature from the vocabulary where the provided predicate
    * function returns true when called using the feature's dimension
    *
    * @param predicate function returning true for any feature to prune
    */
  def prune(predicate: (Int) => Boolean): Unit

  /** Freezes the vocabulary, preventing new features from being added
    *
    * Frozen vocabularies may be pruned; however, any feature not already
    * included in the Vocabulary (or any feature removed from a call to
    * prune) will be treated as out-of-vocabulary.
    */
  def freeze: Vocabulary

  /** Total size in dimensions of the vocabulary. */
  def size: Int

  /** Total assigned dimensions in the vocabulary. */
  def assigned: Int

  /** The minimum number of times a feature must be observed
    *
    * If the Vocabulary is not frozen, any feature observed at least
    * minimumObservations times may be added to the domain. Features
    * observed fewer than this threshold are treated as out-of-vocabulary
    */
  def minimumObservations: Int
  def minimumObservations_=(i: Int): Unit
}

private[indexer] object Vocabulary extends StrictLogging {
  /** Reserved value for out-of-vocabulary features */
  val OOV: Int = -1

  /** Loads a Vocabulary from the provided input stream
    *
    * @param is input stream
    */
  def load(is: FeatureInputStream): Vocabulary = {
    val frozen = is.readBoolean
    val size = Codec.VLuint.read(is)
    val savedFeatures = Codec.VLuint.read(is)

    logger.trace(s"Loading Vocabulary (dim=$size, feat=$savedFeatures)")

    var indexValue = 0
    val index: Map[Feature[_], Int] = (1 to savedFeatures).map(_ => {
      val feature = is.readFeature()
      val delta = Codec.VLuint.read(is)
      indexValue += delta
      (feature -> indexValue)
    }).toMap

    if (frozen)
      new ImmutableVocabulary(index, size)
    else
      new MutableVocabulary(index, size)
  }

  /** Saves a vocabulary to an output stream
    *
    * @param os output stream
    * @param frozen if the vocabulary should be frozen when it is re-loaded
    * @param size total dimensions in the vocabulary
    * @param domain the mapping of features to dimension indices
    */
  def save(os: FeatureOutputStream, frozen: Boolean,
    size: Int, domain: collection.Map[Feature[_], Int]) {

    os.writeBoolean(frozen)
    /* save the dimensionality of the feature index, so that if frozen,
     * we can create properly sized vectors */
    Codec.VLuint.write(os, size)

    /* sort the features in order of increasing index, so that we can
     * delta-encode the indices in the file to take advantage of space
     * savings from the VLuint data type. if any non-buildable features
     * are in the index, log an error since these can't be saved */
    val sortedFeatures = domain.toSeq
      .filter(_._1.isInstanceOf[Buildable[_, _]])
      .sortBy(_._2)

    if (sortedFeatures.size != domain.size)
      logger.error(s"Vocabulary has ${domain.size - sortedFeatures.size} unsaveable dimensions")

    Codec.VLuint.write(os, sortedFeatures.size)
    var lastIndex = 0
    sortedFeatures.foreach({ case (feature, index) => {
      os.writeFeature(feature.asInstanceOf[Feature[_] with Buildable[_, _]])
      Codec.VLuint.write(os, index - lastIndex)
      lastIndex = index
    }})
  }
}

/** Immutable vocabulary
  *
  * All features must be known when the vocabulary is instantiated, or
  * will be treated as out-of-vocabulary
  */
private[indexer] class ImmutableVocabulary(
  private[indexer] val domain: Map[Feature[_], Int],
  override val size: Int) extends Vocabulary {

  // inverted index - stores a mapping of index to feature
  private[this] val inverse: Map[Int, Feature[_]] = {
    domain.map({ case (f, i) => (i -> f) }).toMap
  }

  /** Returns number of assigned dimensions in the Vocabulary. */
  def assigned = domain.size

  /** Returns the dimension index for the feature, or OOV
    *
    * @param f feature
    * @return assigned dimension index for the feature
    */
  def apply(f: Feature[_]) = domain.getOrElse(f, Vocabulary.OOV)

  /** Returns the Feature corresponding to a dimension index, if one exists
    *
    * @param i dimension index
    * @return Some(Feature) if the feature is defined, else None
    */
  def invert(i: Int) = inverse.get(i)

  /** Saves the vocabulary to the output stream
    *
    * @param os output stream
    */
  def save(os: FeatureOutputStream) {
    Vocabulary.save(os, true, size, domain)
  }

  /** Compares if two Vocabularies are equal
    *
    * Vocabularies are considered equal if they transform the same domain of
    * Features to the same dimensions, and are either both mutable or both
    * immutable / frozen.
    *
    * Not consistent with hashCode, intended for unit tests
    */
  override def equals(o: Any): Boolean = o match {
    case x: AnyRef if this.eq(x) => true
    case v: ImmutableVocabulary => {
      size == v.size &&
      assigned == v.assigned &&
      domain == v.domain
    }
    case v: MutableVocabulary => {
      v.synchronized {
        size == v.size &&
        assigned == v.assigned &&
        v.frozen &&
        domain == v.domain
      }
    }
    case _ => false
  }

  /** No-op methods because the vocaubulary is immutable */
  def freeze: Vocabulary = this
  def prune(p: (Int) => Boolean) {}
  def minimumObservations = 0
  def minimumObservations_=(o: Int) {}
}

/** Thread-safe, mutable vocabulary
  *
  * When unfrozen, new dimensions are added as-needed to accommodate out-
  * of-vocabulary features. When frozen, all new features are treated as
  * out-of-vocabulary.
  *
  * All methods are thread-safe.
  */
private[indexer] class MutableVocabulary extends Vocabulary {

  // stores assigned dimensions
  private[indexer] val domain = mutable.Map[Feature[_], Int]()
  private[this] val inverse = mutable.Map[Int, Feature[_]]()
  private[indexer] var frozen = false

  // stores observation counts for not-yet-in-vocabulary features
  private[indexer] val observations = mutable.Map[Feature[_], Int]()
  private[this] var _size = 0

  /** Constructor for reloading mutable vocabularies from alloys */
  private[indexer] def this(m: Map[Feature[_], Int], s: Int) {
    this()
    m.foreach(entry => domain += entry)
    m.foreach({case (feature, index) => inverse.put(index, feature)})
    _size = s
  }

  override var minimumObservations = 0

  /** Returns the number of dimensions in the vocabulary */
  def size = this.synchronized { _size }

  /** Returns the number of assigned dimensions in the vocabulary */
  def assigned = this.synchronized { domain.size }

  /** Returns the Feature corresponding to a dimension index, if one exists
    *
    * @param i dimension index
    * @return Some(Feature) if the feature is defined, else None
    */
  def invert(i: Int): Option[Feature[_]] = inverse.get(i)

  /** Returns the dimension index for the feature, or OOV
    *
    * @param f feature
    * @return assigned dimension index for the feature
    */
  def apply(f: Feature[_]): Int = {
    this.synchronized {
      domain.getOrElse(f, {
        if (frozen) {
          // new features aren't added to frozen vocabularies
          Vocabulary.OOV
        } else {
          /* if the feature has been observed at least minimumObservations
           * times, add a new dimension to the domain and to the inverted
           * index */
          val count = observations.getOrElse(f, 0) + 1
          if (count >= minimumObservations) {
            observations -= f
            domain += (f -> _size)
            inverse += (_size -> f)
            _size += 1
            domain(f)
          } else {
            observations += (f -> count)
            Vocabulary.OOV
          }
        }
      })
    }
  }

  /** Saves the vocabulary to the output stream
    *
    * @param os output stream
    */
  def save(os: FeatureOutputStream) {
    this.synchronized {
      Vocabulary.save(os, frozen, _size, domain)
    }
  }

  /** Compares if two Vocabularies are equal
    *
    * Vocabularies are considered equal if they transform the same domain of
    * Features to the same dimensions, and are either both mutable or both
    * immutable / frozen.
    *
    * Not consistent with hashCode, intended for unit tests
    */
  override def equals(o: Any): Boolean = o match {
    case x: AnyRef if this.eq(x) => true
    case v: ImmutableVocabulary => {
      size == v.size &&
      assigned == v.assigned &&
      domain == v.domain
    }
    case v: MutableVocabulary => {
      v.synchronized {
        _size == v.size &&
        assigned == v.assigned &&
        frozen == v.frozen &&
        domain == v.domain &&
        observations == v.observations
      }
    }
    case _ => false
  }

  /** Removes all features from the Vocabulary where the predicate returns true
    *
    * @param predicate predicate function returning true for dimensions which
    *   should be pruned
    */
  def prune(predicate: (Int) => Boolean) {
    this.synchronized {
      /* construct a list of all feature/index pairs that must be pruned,
       * then prune them */
      domain.filter(e => predicate(e._2))
        .foreach({ case (feature, index) => {
          domain -= feature
          inverse -= index
        }})
    }
  }

  /** Prevents new features from growing the domain */
  def freeze: Vocabulary = {
    this.synchronized {
      val newMutableVocab = new MutableVocabulary(domain.toMap, domain.size)
      newMutableVocab.frozen = true
      newMutableVocab
    }
  }
}
