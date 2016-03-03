package com.idibon.ml.predict

import com.idibon.ml.alloy.Codec
import com.idibon.ml.feature.{FeatureOutputStream, FeatureInputStream, Builder, Buildable, Feature}

/** Basic output result from a predictive model.
  *
  * Model predictions may return any number of PredictResult objects
  */
trait PredictResult {
  /** The label analyzed in this result */
  def label: String
  /** Evaluated probability */
  def probability: Float
  /** Number of features, rules, etc. considered in analysis */
  def matchCount: Int
  /** Bitmask of PredictResultFlag.Values identified for this result */
  def flags: Int
  /** Int (id) for type of prediction this came from (model, rule, combined)*/
  def modelType: PredictTypeFlag.Value

  /** True if the result is FORCED */
  def isForced = ((flags & (1 << PredictResultFlag.FORCED.id)) != 0)

  /**
    * Method to use to check whether two prediction results are close enough.
    *
    * @param other
    * @return
    */
  def isCloseEnough(other: PredictResult): Boolean = {
    this.label == other.label &&
      this.matchCount == other.matchCount &&
      this.flags == other.flags &&
      this.isForced == other.isForced &&
      this.modelType == other.modelType &&
      PredictResult.floatIsCloseEnough(this.probability, other.probability)
  }
}

object PredictResult {
  /** Our tolerance for numerical instability between machines **/
  val PRECISION = 0.001
  /**
    * Method specifically tasked with figuring out whether two float values
    * are within our tolerances for being "equal".
    *
    * @param thisProb
    * @param otherProb
    * @return
    */
  def floatIsCloseEnough(thisProb: Float, otherProb: Float): Boolean = {
    (thisProb - otherProb).abs < PRECISION
  }
}

/** Flags for special properties affecting the prediction result */
object PredictResultFlag extends Enumeration {
  /** The result was affected by (e.g.) a blacklist or whitelist rule */
  val FORCED = Value

  /** constant bitmask for no special flags */
  val NO_FLAGS = 0

  /** Compute a bitmask for a list of enabled result flags */
  def mask(flags: PredictResultFlag.Value*): Int = {
    flags.foldLeft(0)((m, f) => m | (1 << f.id))
  }
}

/** Flags for the type of prediction */
object PredictTypeFlag extends Enumeration {
  val MODEL,RULE,COMBINED=Value
}

/** Optional trait for PredictResult objects that report significant features */
trait HasSignificantFeatures {
  /** Extracted document features that significantly influence the result */
  def significantFeatures: Seq[(Feature[_], Float)]
}

/** Combines PredictResults for the same label across multiple models into
  * a final result.
  */
trait PredictResultReduction[T <: PredictResult] {

  /** Computes a single representative result from multiple partial results */
  def reduce(components: Seq[T]): T
}

/** Basic PredictResult for document classifications */
case class Classification(override val label: String,
                          override val probability: Float,
                          override val matchCount: Int,
                          override val flags: Int,
                          override val significantFeatures: Seq[(Feature[_], Float)],
                          override val modelType: PredictTypeFlag.Value)
  extends PredictResult with HasSignificantFeatures with Buildable[Classification, ClassificationBuilder]{
  /** Stores the data to an output stream so it may be reloaded later.
    *
    * @param output - Output stream to write data
    */
  override def save(output: FeatureOutputStream): Unit = {
    Codec.String.write(output, label)
    output.writeFloat(probability)
    Codec.VLuint.write(output, matchCount)
    Codec.VLuint.write(output, flags)
    Codec.VLuint.write(output, significantFeatures.size)
    Codec.String.write(output, modelType.toString())
    significantFeatures.foreach{case (feat, value) => {
      feat match {
        case f: Feature[_] with Buildable[_, _] => {
          output.writeFeature(f)
          output.writeFloat(value)
        }
        case _ => throw new RuntimeException("Got unsaveable feature.")
      }
    }}
  }
}

/**
  * Class for saving classification results to a stream.
  */
class ClassificationBuilder extends Builder[Classification] {
  /** Instantiates and loads an object from an input stream
    *
    * @param input - Data stream where object was previously saved
    */
  override def build(input: FeatureInputStream): Classification = {
    val label = Codec.String.read(input)
    val prob = input.readFloat()
    val matchCount = Codec.VLuint.read(input)
    val flags = Codec.VLuint.read(input)
    val sigFeatSize = Codec.VLuint.read(input)
    val modelType = Codec.String.read(input)

    val sigFeats = (0 until sigFeatSize).map(_ => {
      val feature = input.readFeature
      val value = input.readFloat()
      (feature, value)
    })
    new Classification(label, prob, matchCount, flags, sigFeats, PredictTypeFlag.withName(modelType))
  }
}

/** Companion class and reduction operation for Classification instances */
object Classification extends PredictResultReduction[Classification] {
  /** Performs a weighted-average of the partial Classifications
    *
    * If any FORCED results exist, only other results with the FORCED flag
    * are included in the reduced result.
    */
  def reduce(components: Seq[Classification]) = {
    components.filter(_.isForced) match {
      case Nil => {
        // no FORCED entries, just average over all items
        harmonic_mean(components, (c: Classification) => c.matchCount)
      }
      case forced => {
        // FORCED items exist, only average those
        harmonic_mean(forced, (c: Classification) => c.matchCount)
      }
    }
  }

  /** Performs a weighted-average to combine multiple classifications
    *
    * The weighting for each component is provided by a caller-provided
    * method. All partial results must be for the same label.
    */
  def average(components: Seq[Classification], fn: (Classification) => Int) = {
    val sumMatches = components.foldLeft(0)((sum, c) => sum + fn(c))

    // perform a weighted average of the prediction probability
    val probability = if (sumMatches <= 0) 0.0f else {
      components.foldLeft(0.0f)(
        (sum, c) => sum + fn(c) * c.probability) / sumMatches
    }

    /* return the union of enabled flags and significant features
     * across all partial models */
    Classification(components.head.label, probability, sumMatches,
      components.foldLeft(0)((mask, c) => mask | c.flags),
      components.flatMap(_.significantFeatures), PredictTypeFlag.COMBINED)
  }

  /** Calculates a harmonic mean of averages of classifications
    *
    * The averages are calculated for each PredictType of classification,
    * then the harmonic mean is taken of those results to get the final
    * combined probability.
    */
  def harmonic_mean(components: Seq[Classification], fn: (Classification) => Int) = {
    if (components.tail.exists(_.label != components.head.label))
      throw new IllegalArgumentException("can not combine across labels")

    //1. separate the components base on type (no reason to keep the type)
    val separated = components.groupBy(c => c.modelType).values

    //2. get the averages of each type
    val weighted_probabilities = separated.map(x => {
     average(x, fn)
    })

    //3. return the harmonic mean of this set of averages
    //h = n / (∑ 1/m)
    val harmonic_mean = divide_or_zero(weighted_probabilities.size,
      weighted_probabilities.foldLeft(0.0f)((sum, p) => sum + divide_or_zero(1.0f, p.probability)))

    /*return the union of enabled flags and significant features
    * across all partial models */
    Classification(components.head.label, harmonic_mean,
      weighted_probabilities.foldLeft(0)((sum, c) => sum + fn(c)),
      components.foldLeft(0)((mask, c) => mask | c.flags),
      components.flatMap(_.significantFeatures),PredictTypeFlag.COMBINED)
  }

  //Divide n/d, return zero if d <= 0
  def divide_or_zero(n: Float, d: Float): Float = {
    if (d <= 0) 0.0f else n/d
  }
}
