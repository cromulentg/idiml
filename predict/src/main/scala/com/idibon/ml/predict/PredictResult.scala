package com.idibon.ml.predict

import com.idibon.ml.alloy.Codec
import com.idibon.ml.feature.tokenizer.Token
import com.idibon.ml.feature.{FeatureOutputStream, FeatureInputStream, Builder, Buildable, Feature}
import com.idibon.ml.predict.crf.{BIOType}

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

  /** True if the result is FORCED */
  def isForced = ((flags & (1 << PredictResultFlag.FORCED.id)) != 0)
  def isRule = ((flags & (1 << PredictResultFlag.RULE.id)) != 0)

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
  /** Forced: the result was affected by (e.g.) a blacklist or whitelist rule */
  /** Rule: this is the result of a rule, not an ML model */
  val FORCED,RULE = Value

  /** constant bitmask for no special flags */
  val NO_FLAGS = 0

  /** Compute a bitmask for a list of enabled result flags */
  def mask(flags: PredictResultFlag.Value*): Int = {
    flags.foldLeft(0)((m, f) => m | (1 << f.id))
  }
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
                          override val significantFeatures: Seq[(Feature[_], Float)])
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

/** Trait for predictive results which only affect a region of the document */
trait HasRegion {
  /** Start of the affected region, in UTF-16 code units. */
  val offset: Int
  /** Length of the affected region, in UTF-16 code units */
  val length: Int
  /** End of the affected region, in UTF-16 code units */
  final val end = offset + length
}

/** Trait for predictive results that are over a set of tokens in a document. */
trait HasTokens {
  /** The sequence of tokens in order */
  val tokens: Seq[Token]

  /**
    * Helper method to save the tokens to an output stream.
    * @param output
    */
  protected def saveTokens(output: FeatureOutputStream) = {
    Codec.VLuint.write(output, tokens.size)
    tokens.foreach(tok => tok.save(output))
  }
}
/** Companion object to store method for loading tokens */
object HasTokens {
  /**
    * Helper method to load tokens.
    * @param input
    * @return
    */
  def loadTokens(input: FeatureInputStream): Seq[Token] = {
    val tokensSize = Codec.VLuint.read(input)
    (0 until tokensSize).map(_ => {
      input.readFeature().asInstanceOf[Token]
    })
  }
}
/** Trait for predictive results that are over a set of token tags in a document. */
trait HasTokenTags {
  /** The sequence of tags in order that would be given to tokens */
  val tags: Seq[BIOType.Value]

  /**
    * Helper method to save token tags to a stream
    * @param output
    */
  protected def saveTokenTags(output: FeatureOutputStream) = {
    Codec.VLuint.write(output, tags.size)
    tags.foreach(tag => Codec.String.write(output, tag.toString))
  }
}
/** Companion object to store method for loading token tags */
object HasTokenTags {
  /**
    * Helper method to load token tags from input stream.
    * @param input
    * @return
    */
  def loadTokenTags(input: FeatureInputStream): Seq[BIOType.Value] = {
    val tagsSize = Codec.VLuint.read(input)
    (0 until tagsSize).map(_ => {
      BIOType.withName(Codec.String.read(input))
    })
  }
}

/** Basic PredictResult for span extraction
  *
  * @param label assigned label name
  * @param probability predictive confidence
  * @param flags prediction flags
  * @param offset start of the span, in UTF-16 code units
  * @param length length of the span, in UTF-16 code units
  * @param tokens sequence of tokens this span represents.
  * @param tags sequence of tags that would map to the tokens produced.
  */
case class Span(override val label: String,
  override val probability: Float, override val flags: Int,
  val offset: Int, val length: Int,
  tokens: Seq[Token] = Seq(),
  tags: Seq[BIOType.Value] = Seq())
    extends PredictResult with HasRegion with HasTokens with HasTokenTags
    with Buildable[Span, SpanBuilder] {

  override val matchCount = 1

  def save(output: FeatureOutputStream) {
    Codec.String.write(output, label)
    output.writeFloat(probability)
    Codec.VLuint.write(output, flags)
    Codec.VLuint.write(output, offset)
    Codec.VLuint.write(output, length)
    saveTokens(output)
    saveTokenTags(output)
  }
}

/** Paired builder class for Classification */
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
    val sigFeats = (0 until sigFeatSize).map(_ => {
      val feature = input.readFeature
      val value = input.readFloat()
      (feature, value)
    })
    new Classification(label, prob, matchCount, flags, sigFeats)
  }
}

/** Paired builder class for Span */
class SpanBuilder extends Builder[Span] {

  /** Read and instantiate a Span from the input stream
    *
    * @param input Data stream where object was previously saved
    */
  def build(input: FeatureInputStream): Span = {
    val label = Codec.String.read(input)
    val prob = input.readFloat()
    val flags = Codec.VLuint.read(input)
    val offset = Codec.VLuint.read(input)
    val length = Codec.VLuint.read(input)
    val tokens = HasTokens.loadTokens(input)
    val tokenTags = HasTokenTags.loadTokenTags(input)
    Span(label, prob, flags, offset, length, tokens, tokenTags)
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
        average(components, (c: Classification) => c.matchCount)
      }
      case forced => {
        // FORCED items exist, only average those
        average(forced, (c: Classification) => c.matchCount)
      }
    }
  }
  /** Performs a weighted-average to combine multiple classifications
    *
    * The weighting for each component is provided by a caller-provided
    * method. All partial results must be for the same label.
    */
  def weighted_average(components: Seq[Classification], fn: (Classification) => Int) = {
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
      components.flatMap(_.significantFeatures))
  }
  /** Calculates the average of multiple classifications grouped by
    * model or rule.
    *
    * The weighted averages of the two groups are calculated, then
    * combined with a harmonic mean to get a true average.
    */
  def average(components: Seq[Classification], fn: (Classification) => Int) = {
    if (components.tail.exists(_.label != components.head.label))
      throw new IllegalArgumentException("can not combine across labels")

    //1. separate the components into rules and not rules
    val models_and_rules = components.groupBy(c => c.isRule).values

    //2. get the averages of each type
    val weighted_probabilities = models_and_rules.map(x => {
      weighted_average(x, fn)
    })

    val n = weighted_probabilities.size

    //all rules, or all ml models
    if (n == 1) {
      Classification(components.head.label, weighted_probabilities.head.probability,
        components.foldLeft(0)((sum, c) => sum + fn(c)),
        components.foldLeft(0)((mask, c) => mask | c.flags),
        components.flatMap(_.significantFeatures))
    } else {
      //harmonic mean of two values = 2*(a*b)/a+b
      val harmonic_mean = 2.0f * divide_or_zero(
        weighted_probabilities.foldLeft(1.0f)((prod, c) => prod * c.probability),
        weighted_probabilities.foldLeft(0.0f)((sum, c) => sum + c.probability))

      Classification(components.head.label, harmonic_mean,
        components.foldLeft(0)((sum, c) => sum + fn(c)),
        components.foldLeft(0)((mask, c) => mask | c.flags),
        components.flatMap(_.significantFeatures))
    }
  }
  /** Divide n/d, return zero if d <= 0 */
  def divide_or_zero(n: Float, d: Float): Float = {
    if (d <= 0) 0.0f else n/d
  }
}
