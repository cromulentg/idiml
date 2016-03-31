package com.idibon.ml.predict

import com.idibon.ml.alloy.Codec
import com.idibon.ml.feature.tokenizer.Token
import com.idibon.ml.feature.{FeatureOutputStream, FeatureInputStream, Builder, Buildable, Feature}
import com.idibon.ml.predict.crf.{BIOType}

import scala.collection.mutable

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
  /** End of the affected region, in UTF-16 code units.
    * i.e. think of it as [offset, end), where end is exclusive of the span. */
  final val end = offset + length
}

/** Trait for predictive results that are over a set of tokens in a document. */
trait HasTokens {
  /** The sequence of tokens in order */
  val tokens: Seq[Token]

  /**
    * Helper method to save the tokens to an output stream.
    *
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
    *
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
    *
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
    *
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

  /** Helper method to return tokens and their tags together **/
  def getTokenNTags(): Seq[(Token, BIOType.Value)] = {
    tokens.zip(tags)
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

/** Companion class and reduction operation for Classification instances */
object Span {
  /** used when comparing how close some numbers are to each other **/
  val ZERO_TOLERANCE: Float = 0.0000003f
  /**
    * Computes a valid set of spans from a contiguous set of overlapping spans.
    *
    * We greedily go from left to right pick the best span.
    *
    * Important to understand: we assume the spans are a contiguous set that overlap,
    * and we want to decide which one or set of them to return.
    * E.g.
    *  1) A-B-C-D
    *  2)   B-C
    *  3)       D-E
    *  4)         E-F-G
    *  5)             G-H-I
    *
    *
    * @param components assumes they are sorted by offset and if at the same offset,
    *                   sorted by decreasing length (so longest first if there is an
    *                   offset tie)
    * @return subset of spans, where none of them overlap
    */
  def greedyReduce(components: Seq[Span], spanChooser: (Span, Seq[Span]) => Span): Seq[Span] = {
    if (components.size <= 1) components
    else {
      val mutableList = mutable.ListBuffer[Span]()
      var head = components.head
      var workspace = components.tail
      /*
       We have a contiguous overlapping set of spans. Our desire is to return a subset of them.
       Perform a greedy reduction:
       1) Grab the head
       2) Get overlapping spans with the head
       3) One by one, choose a strategy to merge/choose the span.
       4) Make head the recently merged span, and advance to next unseen span. Repeat #2-4
       */
      while (workspace.nonEmpty) {
        val overlapping = getOverlappingSpans(head, workspace)
        // if no overlapping spans -- move along
        if (overlapping.isEmpty) {
          mutableList += head
          head = workspace.head
          workspace = workspace.tail
        } else {
          // choose span from overlapping & move workspace along
          head = spanChooser(head, overlapping)
          workspace = workspace.slice(overlapping.size, workspace.length)
        }
      }
      mutableList += head
      mutableList.toSeq
    }
  }

  /**
    * Gets spans that overlap with this particular span.
    *
    * Assumes that the spans passed in are sorted by offset, and that the start
    * span's offset is <= the first span in the spans sequence.
    *
    * @param start span to start at, offset needs to be <= first offset in spans.
    * @param spans sorted list of spans by offset.
    * @return a sequence of spans that overlap with the start span.
    */
  def getOverlappingSpans(start: Span, spans: Seq[Span]): Seq[Span] = {
    spans.takeWhile(span => {
      start.offset <= span.offset && span.offset < start.end
    })
  }

  /**
    * Chooses the span to represent a set of overlapping spans.
    *
    * Takes in ML produced spans and rule produced spans that overlap and returns a single span.
    * Currently the logic is that if we have any rule span, take that span over a predicted
    * one. If we're dealing with multiple rule based spans, then we delegate to special logic
    * for that.
    *
    * Assumes the passed in sequence of spans all overlap with the starting span.
    *
    * Note: without resorting to creating paths of possible spans, this chooses
    * the one span to rule them all; we only return one span, where in certain cases
    * you could return multiple:
    * E.g. with
    * 1)  A B C D
    * 2)    B C
    * 3)        D E F
    * If ABCD isn't the best, then we just return either BC or DEF, when in reality they
    * both could live on. This could be mediated by
    *
    * @param start the span to start comparing at.
    * @param spans the sequence of spans to consider that overlap with the starting span.
    * @return a single span.
    */
  def chooseSpan(start: Span, spans: Seq[Span]): Span = {
    var workspace = spans
    var chosen = start
    while (workspace.nonEmpty) {
      val head = workspace.head
      // if we're both rules
      if (chosen.isRule && head.isRule) {
        // take the greedy way (treat black or white list rules the same as normal)
        chosen = chooseBetweenRuleSpans(chosen, head)
        // if the head is a rule, but chosen is ML
      } else if (!chosen.isRule && head.isRule) {
        // take the rule since we're a ML based span
        chosen = head
      } // if chosen is Rule, but head is ML -- keep rule -- i.e. do nothing
      workspace = workspace.tail
    }
    chosen
  }

  /**
    * Greedily chooses the span with the best absolute probability when subtracted by 0.5.
    *
    * Given a starting span, and a sequence to mull over, looks at each
    * one to decide which one should be returned.
    *
    * @param start the span to start at.
    * @param spans the spans to consider.
    * @return a single span that had a better absolute probability when subtracted by 0.5.
    */
  def chooseRuleSpanGreedily(start: Span, spans: Seq[Span]): Span = {
    spans.foldLeft(start)({case (chosenSpan, span) =>
      Span.chooseBetweenRuleSpans(chosenSpan, span)
    })
  }

  /**
    * Chooses between rule spans.
    *
    * Chooses between rule spans based on which one has the greater number when we subtract
    * the weight by 0.5 and take the absolute value.
    * If that is equal, we tie break on the longest span length, else tie break on the label name.
    *
    * Assumes: that we're only dealing with rule spans -- since this only "makes sense"
    * for them. Also we assume we need a tolerance as to what is zero, since we're dealing with
    * floating point numbers.
    *
    * @param chosen the current span that is chosen.
    * @param contender the span that is the contender for unseating the current span.
    * @return a singe span.
    */
  def chooseBetweenRuleSpans(chosen: Span, contender: Span): Span = {
    val delta: Float = Math.abs(contender.probability - 0.5f) - Math.abs(chosen.probability - 0.5f)
    if (delta > ZERO_TOLERANCE) {
      contender
    } else if (delta < -ZERO_TOLERANCE) {
      // keep chosen since that is higher
      chosen
    } else {
      // equal, tie break on length
      if (contender.length > chosen.length) {
        contender
      } else if (contender.length < chosen.length) {
        chosen
      } else {
        // equal on length, tie break on label
        if (contender.label < chosen.label) {
          contender
        } else {
          chosen
        }
      }
    }
  }

  /**
    * Finds any contiguous overlapping spans and unions them and takes their average.
    *
    * Note: Assumes they are all the same label & sorted in order of offset.
    *
    * E.g. given:
    *  1) A-B-C-D
    *  2)   B-C
    *  3)       D-E
    *  4)           F-G
    *  5)             G-H-I
    * Would union [1, 2, 3] and [4, 5].
    *
    * @param spans to union; assumes they are all the same label & sorted in order
    *              of offset.
    * @return
    */
  def unionAndAverageOverlaps(spans: Seq[Span]): Seq[Span] = {
    spans match {
      case Seq() => Seq()
      case head :: tail => {
        val mutableList = mutable.ListBuffer[Span]()
        var workspace = head :: tail
        while (workspace.nonEmpty) {
          val head = workspace.head
          val overlapping = Span.getContiguousOverlappingSpans(head, workspace.tail)
          mutableList += Span.unionAndAverage(head +: overlapping)
          workspace = workspace.tail.slice(overlapping.size, tail.length)
        }
        mutableList.toSeq
      }
    }
  }

  /**
    * Gets largest set of contiguous overlapping spans starting with the starting span.
    *
    * Assumes spans are sorted by (offset, -length), and the start span has an
    * offset <= the spans.head.offset.
    *
    * @param start the starting span, should have offset <= offsets in spans.
    * @param spans sequence of spans, sorted by (offset, -length).
    * @return a sequence of spans that contiguously overlap with the start span.
    */
  def getContiguousOverlappingSpans(start: Span, spans: Seq[Span]): Seq[Span] = {
    // the right most span to check for overlaps with
    var current = start
    spans.takeWhile(span => {
      // make sure the span offset is within the bounds of the head start & end.
      if (current.offset <= span.offset && span.offset < current.end) {
        current = span
        true
      } else {
        false
      }
    })
  }

  /**
    * Unions a sequence of spans and averages their probabilities.
    *
    * You only case this method makes sense is when there is a single string
    * of overlaps. e.g.
    *  A-B-C-D
    *    B-C
    *      C-D-E
    *          E-F
    * Note: Assumes they are all the same label & sorted in order of offset.
    *
    * Unions the flags and averages the probability, and unions them into
    * a single span.
    *
    * @param spans to union; assumes they are all the same label & sorted in order
    *              of offset.
    * @return a single unioned span
    */
  def unionAndAverage(spans: Seq[Span]): Span = {
    assert(spans.nonEmpty, "can only union a non-empty sequence of spans")
    val sum = spans.map(_.probability).sum
    val flags = spans.map(_.flags).foldLeft(0)({case (accum, flag) => accum | flag})
    val maxEnd = spans.map(_.end).max
    val minOffset = spans.map(_.offset).min
    new Span(spans.head.label, sum / spans.length.toFloat, flags, minOffset, maxEnd - minOffset)
  }
}
