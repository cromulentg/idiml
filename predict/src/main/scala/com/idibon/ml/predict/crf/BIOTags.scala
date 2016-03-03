package com.idibon.ml.predict.crf

/** ConLL-style sub-tag for spans
  *
  * In NER tasks, every token within a labeled span is identified using
  * one of two types: either the beginning of the span (for the first token
  * within the span) or inside (for subsequent tokens). This is typically
  * implemented by training 2 pseudo-labels inside the model for each span
  * label (B-label and I-label, respectively). Tokens not within any span
  * are used to train a reserved model for the OUTSIDE pseudo-label
  */
object BIOType extends Enumeration {
  val BEGIN, INSIDE, OUTSIDE = Value
}

trait BIOTag {
  val bio: BIOType.Value
}

object BIOTag {

  /** Parses a string representation of a BIOTag back to the original object */
  def apply(s: String): BIOTag = s.charAt(0) match {
    case 'B' if s.length() > 1 => BIOLabel(BIOType.BEGIN, s.substring(1))
    case 'I' if s.length() > 1 => BIOLabel(BIOType.INSIDE, s.substring(1))
    case 'O' if s.length() == 1 => BIOOutside
    case _ => throw new IllegalArgumentException("Invalid tag")
  }
}

/** A BIOTag consisting of a tag type and a label name
  *
  * The tag type must be either BEGIN or INSIDE
  */
case class BIOLabel(bio: BIOType.Value, label: String) extends BIOTag {

  if (bio == BIOType.OUTSIDE)
    throw new IllegalArgumentException("Invalid type for BIOLabel")

  override def toString() = bio match {
    case BIOType.BEGIN => s"B$label"
    case BIOType.INSIDE => s"I$label"
  }
}

/** Singleton representation for outside-tagged items */
object BIOOutside extends BIOTag {
  val bio = BIOType.OUTSIDE

  override def toString() = "O"
}
