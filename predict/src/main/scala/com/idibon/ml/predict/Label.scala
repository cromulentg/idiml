package com.idibon.ml.predict

import java.util.UUID
import com.idibon.ml.alloy.Codec
import com.idibon.ml.feature.{FeatureInputStream, FeatureOutputStream, Builder, Buildable}

/** A Label represents a classification within a PredictModel
  *
  * Labels have a UUID which is guaranteed to be shared across all uses
  * of the label (even across Alloy versions), and a name that may change
  * from version to version.
  *
  * @param uuid the universally unique identifier for the label
  * @param name the label's human-readable name
  */
case class Label(uuid: UUID, name: String)
    extends Buildable[Label, LabelBuilder] {

  /** save the label to an OutputStream */
  def save(output: FeatureOutputStream) {
    output.writeLong(uuid.getMostSignificantBits())
    output.writeLong(uuid.getLeastSignificantBits())
    Codec.String.write(output, name)
  }

  /** alternate constructor for UUID strings */
  def this(uuid: String, name: String) {
    this(UUID.fromString(uuid), name)
  }
}

/** Paired Builder for SimpleLabel instances */
class LabelBuilder extends Builder[Label] {
  def build(input: FeatureInputStream) = {
    Label(new UUID(input.readLong, input.readLong), Codec.String.read(input))
  }
}
