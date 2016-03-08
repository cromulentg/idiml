package com.idibon.ml.feature.chain

import com.idibon.ml.feature._

import com.idibon.ml.alloy.Codec

/** Generic feature used to represent features for attached links in a chain
  *
  * Features may be represented for offsets of up to [-128 .. 127] chain
  * positions from the current link.
  *
  * @param offset relative distance between links
  * @param feature the feature to capture from the neighboring link
  */
case class ChainFeature[+T <: Feature[_]](offset: Byte, feature: T)
    extends Feature[ChainFeature[T]]
    with Buildable[ChainFeature[_], ChainFeatureBuilder] {

  if (!feature.isInstanceOf[Buildable[_, _]])
    throw new IllegalArgumentException("Non-buildable feature")

  def get = this

  def getHumanReadableString = feature.getHumanReadableString

  def save(output: FeatureOutputStream) {
    output.write(offset)
    val f = this.feature.asInstanceOf[Feature[_] with Buildable[_, _]]
    output.writeFeature(f)
  }
}

/** Paired builder to re-load saved ChainFeature objects */
class ChainFeatureBuilder extends Builder[ChainFeature[_]] {
  def build(input: FeatureInputStream) = {
    val offset = input.readByte()
    ChainFeature(offset, input.readFeature)
  }
}
