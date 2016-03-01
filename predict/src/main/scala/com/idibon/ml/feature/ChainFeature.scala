package com.idibon.ml.feature

import com.idibon.ml.alloy.Codec

/** Generic feature used to represent features for attached links in a chain
  *
  * Features may be represented for offsets of up to [-128 .. 127] chain
  * positions from the current link.
  *
  * @param offset relative distance between links
  * @param feature the feature to capture from the neighboring link
  */
case class ChainFeature(offset: Byte, feature: Feature[_])
    extends Feature[Feature[_]]
    with Buildable[ChainFeature, ChainFeatureBuilder] {

  if (!feature.isInstanceOf[Buildable[_, _]])
    throw new IllegalArgumentException("Non-buildable feature")

  def get = this.feature

  def getHumanReadableString = feature.getHumanReadableString

  def save(output: FeatureOutputStream) {
    output.write(offset)
    val f = this.feature.asInstanceOf[Feature[_] with Buildable[_, _]]
    output.writeFeature(f)
  }
}

/** Paired builder to re-load saved ChainFeature objects */
class ChainFeatureBuilder extends Builder[ChainFeature] {
  def build(input: FeatureInputStream) = {
    val offset = input.readByte()
    ChainFeature(offset, input.readFeature)
  }
}
