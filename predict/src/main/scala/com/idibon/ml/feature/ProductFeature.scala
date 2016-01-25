package com.idibon.ml.feature

import com.idibon.ml.alloy.Codec
import com.typesafe.scalalogging.Logger

/** Generic feature representing products of other features, such as N-grams
  * and skip-grams (both products of an arbitrary number of Tokens).
  */
case class ProductFeature(features: Seq[Feature[_]])
    extends Feature[Seq[Feature[_]]]
    with Buildable[ProductFeature, ProductFeatureBuilder] {

  def get = this.features

  /** Saves this ProductFeature to an output stream */
  def save(output: FeatureOutputStream) {
    Codec.VLuint.write(output, this.features.length)
    this.features.foreach(_ match {
      case f: Feature[_] with Buildable[_, _] => output.writeFeature(f)
      case f => {
        ProductFeature.logger.warn(s"Unable to save $f")
        output.writeFeature(new StringFeature(f.toString))
      }
    })
  }
}

object ProductFeature {
  val logger = Logger(org.slf4j.LoggerFactory
    .getLogger(classOf[ProductFeature]))
}

/** Paired builder class for ProductFeature */
class ProductFeatureBuilder extends Builder[ProductFeature] {
  /** Loads a ProductFeature from an input stream */
  def build(input: FeatureInputStream) = {
    val length = Codec.VLuint.read(input)
    new ProductFeature((0 until length).map(i => input.readFeature))
  }
}
