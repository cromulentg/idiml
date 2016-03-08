package com.idibon.ml.feature.chain

import com.idibon.ml.feature._
import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}
import com.idibon.ml.alloy.Alloy

import org.json4s.JsonDSL._
import org.json4s.JObject

/** Creates features from a neighborhood around each link in a Chain
  *
  * Sequence models can benefit by including the features from adjacent links
  * in the current link's feature vector. ChainNeighborhood specifies a window
  * (number of links before / after the current one) such that all of the
  * features defined for each link on the input chain within that window are
  * added as unique features for the current link.
  *
  * The current link's features are not included in the output Chain.
  *
  * NB: chain neighborhood multiplies the feature dimensionality by the window
  * size (since each feature in the input chain will be converted into
  * before + after ChainFeature variations).
  */
class ChainNeighborhood(before: Int, after: Int) extends FeatureTransformer
    with Archivable[ChainNeighborhood, ChainNeighborhoodLoader] {

  /** Saves the transform to an alloy */
  def save(writer: Alloy.Writer) = {
    Some(("before" -> before) ~ ("after" -> after))
  }

  def apply(chain: Chain[Seq[Feature[_]]]): Chain[Seq[ChainFeature[_]]] = {
    chain.map(link => {
      var head = link.previous
      val preceding = (1 to before).flatMap(distance => {
        val rv = head.map(_.value.map(f => ChainFeature((-distance).toByte, f)))
        head = head.flatMap(_.previous)
        rv.getOrElse(Seq())
      })
      head = link.next
      val following = (1 to after).flatMap(distance => {
        val rv = head.map(_.value.map(f => ChainFeature(distance.toByte, f)))
        head = head.flatMap(_.next)
        rv.getOrElse(Seq())
      })
      preceding ++ following
    })
  }
}

class ChainNeighborhoodLoader extends ArchiveLoader[ChainNeighborhood] {
  /** Loads the transform from an alloy */
  def load(engine: Engine, r: Option[Alloy.Reader], c: Option[JObject]) = {
    implicit val formats = org.json4s.DefaultFormats
    val config = c.get.extract[ChainNeighborhoodConfig]
    if (config.before > 127 || config.after > 127)
      throw new IllegalArgumentException("Neighborhood too large")
    new ChainNeighborhood(config.before.toByte, config.after.toByte)
  }
}

/** JSON schema for the chain neighborhood configuration */
case class ChainNeighborhoodConfig(before: Int, after: Int)
