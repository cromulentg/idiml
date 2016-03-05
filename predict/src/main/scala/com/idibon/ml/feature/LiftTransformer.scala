package com.idibon.ml.feature

import scala.collection.mutable.ArrayBuffer

/**
  * Use this to wrap a "Single Feature" i.e. a transformer that
  * only outputs a single feature, into a sequence for downstream
  * consumption by say the IndexTransformer. This is needed if you
  * want this to be output as a feature for consideration by a
  * ML system.
  *
  * Why the name Lift? See https://wiki.haskell.org/Lifting.
  * */
class LiftTransformer extends FeatureTransformer {

  /**
    * Wraps a single feature into a sequence of features.
    * Because this is a variadic argument, scala implicitly wraps
    * this into a sequence, so we can just return that!
    *
    * @param feature A single feature, or multiple individual features
    * @return wraps the passed in feature as a sequence.
    */
  def apply(feature: Feature[_]*): Seq[Feature[_]] = feature
}

/** Applies the lift transformation to each link across one or more chains
  *
  * Many chain feature transformations produce chains with singleton features
  * e.g., tokenization, which generates Chain[Token]. Construction of the final
  * feature vector requires all features for each link be collapsed into a
  * sequence. The ChainLiftTransformer performs a link-wise collation of
  * features into a Seq[Feature[_]] for each link
  */
class ChainLiftTransformer extends FeatureTransformer {

  def apply(chains: Chain[Feature[_]]*): Chain[Seq[Feature[_]]] = {
    /* prepare the return chain by lifting each link from the chain
     * provided as the first argument into a single-element Seq[Feature[_]] */
    val lifted = new ArrayBuffer[Seq[Feature[_]]](chains.head.size)
    chains.head.foreach(f => lifted += Seq[Feature[_]](f.value))

    /* then prepend the feature from each subsequent chains to its Seq.
     * NB: the features will be in reverse order in the Seq from the
     * order specified in the argument list */
    chains.tail.foreach(chain => {
      chain.toIterable.zipWithIndex.foreach({ case (link, index) => {
        lifted(index) = link.value +: lifted(index)
      }})
    })
    Chain(lifted)
  }
}
