package com.idibon.ml.feature

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

  /** Produces a sequence of shapes from a sequence of Tokens.
    *
    * @param feature
    * @return wraps the passed in feature as a sequence.
    */
  def apply(feature: Feature[_]): Seq[Feature[_]] = {
    Seq(feature)
  }
}
