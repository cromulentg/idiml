package com.idibon.ml.feature {

  /** FeatureTransformers convert from one or more lists of input features
    * into a list of output features. Marker interface.
    *
    * FeatureTransformer implementations must implement an #apply method
    * that takes as input Seqs of features (of whatever feature sub-
    * types are requireed by the transform) and returns a Seq of features,
    * such as the following signatures:
    *
    * * [A<:Feature[A], B<:Feature[B]](Seq[A], metadata: Seq[B]) -> Seq[F]
    * * (Seq[Feature[_]*) -> Seq[F]
    * * (
    */
  trait FeatureTransformer {
    def numDimensions: Int = ???
    def prune(transform: Int => Boolean): Unit = ???
    def getHumanReadableFeature(indexes: Set[Int]): List[(Int, String)] = ???
  }
}
