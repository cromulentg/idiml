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
  trait FeatureTransformer {}

  trait TerminableTransformer {
    /**
      * Transformers should implement an idempotent freeze. Meaning if it's
      * called multiple times, it wont change from its initial frozen state.
      */
    def freeze(): Unit

    /**
      * This is the number of dimensions that is represented by the transform.
      * Note that pruning just reduces what is stored, not what it represented
      * overall. This should stay fixed once a transform is frozen.
      * @return
      */
    def numDimensions: Int

    /**
      * Function to capture feature selection essentially. A predicate function
      * is passed in to inform feature transforms what should not be kept.
      * @param transform
      */
    def prune(transform: Int => Boolean): Unit

    /**
      * For implementing significant features. Takes in a set of integers that
      * should map to the internal representation of that feature transformer,
      * and maps them to a human readable string. If the feature transformer
      * doesn't make sense to have something human interpretable, then it should
      * still return a string, no matter how non-sensical it might be.
      * @param indexes
      * @return
      */
    def getHumanReadableFeature(indexes: Set[Int]): List[(Int, String)]
  }
}
