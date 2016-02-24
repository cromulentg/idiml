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

  /** TerminableTransformers convert from one or more input features into
    * a representative Vector output with known dimensionality.
    *
    * Any FeatureTransformer specified as the output of a FeaturePipeline must
    * implement TerminableTransformer
    */
  trait TerminableTransformer extends FeatureTransformer {
    /**
      * Transformers should implement an idempotent freeze. Meaning if it's
      * called multiple times, it wont change from its initial frozen state.
      */
    def freeze(): Unit

    /** Returns the dimensionality of the generated Vectors
      *
      * May return None before the Transform is frozen
      *
      * This is the number of dimensions that is represented by the transform.
      * Note that pruning just reduces what is stored, not what it represented
      * overall. This should stay fixed once a transform is frozen.
      * @return
      */
    def numDimensions: Option[Int]

    /**
      * Function to capture feature selection essentially. A predicate function
      * is passed in to inform feature transforms what should not be kept.
      * @param transform
      */
    def prune(transform: (Int) => Boolean): Unit

    /** Returns the original feature corresponding to a dimension index
      *
      * For TerminableTransformer implementations where unique features are
      * mapped to specific dimensions in the output feature vector, this method
      * should perform the inverse transformation, returning the original
      * Feature for a specific dimension. This can be used to perform model
      * introspection and report the significant predictive features for a
      * predictive operation.
      *
      * Returns None if it is not possible to determine the Feature for the
      * provided index.
      *
      * @param index index of a dimension in the output vector to invert
      * @return the Feature corresponding to that index, or None
      */
    def getFeatureByIndex(index: Int): Option[Feature[_]]
  }
}
