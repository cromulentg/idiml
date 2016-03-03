package com.idibon.ml.feature

/** Trait for elements in a FeatureGraph which must be frozen before use
  *
  * Some FeatureTransformers learn and store internal state based on the
  * set of features and documents that have prevoiusly been transformed, such as
  * a Vocabulary. These objects should be frozen after initialization / priming,
  * to ensure that the transformation function does not change as it is applied.
  */
trait Freezable[T <: Freezable[T]] {
  /** Freezes the object, preventing future use from modifying internal state
    *
    * Freeze should be idempotent: freezing an object which is already frozen
    * should have no effect.
    */
  def freeze(): T
}
