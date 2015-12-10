import org.apache.spark.mllib.linalg.Vector
import scala.reflect.runtime.universe._

package com.idibon.ml.feature {

  /** FeatureTransformers convert from one or more lists of input features
    * into a list of output features.
    * 
    * FeatureTransformer implementations are expected to ?
    */
  trait FeatureTransformer[F <: Feature[_]] {

    /** Returns the required input data for this transformer.
      * 
      * Input data requirements are defined as a map from the named input
      * field to the internal data representation that is allowed for that
      * field.
      * 
      * "tokens" => typeOf[String]
      */
    def input: Map[String, Type]

    /** Returns the optional configuration parameters for this transformer */
    def options: Option[Map[String, Type]]

    /**
      * Applies the feature transformation to the provided input features
      *
      * 
      */
    def apply(input: Map[String, Seq[Feature[_]]]): Seq[F]
  }

  /** A FeatureSpace is a special type of FeatureTransformer used to convert
    * from intermediate feature data types (e.g., Strings) into vectors */
  trait FeatureSpace[V <: Vector]
      extends FeatureTransformer[Feature[V]] with Archivable {
  }
}
