import org.json4s._
import org.apache.spark.mllib.linalg.Vector
import com.idibon.ml.alloy.Alloy

package com.idibon.ml.feature {

  /** The feature pipeline transforms documents into feature vectors
    *
    *
    */
  trait FeaturePipeline extends Archivable {
    /** Applies the entire feature pipeline to the provided document.
      *
      * Returns a sequence of Vectors, one for each FeatureSpace included
      * in this pipeline.
      */
    def apply(document: Any): Seq[Vector] = {
      List[Vector]()
    }
  }
}
