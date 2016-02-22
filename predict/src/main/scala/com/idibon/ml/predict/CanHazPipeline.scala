package com.idibon.ml.predict

import com.idibon.ml.feature.{FeaturePipeline, FeaturePipelineLoader, Feature}
import com.idibon.ml.common.Engine
import com.idibon.ml.alloy.Alloy
import org.json4s._
import org.apache.spark.mllib.linalg.SparseVector

/** Mixin trait for models that may optionally include a single FeaturePipeline
  *
  * Note: if more than one pipeline is defined, only the first is used */
trait CanHazPipeline {

  def featurePipeline: Option[FeaturePipeline]

  /** Transforms a document using the feature pipeline
    *
    * If the current object has a valid feature pipeline, this method will
    * transform the provided Document using the pipeline and return a new,
    * transformed Document. If not, the original document is returned
    * unmodified.
    *
    * @param doc Document to transform
    * @return the transformed document if a transform exists, else doc
    */
  def applyPipelineIfPresent(doc: Document): Document = {
    featurePipeline.map(p => {
      /* pass the feature pipeline's getFeatureByVector method on to
       * any models that need to return the original features */
      Document(doc.json, Some((p(doc.json), p.getFeaturesByVector _)))
    }).getOrElse(doc)
  }

  /** Saves the feature pipeline, if present, to an Alloy
    *
    * @param writer alloy writer configured for the current object's namespace
    */
  def savePipelineIfPresent(writer: Alloy.Writer): JField = {
    JField(CanHazPipeline.KEY,
      featurePipeline.flatMap(_.save(writer.within(CanHazPipeline.KEY)))
        .getOrElse(JNothing))
  }
}

object CanHazPipeline {
  val KEY = "featurePipeline"

  /** Loads a single feature pipeline, if one is defined in the current namespace
    *
    */
  def loadPipelineIfPresent(engine: Engine, reader: Option[Alloy.Reader],
      config: Option[JObject]): Option[FeaturePipeline] = {

    implicit val formats = org.json4s.DefaultFormats
    val meta = (config.get \ KEY).extract[Option[JObject]]
    meta.map(pipeConfig => new FeaturePipelineLoader()
      .load(engine, reader.map(_.within(KEY)), Some(pipeConfig)))
  }

  /** Combines a list of features with the model weights for each
    *
    * Filters out any un-invertible features from the returned list.
    */
  def zipFeaturesAndWeights(weights: SparseVector,
      features: Seq[Option[Feature[_]]]): Seq[(Feature[_], Float)] = {

    features.zipWithIndex
      .filter({ case (feat, _) => feat.isDefined })
      .map({ case (feat, index) => (feat.get, weights.values(index).toFloat) })
  }
}
