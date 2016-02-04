package com.idibon.ml.train.alloy

import com.idibon.ml.common.Engine
import com.idibon.ml.feature.{FeaturePipeline, FeaturePipelineLoader}
import org.json4s.JObject

/**
  * Trait that deals with creating a single feature pipeline from configuration data.
  */
trait OneFeaturePipeline {

  /**
    * Creates a single feature pipeline from the passed in configuration.
    *
    * @param config
    * @return
    */
  def createFeaturePipeline(engine: Engine, config: JObject): FeaturePipeline = {
    implicit val formats = org.json4s.DefaultFormats

    new FeaturePipelineLoader().load(engine, None, Some(config))
  }
}
