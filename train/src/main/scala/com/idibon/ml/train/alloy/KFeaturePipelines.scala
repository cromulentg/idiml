package com.idibon.ml.train.alloy

import com.idibon.ml.common.Engine
import com.idibon.ml.feature.{FeaturePipeline, FeaturePipelineLoader}
import org.json4s.JObject

/**
  * Trait that deals with creating a single feature pipeline from configuration data.
  */
trait KFeaturePipelines {

  /**
    * Creates a Seq of feature pipelines from the passed in configuration.
    *
    * @param config
    * @return
    */
  def createFeaturePipelines(engine: Engine, config: JObject): Seq[FeaturePipeline]= {
    implicit val formats = org.json4s.DefaultFormats

    val pipelinesJson = (config \ "pipelines").extract[List[JObject]]

    pipelinesJson.map(p => new FeaturePipelineLoader().load(engine, None, Some(p)))
  }
}
