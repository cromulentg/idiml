package com.idibon.ml.train.alloy

import java.util

import com.idibon.ml.common.Engine
import com.idibon.ml.predict.{Classification, PredictModel}
import com.idibon.ml.predict.ensemble.GangModel
import com.idibon.ml.predict.ml.MLModel
import com.idibon.ml.predict.rules.DocumentRules
import com.idibon.ml.train.datagenerator.SparkDataGenerator
import com.idibon.ml.train.furnace.Furnace
import com.typesafe.scalalogging.StrictLogging
import org.json4s.JObject

/**
  * Static class for storing Multi-class constants.
  */
object MultiClass {
  // key to use to get the single model that handles all classes.
  val MODEL_KEY = "\u000all"
}

/**
  * This class creates a single model that handles prediction for all labels. i.e. multinomial model.
  * So instead of a label -> model mapping, this returns a single model mapped to MultiClass.MODEL_KEY.
  *
  * It (naturally) uses a single feature pipeline.
  *
  * CAVEAT:
  *  - Assumes labels are mutually exclusive. i.e. there is only ever one that is correct.
  *
  * @param builder
  */
class MultiClass1FP(builder: MultiClass1FPBuilder)
  extends BaseTrainer(builder.engine,
    builder.dataGenBuilder.build(),
    builder.furnaceBuilder.build(builder.engine)) with OneFeaturePipeline with StrictLogging {
  /**
    * This is the method where each alloy trainer does its magic and creates the MLModel(s) required.
    *
    * @param rawData
    * @param dataGen
    * @param pipelineConfig
    * @return
    */
  override def melt(rawData: () => TraversableOnce[JObject],
                    dataGen: SparkDataGenerator,
                    pipelineConfig: Option[JObject],
                    classification_type: String):
  Map[String, PredictModel[Classification]] = {
    // create one feature pipeline
    val rawPipeline = pipelineConfig match {
      case Some(config) => createFeaturePipeline(this.engine, config)
      case _ => throw new IllegalArgumentException("No feature pipeline config passed.")
    }
    // prime the pipeline
    val primedPipeline = rawPipeline.prime(rawData())
    // create featurized data once since we only have one feature pipeline
    val featurizedData = furnace.featurizeData(rawData, dataGen, List(primedPipeline)).head match {
      case Some(data) => data(MultiClass.MODEL_KEY) // should be only MultiClass.MODEL_KEY
      case _ => throw new RuntimeException("Failed to create training data.")
    }
    val featuresUsed = new util.HashSet[Int](100000)
    // delegate to the furnace for producing MLModel for all labels
    val model = furnace.fit(MultiClass.MODEL_KEY, List(featurizedData), Some(List(primedPipeline)))
    // add what was used so we can prune it from the global feature pipeline.
    model.getFeaturesUsed().foreachActive((index, _) => featuresUsed.add(index))

    logger.info(s"Fitted models, ${featuresUsed.size()} features used.")
    // function to pass down so that the feature transforms can prune themselves.
    // i.e. if it isn't used, remove it.
    def isNotUsed(featureIndex: Int): Boolean = {
      !featuresUsed.contains(featureIndex)
    }
    // prune unused features from global feature pipeline
    primedPipeline.prune(isNotUsed)
    // return MLModel
    Map(MultiClass.MODEL_KEY -> model)
  }
}
