package com.idibon.ml.train.alloy

import java.util

import com.idibon.ml.predict.ml.MLModel
import com.idibon.ml.train.datagenerator.SparkDataGenerator
import com.typesafe.scalalogging.StrictLogging
import org.json4s.JObject

import scala.util.{Failure, Try}

/**
  * Trains K models using a global feature pipeline.
  *
  * @param builder
  */
class KClass1FP(builder: KClass1FPBuilder)
  extends BaseTrainer(builder.engine,
    builder.dataGenBuilder.build(),
    builder.furnaceBuilder.build(builder.engine)) with OneFeaturePipeline with StrictLogging {

  /**
    * Implements the overall algorithm for putting together the pieces required for an alloy.
    *
    * @param rawData
    * @param dataGen
    * @param pipelineConfig
    * @return
    */
  override def melt(rawData: () => TraversableOnce[JObject],
                    dataGen: SparkDataGenerator,
                    pipelineConfig: Option[JObject]): Try[Map[String, MLModel]] = {
    // create one feature pipeline
    val rawPipeline = pipelineConfig match {
      case Some(config) => createFeaturePipeline(this.engine, config)
      case _ => return Failure(new IllegalArgumentException("No feature pipeline config passed."))
    }
    // prime the pipeline
    val primedPipeline = rawPipeline.prime(rawData())
    // create featurized data once since we only have one feature pipeline
    val featurizedData = furnace.featurizeData(rawData, dataGen, primedPipeline)
    val featuresUsed = new util.HashSet[Int](100000)
    // delegate to the furnace for producing MLModels for each label
    val models = featurizedData match {
      case Some(featureData) => featureData.par.map {
        case (label, data) => {
          val model = furnace.fit(label, data, primedPipeline)
          (label, model, model.getFeaturesUsed())
        }
      }.toList.map({ // remove parallelism and gather all features used.
        case (label, model, usedFeatures) => {
          // add what was used so we can prune it from the global feature pipeline.
          usedFeatures.foreachActive((index, _) => featuresUsed.add(index))
          (label, model)
        }
      })
      case None => return Failure(new RuntimeException("Failed to create training data."))
    }
    logger.info(s"Fitted models, ${featuresUsed.size()} features used.")
    // function to pass down so that the feature transforms can prune themselves.
    // i.e. if it isn't used, remove it.
    def isNotUsed(featureIndex: Int): Boolean = {
      !featuresUsed.contains(featureIndex)
    }
    // prune unused features from global feature pipeline
    primedPipeline.prune(isNotUsed)
    // return MLModels
    Try(models.toMap)
  }
}
