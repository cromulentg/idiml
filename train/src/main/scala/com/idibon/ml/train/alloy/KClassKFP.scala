package com.idibon.ml.train.alloy

import java.util

import com.idibon.ml.feature.FeaturePipeline
import com.idibon.ml.predict.{PredictModel, Classification}
import com.idibon.ml.predict.ensemble.GangModel
import com.idibon.ml.train.datagenerator.SparkDataGenerator
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.sql.DataFrame
import org.json4s.JObject

/**
  * Trains K models using multiple feature pipelines.
  *
  * @param builder
  */
class KClassKFP(builder: KClassKFPBuilder)
  extends BaseTrainer(builder.engine,
    builder.dataGenBuilder.build(),
    builder.furnaceBuilder.build(builder.engine)) with KFeaturePipelines with StrictLogging {

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
                    pipelineConfig: Option[JObject],
                    classification_type: String = "classification.single"): Map[String, PredictModel[Classification]] = {

    // 1. Create feature pipelines
    val rawPipelines = pipelineConfig match {
      case Some(config) => createFeaturePipelines(this.engine, config)
      case _ => throw new IllegalArgumentException("No feature pipeline config passed.")
    }

    // 2. Prime the pipelines
    // TODO: Add save/load to keep this from blowing up working memory
    val primedPipelines = rawPipelines.map(p => p.prime(rawData()))

    // 3. Featurize the data for each pipeline
    //    Returns: Seq[Option[Map[String, DataFrame]]]
    val featurizedData = furnace.featurizeData(rawData, dataGen, primedPipelines)

    //    Build a tuple of the pipeline and its associated per-label feature vector map
    val pipelinesFeatures = primedPipelines zip featurizedData

    // 4. Fit the models
    val pipelineFeaturesMap : Seq[(FeaturePipeline, Map[String, DataFrame])] = pipelinesFeatures.map {
      case (p : FeaturePipeline, f: Option[Map[String, DataFrame]]) => (p, f.get)
    }

    val pipelineLabelFeatures : Seq[(FeaturePipeline, String, DataFrame)] = pipelineFeaturesMap.map {
      case (p: FeaturePipeline, f: Map[String, DataFrame]) => {
        f.map{
          case (label : String, features : DataFrame) => (p, label, features)
        }
      }
    }.flatten

    val labelFeatures : Seq[(String, PredictModel[Classification], Double)] = pipelineLabelFeatures.map {
      case (pipeline, label, features) => {
        // Get a copy of the pipeline
        val pipelineCopy = pipeline.prime(None)

        // Fit the model & evaluate it
        val model = furnace.fit(label, List(features), Some(List(pipelineCopy)))
        val metric = model.getEvaluationMetric()

        // 5. Prepare for pruning
        val featuresUsed = new util.HashSet[Int](100000)
        //    Function to pass down so that the feature transforms can prune themselves.
        //    i.e. if it isn't used, remove it.
        def isNotUsed(featureIndex: Int): Boolean = {
          !featuresUsed.contains(featureIndex)
        }
        //    Gather all features used.
        val usedFeatures = model.getFeaturesUsed()
        //    add what was used so we can prune it from the feature pipeline.
        usedFeatures.foreachActive((index, _) => featuresUsed.add(index))
        logger.info(s"Fitted models, ${featuresUsed.size()} features used.")

        // 6. Prune unused features
        pipelineCopy.prune(isNotUsed)

        // Now that the furnace includes the pipeline, there's no need to keep track of it
        (label, model, metric)
      }
    }

    // 7. Find the winning models & pipeline
    var winningModels = scala.collection.mutable.Map[String, PredictModel[Classification]]()
    val labelModelsGroupBy = labelFeatures.groupBy(_._1)

    // Iterate over the groupBy results to find the model with the best evaluation metric per label
    labelModelsGroupBy.foreach {
      case (label: String, models: List[(String, PredictModel[Classification], Double)]) => {
        var maxEvalMetric: Double = 0.0
        // Set this value temporarily
        var maxModel: PredictModel[Classification] = models.head._2
        models.foreach {
          case (_: String, model: PredictModel[Classification], evalMetric: Double) => {
            if (evalMetric > maxEvalMetric) {
              maxEvalMetric = evalMetric
              maxModel = model
            }
          }
        }
        // Store the model with the best evaluation metric
        winningModels += (label -> maxModel)
      }
    }

    // 8. Return winning models & pipeline
    Map("kclasskfp" -> new GangModel(winningModels.toMap, None))
  }
}

