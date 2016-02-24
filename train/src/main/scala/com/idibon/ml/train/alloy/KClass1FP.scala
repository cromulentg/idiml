package com.idibon.ml.train.alloy

import java.util

import com.idibon.ml.feature.{Feature, FeaturePipeline}
import com.idibon.ml.predict.ml.metrics.{MetricHelper}
import com.idibon.ml.predict.ml.{TrainingSummary}
import com.idibon.ml.predict.{PredictOptions, Document, PredictModel, Classification}
import com.idibon.ml.predict.ensemble.GangModel
import com.idibon.ml.train.datagenerator.{MultiClassDataFrameGenerator, SparkDataGenerator}
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.mllib.evaluation.{MultilabelMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Vector
import org.json4s.JObject

/**
  * Trains K models using a global feature pipeline.
  *
  * @param builder
  */
class KClass1FP(builder: KClass1FPBuilder)
  extends BaseTrainer(builder.engine,
    builder.dataGenBuilder.build(),
    builder.furnaceBuilder.build(builder.engine))
    with OneFeaturePipeline with StrictLogging with MetricHelper {

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
                    classification_type: String): Map[String, PredictModel[Classification]] = {

    // create one feature pipeline
    val rawPipeline = pipelineConfig match {
      case Some(config) => createFeaturePipeline(this.engine, config)
      case _ => throw new IllegalArgumentException("No feature pipeline config passed.")
    }
    // prime the pipeline
    val primedPipeline = rawPipeline.prime(rawData())
    // create featurized data once since we only have one feature pipeline
    val featurizedData = furnace.featurizeData(rawData, dataGen, List(primedPipeline))
    val featuresUsed = new util.HashSet[Int](100000)
    // delegate to the furnace for producing MLModels for each label
    val models = featurizedData.head match {
      case Some(featureData) => featureData.par.map {
        case (label, data) => {
          val model = furnace.fit(label, List(data), None)
          (label, model, model.getFeaturesUsed())
        }
      }.toList.map({ // remove parallelism and gather all features used.
        case (label, model, usedFeatures) => {
          // add what was used so we can prune it from the global feature pipeline.
          usedFeatures.foreachActive((index, _) => featuresUsed.add(index))
          (label, model)
        }
      })
      case None => throw new RuntimeException("Failed to create training data.")
    }
    val gangTrainingSummary = classification_type match {
      case "classification.single" => Some( // do multi-class
         Seq(createMulticlassSummary("kclass1FPGang", rawData, primedPipeline, models)))
      case "classification.multiple" => Some(  // do multi-label
        Seq(createMultilabelSummary("kclass1FPGang", rawData, primedPipeline, models)))
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
    Map("kclass1fp" -> new GangModel(models.toMap, Some(primedPipeline)) {
      override val trainingSummary = gangTrainingSummary
    })
  }

  /**
    * Helper function to create a multilabel summary from k-binary classifiers.
    *
    * The assumption is that the classification task is to output as many labels it thinks it should.
    * It assumes we only take the label above the suggested threshold for consideration.
    *
    * @param identifier
    * @param rawData
    * @param pipeline
    * @param models
    * @return
    */
  def createMultilabelSummary(identifier: String,
                              rawData: () => TraversableOnce[JObject],
                              pipeline: FeaturePipeline,
                              models: List[(String, PredictModel[Classification])]): TrainingSummary  = {
    // 1. Get positive data points. Using the multiclass one is fine, since we just
    // want labeled points where each label has it's own double value.
    val (doubleLabelToUUIDLabel, dataPoints) = createPositiveLPs(pipeline, rawData)
    val uuidToDoubleLabel = doubleLabelToUUIDLabel.map(x => (x._2, x._1))
    val modelThresholds: Map[String, Float] = getModelThresholds(models)
    modelThresholds.foreach({case (uuidLabel, threshold) =>
      logger.info(s"Found $threshold for label $uuidLabel")})

    val predictions = dataPoints.map({case (positives, vector) => {
      val doc = new Document(null, Some(vector, (v: Vector) => Seq(None)))
      // for each model, predict on this vector
      val modelPredictions = models.map({case(uuidLabel, model) => {
        val singlePrediction = model.predict(doc, PredictOptions.DEFAULT)
        val threshold = modelThresholds(uuidLabel)
        // get only predictions that are above our threshold
        val classificationsOverThreshold = singlePrediction.filter(c => c.probability >= threshold)
        if (classificationsOverThreshold.isEmpty) {
          (uuidLabel, 0.0f)
        } else {
          (uuidLabel, classificationsOverThreshold.maxBy(c => c.probability).probability)
        }

      }})// filter no predictions
        .filter({case (uuidLabel, prob) => prob != 0.0f})
        // create single result combining altogether
        .foldLeft(List[Double]())({
        case (results, (uuidLabel, prob)) => uuidToDoubleLabel(uuidLabel) :: results
        case _ => List()
      })
      // return prediction (predicted double labels, actual double labels)
      (modelPredictions.toArray, positives.toArray)
    }})
    val predictionRDDs = engine.sparkContext.parallelize(predictions)
    val metrics = new MultilabelMetrics(predictionRDDs)
    logger.info(stringifyMultilabelMetrics(metrics))
    new TrainingSummary(identifier, createMultilabelMetrics(metrics, doubleLabelToUUIDLabel))
  }

  /**
    * Helper function to create a multiclass summary from k-binary classifiers.
    *
    * The assumption is that the classification task is to only output one label,
    * and we assume we take the highest label that is above its confidence threshold.
    *
    * @param identifier
    * @param rawData
    * @param pipeline
    * @param models
    * @return
    */
  def createMulticlassSummary(identifier: String,
                              rawData: () => TraversableOnce[JObject],
                              pipeline: FeaturePipeline,
                              models: List[(String, PredictModel[Classification])]): TrainingSummary = {
    // need to get all POSITIVE data points -- so use Multiclass to get that for me.
    val (doubleLabelToUUIDLabel, dataPoints) = createPositiveLPs(pipeline, rawData)
    val uuidToDoubleLabel = doubleLabelToUUIDLabel.map(x => (x._2, x._1))
    val modelThresholds: Map[String, Float] = getModelThresholds(models)
    modelThresholds.foreach({case (uuidLabel, threshold) =>
      logger.info(s"Found $threshold for label $uuidLabel")})
    val predictions = dataPoints.map({case (positives, vector) => {
      val doc = new Document(null, Some(vector, (v: Vector) => Seq(None)))
      // for each model, predict on this vector
      val modelPredictions = models.map({case(uuidLabel, model) => {
        val singlePrediction = model.predict(doc, PredictOptions.DEFAULT)
        val threshold = modelThresholds(uuidLabel)
        // get only predictions that are above our threshold
        val classificationsOverThreshold = singlePrediction.filter(c => c.probability >= threshold)
        if (classificationsOverThreshold.isEmpty) {
          (uuidLabel, 0.0f)
        } else {
          (uuidLabel, classificationsOverThreshold.maxBy(c => c.probability).probability)
        }
      }})
      // get the prediction with the highest probability
      val modelPrediction = modelPredictions.maxBy({case (uuidLabel, classificationProb) => classificationProb})
      // return prediction (predicted double label, actual double label)
      (uuidToDoubleLabel(modelPrediction._1), positives.head)
    }})
    val predictionRDDs = engine.sparkContext.parallelize(predictions)
    val metrics = new MulticlassMetrics(predictionRDDs)
    logger.info(stringifyMulticlassMetrics(metrics))
    new TrainingSummary(identifier, createMultiClassMetrics(metrics, doubleLabelToUUIDLabel))
  }

}
