package com.idibon.ml.train.furnace

import com.idibon.ml.common.Engine
import com.idibon.ml.feature.FeaturePipeline
import com.idibon.ml.predict.{PredictModel, Classification}
import com.idibon.ml.predict.ml.MLModel
import com.idibon.ml.train.datagenerator.SparkDataGenerator
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.sql.DataFrame
import org.json4s.JsonAST.JObject

/**
  * Performs both native Spark cross validation and xval of different feature pipelines to choose a logistic regression
  * model.
  *
  * @param builder
  */
class XValWithFPLogisticRegressionFurnace(builder: XValWithFPLogisticRegressionFurnaceBuilder)
  extends Furnace[Classification] with StrictLogging {
  val engine: Engine = builder.engine
  val maxIterations = builder.maxIterations
  val regressionParams = builder.regParam
  val tolerances = builder.tolerance
  val elasticNetParams = builder.elasticNetParam
  val numberOfFolds = builder.numFolds

  // TODO: Verify that only one of these is needed instead of one per pipeline
  val xvalFurnace = new XValLogisticRegressionFurnaceBuilder(maxIterations, regressionParams, tolerances,
    elasticNetParams, numberOfFolds).build(this.engine)

  var winningModel : PredictModel[Classification] = _
  var winningPipeline : FeaturePipeline = _

  /**
    * Function to generate the best LR model based on Spark's native xval
    *
    * @param label
    * @param data a List of training data for use with the pipeline at the same index location
    * @param pipelines a List of featurization pipelines for use with the data at the same index location
    * @return a List of tuples: (trained MLModels that are a best fit for each pipeline, areaUnderROC)
    */
  private def getBestModels(label: String,
                            data: Seq[DataFrame],
                            pipelines: Seq[FeaturePipeline]): Seq[(PredictModel[Classification], Double)] = {
    // Create a tuple of pipelines and the associated training data
    val pipelinesAndData = pipelines zip data

    // Using a standard xval furnace, retrieve the best model for each pipeline
    val models = pipelinesAndData.map {
      case (p: FeaturePipeline, d: DataFrame) => xvalFurnace.fit(label, List(d), Some(List(p)))
    }.toList

    val evalMetrics = models.map {
      case m: MLModel[Classification] => m.getEvaluationMetric()
    }

    models zip evalMetrics
  }


  /**
    * Function to generate the best LR model based on Spark's native xval
    *
    * @param evaluationMetric a List of tuples: (trained MLModels that are a best fit for each pipeline, evaluation metric)
    * @return an MLModel with the "best" evaluation metric
    */
  private def findWinningModelIndex(evaluationMetric: Seq[(PredictModel[Classification], Double)]) : Int = {
    var maxEvalMetric: Double = evaluationMetric.head._2
    var bestModelIndex: Int = 0

    var i = 0
    evaluationMetric.foreach { case (m, e) => {
        if (e > maxEvalMetric) {
          maxEvalMetric = e
          bestModelIndex = i
        }
        i = i + 1
      }
    }

    bestModelIndex
  }

  def getWinningPipeline() : FeaturePipeline = {
    winningPipeline
  }

  def getWinningModel() : PredictModel[Classification] = {
    winningModel
  }


  /**
    * Function to take care of featurizing the data.
    * Converts the input documents and feature pipeline into DataFrames
    *
    * @param rawData a function that returns an iterator over training documents.
    * @param dataGen SparkDataGenerator that creates the data splits for training.
    * @param featurePipelines the feature pipeline to use for labeled point creation.
    * @return a map from label name to the training DataFrame for that label
    */
  override def featurizeData(rawData: () => TraversableOnce[JObject],
                             dataGen: SparkDataGenerator,
                             featurePipelines: Seq[FeaturePipeline]): Seq[Option[Map[String, DataFrame]]] = {
    // produces data frames for each label
    featurePipelines.map {
      case p: FeaturePipeline => dataGen.getLabeledPointData(this.engine, p, rawData)
    }.toList
  }

  /**
    * Function takes a data frame of data
    *
    * @param label
    * @param data a List of training data for use with the pipeline at the same index location
    * @param pipelines a List of featurization pipelines for use with the data at the same index location
    * @return a List of trained MLModels that are a best fit for each pipeline in order
    */
  override def fit(label: String,
                   data: Seq[DataFrame],
                   pipelines: Option[Seq[FeaturePipeline]]): PredictModel[Classification] = {
    // First get the best model from Spark's xval for each pipeline
    val pipelines_not_null = pipelines match {
      case Some(p) => p
      case None => throw new RuntimeException("Failed to fit " + this.getClass())
    }
    val bestModels : Seq[(PredictModel[Classification], Double)] = getBestModels(label, data, pipelines_not_null)

    val winningModelIndex : Int = findWinningModelIndex(bestModels)
    val winningModel = bestModels(winningModelIndex)._1

    winningModel
  }

}
