package com.idibon.ml.train.furnace

import com.idibon.ml.alloy.HasTrainingSummary
import com.idibon.ml.common.Engine
import com.idibon.ml.feature.{Buildable, FeaturePipeline}
import com.idibon.ml.predict.ml.metrics._
import com.idibon.ml.predict.{PredictModel, Classification}
import com.idibon.ml.predict.ml.{TrainingSummary, IdibonLogisticRegressionModel}
import com.idibon.ml.train.datagenerator.SparkDataGenerator
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{TrainValidationSplit, ParamGridBuilder, CrossValidator}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.json4s.JsonAST.JObject
import org.json4s._
import org.json4s.native.JsonMethods


/**
  * Abstract class that makes it easy to train simple logistic regression models using the
  * spark ml LogisticRegression implementation.
  *
  * @param engine the engine context to use for RDD & DataFrame generation
  * @tparam T The type of trainer to expect to train on.
  */
abstract class LogisticRegressionFurnace[T](engine: Engine)
    extends Furnace[Classification] with StrictLogging with MetricHelper {

  /**
    * Function to take care of featurizing the data.
    * Converts the input documents and feature pipeline into DataFrames
    *
    * @param rawData a function that returns an iterator over training documents.
    * @param dataGen SparkDataGenerator that creates the data splits for training.
    * @param featurePipeline the feature pipeline to use for labeled point creation.
    * @return a map from label name to the training DataFrame for that label
    */
  override def featurizeData(rawData: () => TraversableOnce[JObject],
                             dataGen: SparkDataGenerator,
                             featurePipeline: FeaturePipeline): Option[Map[String, DataFrame]] = {
    // produces data frames
    dataGen.getLabeledPointData(this.engine, featurePipeline, rawData)
  }

  /**
    * Function takes a data frame of data
    *
    * @param data
    * @return
    */
  override def fit(label: String, data: DataFrame, pipeline: Option[FeaturePipeline]) = {
    val lr = fitModel(getEstimator(), data)
    logTrainingSummary(label, lr)
    // wrap into one we want
    val wrapper = IdibonSparkLogisticRegressionModelWrapper.wrap(lr)
    // create MLModel for label:
    new IdibonLogisticRegressionModel(label, wrapper, pipeline) with HasTrainingSummary {
      override val trainingSummary = Some(Seq(createTrainingSummary(label, lr, data)))
    }
  }

  /**
    * Gathers metrics and packages them into a training summary.
    *
    * @param label the string value to group these metrics by.
    * @param lrm the logistic regression model trained to inspect.
    * @param data the training dataframe.
    * @return
    */
  def createTrainingSummary(label: String,
                            lrm: LogisticRegressionModel,
                            data: DataFrame): TrainingSummary = {
    // we already have training summary data courtesy of the summary in the model.
    val summary = lrm.summary.asInstanceOf[BinaryLogisticRegressionTrainingSummary]
    val maxFMeasure = summary.fMeasureByThreshold.select(max("F-Measure")).head().getDouble(0)
    val bestThreshold = summary.fMeasureByThreshold
      .where(summary.fMeasureByThreshold.col("F-Measure") === maxFMeasure)
      .select("threshold").head().getDouble(0)
    val metrics = Seq[Metric with Buildable[_,_]](
      new FloatMetric(MetricType.BestF1Threshold, MetricClass.Binary, bestThreshold.toFloat),
      new FloatMetric(
        MetricType.AreaUnderROC, MetricClass.Binary, summary.areaUnderROC.toFloat),
      new PointsMetric(
        MetricType.ReceiverOperatingCharacteristic,
        MetricClass.Binary,
        convertDataFrame(summary.roc, "TPR", "FPR")),
      new PointsMetric(
        MetricType.PrecisionRecallCurve,
        MetricClass.Binary,
        convertDataFrame(summary.pr, "precision", "recall")),
      new PointsMetric(
        MetricType.F1ByThreshold,
        MetricClass.Binary,
        convertDataFrame(summary.fMeasureByThreshold, "threshold", "F-Measure")),
      new PointsMetric(
        MetricType.PrecisionByThreshold,
        MetricClass.Binary,
        convertDataFrame(summary.precisionByThreshold, "threshold", "precision")),
      new PointsMetric(
        MetricType.RecallByThreshold,
        MetricClass.Binary,
        convertDataFrame(summary.recallByThreshold, "threshold", "recall")),
      new PropertyMetric(
        MetricType.HyperparameterProperties,
        MetricClass.Hyperparameter,
        lrm.parent.extractParamMap().toSeq.map(pp => {
          (pp.param.toString(), pp.value.toString)
        }))
    )
    val doubleToString = Map(1.0 -> "positive", 0.0 -> "negative")
    val dataSizes = getLabelCounts(data)
      .map({case(lab, size) => {
        new LabelIntMetric(MetricType.LabelCount, MetricClass.Binary,
          doubleToString(lab), size)
      }}).asInstanceOf[Seq[Metric with Buildable[_, _]]]
    new TrainingSummary(label, metrics ++ dataSizes)
  }

  /**
    * Helper method to convert a dataframe of points to a sequence of float to float.
    *
    * @param df the dataframe to get data from.
    * @param xName the x-axis name.
    * @param yName the y-axis name.
    * @return map of floats.
    */
  def convertDataFrame(df: DataFrame, xName: String, yName: String): Seq[(Float, Float)] = {
    implicit val formats = org.json4s.DefaultFormats
    df.toJSON.collect()
      .map(x => JsonMethods.parse(x).extract[Map[String, Double]])
      .map(m => (m(xName).toFloat, m(yName).toFloat))
  }

  /**
    * Method that fits data and returns a Logistic Regression Model ready for battle.
    *
    * @param estimator
    * @param data
    * @return
    */
  protected def fitModel(estimator: T, data: DataFrame): LogisticRegressionModel

  /**
    * Returns the thing that will do the training.
    *
    * @return
    */
  protected def getEstimator(): T

  /**
    *
    * @param label
    * @param lr
    */
  protected [furnace] def logTrainingSummary(label: String, lr: LogisticRegressionModel): Unit = {
    val binarySummary = lr.summary.asInstanceOf[BinaryLogisticRegressionSummary]
    // TODO: figure out how to output these summary stats - e.g. binarySummary.roc.show(100, false)
    val fMeasure: DataFrame = binarySummary.fMeasureByThreshold
    val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
    val bestThreshold = fMeasure.where(fMeasure.col("F-Measure") === maxFMeasure)
      .select("threshold").head().getDouble(0)
    // append info to atomic log line
    logger.info(s"Model for $label was fit using parameters: ${lr.parent.extractParamMap}\n" +
      s"Best threshold as determined by F-Measure is $bestThreshold for $label\n" +
      s"Area under ROC: ${binarySummary.areaUnderROC} for $label\n")
  }
}

/**
  * Trains a logistic regression model with the passed in parameters.
  *
  * @param builder
  */
class SimpleLogisticRegression(builder: SimpleLogisticRegressionBuilder)
  extends LogisticRegressionFurnace[LogisticRegression](builder.engine) {

  val maxIterations = builder.maxIterations
  val elasticNetParam = builder.elasticNetParam.head
  val regParam = builder.regParam.head
  val tolerance = builder.tolerance.head

  /**
    * Method that fits data and returns a Logistic Regression Model ready for battle.
    *
    * @param estimator
    * @param data
    * @return
    */
  override protected def fitModel(estimator: LogisticRegression, data: DataFrame): LogisticRegressionModel = {
    estimator.fit(data.cache())
  }

  /**
    * Returns the thing that will do the training.
    *
    * @return
    */
  override protected def getEstimator(): LogisticRegression = {
    new LogisticRegression()
      .setElasticNetParam(elasticNetParam)
      .setMaxIter(maxIterations)
      .setRegParam(regParam)
      .setTol(tolerance)
  }
}


/**
  * Performs cross validation to choose a logistic regression model.
  *
  * @param builder
  */
class XValLogisticRegression(builder: XValLogisticRegressionBuilder)
  extends LogisticRegressionFurnace[CrossValidator](builder.engine) {
  val maxIterations = builder.maxIterations
  val regressionParams = builder.regParam
  val elasticNetParams = builder.elasticNetParam
  val numberOfFolds = builder.numFolds
  val tolerances = builder.tolerance

  /**
    * Method that fits data and returns a Logistic Regression Model ready for battle.
    *
    * @param estimator
    * @param data
    * @return
    */
  protected def fitModel(estimator: CrossValidator, data: DataFrame): LogisticRegressionModel = {
    estimator.fit(data.cache()).bestModel.asInstanceOf[LogisticRegressionModel]
  }

  /**
    * Creates the cross validator for finding the parameters for the LogisticRegressionModel.
    *
    * @return
    */
  override protected def getEstimator(): CrossValidator = {
    val lr = new LogisticRegression().setMaxIter(maxIterations)
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, regressionParams)
      .addGrid(lr.elasticNetParam, elasticNetParams)
      .addGrid(lr.tol, tolerances)
      .build()
    logger.info("LogisticRegression parameters:\n" + lr.explainParams() + "\n")
    /* We now treat the LR as an Estimator, wrapping it in a CrossValidator instance.
     This will allow us to only choose parameters for the LR stage.
     A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
     Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
     is areaUnderROC.*/
    val cv = new CrossValidator()
      .setEstimator(lr)
      // TODO: decide on best evaluator (this uses ROC)
      .setEvaluator(new BinaryClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(numberOfFolds) // Use 3+ in practice
    cv
  }
}


/**
  * Performs hold out set evaluation to choose a logistic regression model.
  *
  * It randomly splits the incoming data and then uses it to figure out which options work the best.
  *
  * @param builder
  */
class HoldOutSetLogisticRegressionFurnace(builder: HoldOutSetLogisticRegressionFurnaceBuilder)
  extends LogisticRegressionFurnace[TrainValidationSplit](builder.engine) {
  val maxIterations = builder.maxIterations
  val regressionParams = builder.regParam
  val elasticNetParams = builder.elasticNetParam
  val tolerances = builder.tolerance
  val trainingSplit = builder.trainingSplit

  /**
    * Method that fits data and returns a Logistic Regression Model ready for battle.
    *
    * @param estimator
    * @param data
    * @return
    */
  protected def fitModel(estimator: TrainValidationSplit, data: DataFrame): LogisticRegressionModel = {
    estimator.fit(data.cache()).bestModel.asInstanceOf[LogisticRegressionModel]
  }

  /**
    * Creates the cross validator for finding the parameters for the LogisticRegressionModel.
    *
    * @return
    */
  override protected def getEstimator(): TrainValidationSplit = {
    val lr = new LogisticRegression().setMaxIter(maxIterations)
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, regressionParams)
      .addGrid(lr.elasticNetParam, elasticNetParams)
      .addGrid(lr.tol, tolerances)
      .build()
    logger.info("LogisticRegression parameters:\n" + lr.explainParams() + "\n")
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new BinaryClassificationEvaluator()) //Uses Area Under ROC Curve
      .setEstimatorParamMaps(paramGrid)
      // 80% of the data will be used for training and the remaining 20% for validation.
      .setTrainRatio(trainingSplit)
    trainValidationSplit
  }
}

/**
  * Creates individual furnaces per label.
  *
  * @param builder
  */
class PerLabelFurnace(builder: PerLabelFurnaceBuilder)
  extends Furnace[Classification] with StrictLogging {
  val engine: Engine = builder.engine
  val labelFurnaces: Map[String, Furnace[Classification]] = builder.builtFurnaces
  /**
    * Function fits a model to data in the dataframe.
    *
    * @param label
    * @param data
    * @param pipeline
    * @return
    */
  override def fit(label: String,
                   data: DataFrame,
                   pipeline: Option[FeaturePipeline]): PredictModel[Classification] = {
    labelFurnaces(label).fit(label, data, pipeline)
  }

  /**
    * Function is used for featurizing data.
    *
    * @param rawData
    * @param dataGen
    * @param featurePipeline
    * @return
    */
  override def featurizeData(rawData: () => TraversableOnce[JObject],
                             dataGen: SparkDataGenerator,
                             featurePipeline: FeaturePipeline): Option[Map[String, DataFrame]] = {
    // produces data frames
    dataGen.getLabeledPointData(this.engine, featurePipeline, rawData)
  }
}
