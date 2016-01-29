package com.idibon.ml.train.furnace

import java.io.File
import java.nio.file.FileSystems
import java.security.SecureRandom

import com.idibon.ml.common.Engine
import com.idibon.ml.feature.FeaturePipeline
import com.idibon.ml.predict.ml.{IdibonLogisticRegressionModel, MLModel}
import com.idibon.ml.train.RDDGenerator
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.ml.classification.{LogisticRegression, BinaryLogisticRegressionSummary, LogisticRegressionModel, IdibonSparkLogisticRegressionModelWrapper}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.json4s.JsonAST.JObject
import org.json4s._

import scala.util.{Try, Success, Failure}

/**
  * Abstract class that makes it easy to train simple logistic regression models using the
  * spark ml LogisticRegression implementation.
  *
  * @param engine the engine context to use for RDD & DataFrame generation
  * @tparam T The type of trainer to expect to train on.
  */
abstract class LogisticRegressionFurnace[T](engine: Engine) extends Furnace with StrictLogging {

  /**
    * Function to take care of featurizing the data.
    * Converts the input documents and feature pipeline into DataFrames
    *
    * Creates a set of labeled points for each label in the training document
    * set, writes the points out to a temporary Parquet file, and re-loads
    * the file as a DataFrame ready for training.
    *
    * @param rawData a function that returns an iterator over training documents.
    * @param rDDGenerator RDDGenerator that creates the data splits for training.
    * @param featurePipeline the feature pipeline to use for labeled point creation.
    * @return a map from label name to the training DataFrame for that label
    */
  override def featurizeData(rawData: () => TraversableOnce[JObject],
                             rDDGenerator: RDDGenerator,
                             featurePipeline: FeaturePipeline): Option[Map[String, DataFrame]] = {
    /* use a random subdirectory within the system temp directory for
     * storing the intermediate training files */
    val trainerTemp = FileSystems.getDefault.getPath(
      System.getProperty("java.io.tmpdir"), "idiml", "training",
      Math.abs(SecureRandom.getInstance("SHA1PRNG").nextInt).toString).toFile

    trainerTemp.mkdirs();

    val sqlContext = new org.apache.spark.sql.SQLContext(engine.sparkContext)
    val rdds = rDDGenerator.getLabeledPointRDDs(this.engine, featurePipeline, rawData)
      // convert RDDs to data frames
    val files = rdds.zipWithIndex.map({ case ((label, rdd), index) => {
      (label, Try({
        /* can't call File.createTempFile here, because the parquet writer
         * doesn't like to overwrite files, including the empty file created
         * by File.createTempFile, so use the integer index of the label
         * within a random temp directory. :angry: */
        val file = new File(trainerTemp, s"idiml-${index}.parquet")
        logger.info(s"Saving RDD for $label to $file")
        try {
          sqlContext.createDataFrame(rdd)
            .write.parquet(file.getAbsolutePath)
          file
        } catch {
          case error: Throwable => {
            /* if saving fails for any reason, delete the temporary file
             * and map store a Failure in the map */
            logger.error(s"Failed to save training data for $label", error)
            file.delete
            throw error
          }
        }
      }))
    }})

    /* add a shutdown hook to make sure that we clean up all temp files,
     * regardless of how the JVM terminates */
    java.lang.Runtime.getRuntime.addShutdownHook(new Thread() {
      override def run {
        // delete all of the temporary files
        files.foreach(_ match {
          case (labelName: String, file: Success[File]) => file.get.delete
          case _ => {}
        })
        // and the random parent folder
        trainerTemp.delete
      }
    })

    // only train if all labels were successfully stored
    if (files.exists({ case (_, file) => file.isFailure })) {
      None
    } else {
      Some(files.map({ case (label, file) => {
        label -> sqlContext.read.parquet(file.get.getAbsolutePath)
      }}))
    }
  }

  /**
    * Function takes a data frame of data
    *
    * @param data
    * @return
    */
  override def fit(label: String, data: DataFrame, pipeline: FeaturePipeline): MLModel = {
    val lr = fitModel(getEstimator(), data)
    logTrainingSummary(label, lr)
    // wrap into one we want
    val wrapper = IdibonSparkLogisticRegressionModelWrapper.wrap(lr)
    // create MLModel for label:
    new IdibonLogisticRegressionModel(label, wrapper, pipeline)
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
  * @param engine
  */
class SimpleLogisticRegression(engine: Engine) extends LogisticRegressionFurnace[LogisticRegression](engine) {
  //TODO: make this configurable
  val maxIterations = 100
  val elasticNetParam = 0.5
  val regressionParam = 0.1

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
      .setRegParam(regressionParam)
  }
}

/**
  * Performs cross validation to choose a logistic regression model.
  *
  * @param engine
  */
class XValLogisticRegression(engine: Engine) extends LogisticRegressionFurnace[CrossValidator](engine) {
  //TODO: make this configurable
  val maxIterations = 100
  val regressionParams = Array(0.001, 0.01, 0.1)
  val elasticNetParams = Array(0.9, 1.0)
  val numberOfFolds = 10

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
