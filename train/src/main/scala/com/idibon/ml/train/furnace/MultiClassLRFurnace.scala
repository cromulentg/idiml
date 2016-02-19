package com.idibon.ml.train.furnace

import com.idibon.ml.alloy.HasTrainingSummary
import com.idibon.ml.common.Engine
import com.idibon.ml.feature.{Buildable, FeaturePipeline}
import com.idibon.ml.predict.Classification
import com.idibon.ml.predict.ml.metrics._
import com.idibon.ml.predict.ml.{TrainingSummary, IdibonMultiClassLRModel}
import com.idibon.ml.train.alloy.MultiClass
import com.idibon.ml.train.datagenerator.SparkDataGenerator
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.mllib.classification.{LogisticRegressionModel, IdibonSparkMLLIBLRWrapper, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.{Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.DataFrame
import org.json4s.JsonAST.JObject

/**
  * This builds a multinomial LR model based off of MLLIB.
  *
  * @param builder
  */
class MultiClassLRFurnace(builder: MultiClassLRFurnaceBuilder)
  extends Furnace[Classification] with StrictLogging with MetricHelper {
  val engine: Engine = builder.engine
  val maxIterations = builder.maxIterations
  val tolerance = builder.tolerance.head
  val regParam = builder.regParam.head
  var labelToInt: Map[String, Int] = Map()

  /**
    * Function fits a model to data in the dataframe.
    *
    * @param label
    * @param data
    * @param pipelines
    * @return
    */
  override def fit(label: String, data: Seq[DataFrame], pipelines: Option[Seq[FeaturePipeline]]) = {
    // convert to rdd
    val rddData = data.head.rdd.map(row => LabeledPoint(row.getDouble(0), row.getAs[Vector](1)))
    val trainer = new LogisticRegressionWithLBFGS()
      .setNumClasses(labelToInt.size)
      .setIntercept(true)
    trainer.optimizer
      .setConvergenceTol(tolerance)
      .setRegParam(regParam)
      .setNumIterations(maxIterations)
    // run training
    val model = trainer.run(rddData.cache())
    val wrapper = IdibonSparkMLLIBLRWrapper.wrap(model)
    val pipeline = pipelines match {
      case Some(p) => Some(p.head)
      case None => None
    }
    new IdibonMultiClassLRModel(labelToInt, wrapper, pipeline) with HasTrainingSummary {
      override val trainingSummary = Some(Seq(createTrainingSummary(label, model, data.head)))
    }
  }

  /**
    * Gathers metrics and packages them into a training summary.
    *
    * @param label the string value to group these metrics by.
    * @param model the multiclass LR model
    * @param data
    * @return
    */
  def createTrainingSummary(label: String,
                            model: LogisticRegressionModel,
                            data: DataFrame): TrainingSummary = {
    // create training stats using the training data.
    val basicResults = data.rdd.map(row => {
      val prediction = model.predict(row(1).asInstanceOf[Vector])
      (prediction, row.getAs[Double](0))
    })
    val metrics = new MulticlassMetrics(basicResults)
    logger.info(stringifyMulticlassMetrics(metrics))
    val mmetrics = createMulticlassMetrics(metrics, labelToInt.map(x => (x._2.toDouble, x._1)))
    val intToLabel = labelToInt.map(x => (x._2, x._1))
    val dataSizes = getLabelCounts(data)
      .map({case(lab, size) => {
        new LabelIntMetric(MetricTypes.LabelCount, MetricClass.Multiclass,
          intToLabel(lab.toInt), size)
      }}).asInstanceOf[Seq[Metric with Buildable[_, _]]]
    new TrainingSummary(label, mmetrics ++ dataSizes)
  }

  /**
    * Function is used for featurizing data.
    *
    * It inspects the labels returned by the data generator to know how to
    * map string labels to integer indexes used by the featurized data. This
    * is then used in Fit to determine how many classes there are.
    *
    * @param rawData
    * @param dataGen
    * @param featurePipelines primed pipeline
    * @return
    */
  override def featurizeData(rawData: () => TraversableOnce[JObject],
                             dataGen: SparkDataGenerator,
                             featurePipelines: Seq[FeaturePipeline]):
  Seq[Option[Map[String, DataFrame]]] = {
    val data = dataGen.getLabeledPointData(this.engine, featurePipelines.head, rawData)

    // lets set what we're actually fitting to what integer label
    labelToInt = data match {
      case Some(data) => {
        // recreate label to Int
        data
          .filter(x => !x._1.equals(MultiClass.MODEL_KEY))
          .map({case (label, df) => (label, df.collect()(0).getDouble(0).toInt)})
      }
      case _ => return List(None)
    }
    // only grab the data frame in the map that makes sense
    val returnMe = data.map(_.filter(x => x._1.equals(MultiClass.MODEL_KEY)))

    returnMe match {
      case features: Some[_] => List(features)
      case _ => List(None)
    }
  }
}
