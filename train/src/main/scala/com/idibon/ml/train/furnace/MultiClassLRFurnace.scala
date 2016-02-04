package com.idibon.ml.train.furnace

import com.idibon.ml.common.Engine
import com.idibon.ml.feature.FeaturePipeline
import com.idibon.ml.predict.Classification
import com.idibon.ml.predict.ml.{IdibonMultiClassLRModel}
import com.idibon.ml.train.alloy.MultiClass
import com.idibon.ml.train.datagenerator.SparkDataGenerator
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.mllib.classification.{IdibonSparkMLLIBLRWrapper, LogisticRegressionWithLBFGS}
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
  extends Furnace[Classification] with StrictLogging {
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
    * @param pipeline
    * @return
    */
  override def fit(label: String, data: DataFrame, pipeline: FeaturePipeline) = {
    // convert to rdd
    val rddData = data.rdd.map(row => LabeledPoint(row.getDouble(0), row.getAs[Vector](1)))
    val trainer = new LogisticRegressionWithLBFGS()
      .setNumClasses(labelToInt.size)
      .setIntercept(true)
    trainer.optimizer
      .setConvergenceTol(tolerance)
      .setRegParam(regParam)
      .setNumIterations(maxIterations)
    // run training
    val model = trainer.run(rddData.cache())

    // TODO: log some stats?
    val wrapper = IdibonSparkMLLIBLRWrapper.wrap(model)
    new IdibonMultiClassLRModel(labelToInt, wrapper, pipeline)
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
    * @param featurePipeline primed pipeline
    * @return
    */
  override def featurizeData(rawData: () => TraversableOnce[JObject],
                             dataGen: SparkDataGenerator,
                             featurePipeline: FeaturePipeline):
  Option[Map[String, DataFrame]] = {
    val data = dataGen.getLabeledPointData(this.engine, featurePipeline, rawData)

    // lets set what we're actually fitting to what integer label
    labelToInt = data match {
      case Some(data) => {
        // recreate label to Int
        data
          .filter(x => !x._1.equals(MultiClass.MODEL_KEY))
          .map({case (label, df) => (label, df.collect()(0).getDouble(0).toInt)})
      }
      case _ => return None
    }
    // only grab the data frame in the map that makes sense
    data.map(_.filter(x => x._1.equals(MultiClass.MODEL_KEY)))
  }
}
