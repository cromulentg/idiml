package com.idibon.ml.train.furnace

import com.idibon.ml.common.Engine
import com.idibon.ml.feature.FeaturePipeline
import com.idibon.ml.predict.ml.{IdibonMultiClassLRModel, MLModel}
import com.idibon.ml.train.SparkDataGenerator
import com.idibon.ml.train.alloy.MultiClass
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.mllib.classification.{IdibonSparkMLLIBLRWrapper, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.json4s.JsonAST.JObject

/**
  * This builds a multinomial LR model based off of MLLIB.
  * @param engine
  */
class MultiClassLRFurnace(engine: Engine) extends Furnace[RDD[LabeledPoint]] with StrictLogging {

  val maxIterations = 100
  var labelToInt: Map[String, Int] = Map()

  /**
    * Function fits a model to data in the dataframe.
    *
    * @param label
    * @param data
    * @param pipeline
    * @return
    */
  override def fit(label: String, data: RDD[LabeledPoint], pipeline: FeaturePipeline): MLModel = {
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(labelToInt.size)
      .setIntercept(true)
      .run(data.cache())

    // TODO: log some stats?
    val wrapper = IdibonSparkMLLIBLRWrapper.wrap(model)
    new IdibonMultiClassLRModel(labelToInt, wrapper, pipeline)
  }

  /**
    * Function is used for featurizing data.
    *
    * @param rawData
    * @param dataGen
    * @param featurePipeline primed pipeline
    * @return
    */
  override def featurizeData(rawData: () => TraversableOnce[JObject],
                             dataGen: SparkDataGenerator[RDD[LabeledPoint]],
                             featurePipeline: FeaturePipeline):
  Option[Map[String, RDD[LabeledPoint]]] = {
    val data = dataGen.getLabeledPointData(this.engine, featurePipeline, rawData)
    // lets set what we're actually fitting to what integer label
    labelToInt = data match {
      case Some(data) => {
        // recreate label to Int
        data
          .filter(x => !x._1.equals(MultiClass.MODEL_KEY))
          .map({case (label, df) => (label, df.collect()(0).label.toInt)})
      }
      case _ => return None
    }
    // only grab the data frame in the map that makes sense
    data.map(_.filter(x => x._1.equals(MultiClass.MODEL_KEY)))
  }
}
