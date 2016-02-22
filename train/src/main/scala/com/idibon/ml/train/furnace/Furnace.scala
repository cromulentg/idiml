package com.idibon.ml.train.furnace

import com.idibon.ml.feature.FeaturePipeline
import com.idibon.ml.predict.{PredictModel, PredictResult}
import com.idibon.ml.predict.ml.{MLModel}
import com.idibon.ml.train.datagenerator.SparkDataGenerator
import org.apache.spark.sql.DataFrame
import org.json4s.JsonAST.JObject

/**
  * Trait that produces MLModels.
  * We stick elements into the furnace to produce items that go into an alloy.
  */
trait Furnace[T <: PredictResult] {

  /**
    * Function fits a model to data in the dataframe.
    *
    * @param label
    * @param data
    * @param pipelines
    * @return
    */
  def fit(label: String,
          data: Seq[DataFrame],
          pipelines: Option[Seq[FeaturePipeline]]): PredictModel[T]

  /**
    * Function is used for featurizing data.
    *
    * @param rawData
    * @param dataGen
    * @param featurePipelines
    * @return
    */
  def featurizeData(rawData: () => TraversableOnce[JObject],
    dataGen: SparkDataGenerator,
    featurePipelines: Seq[FeaturePipeline]): Seq[Option[Map[String, DataFrame]]]
}


