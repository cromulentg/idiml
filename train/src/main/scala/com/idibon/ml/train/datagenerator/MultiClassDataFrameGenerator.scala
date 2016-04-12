package com.idibon.ml.train.datagenerator

import com.idibon.ml.common.Engine
import com.idibon.ml.feature.FeaturePipeline
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.json4s.JsonAST.{JBool, JString}
import org.json4s._

import scala.collection.mutable

/**
  * Generator that produces data multiclass dataframe for training.
  *
  * Specifically this just implements taking data and creating the right labeled points from
  * it for training 1 multi-class classifiers.
  */
class MultiClassDataFrameGenerator(builder: MultiClassDataFrameGeneratorBuilder)
  extends SparkDataGenerator {

  val scale = builder.scale.build()
  val lpg = new MulticlassLabeledPointGenerator
}
