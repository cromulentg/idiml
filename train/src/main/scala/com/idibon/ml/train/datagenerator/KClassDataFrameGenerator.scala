package com.idibon.ml.train.datagenerator

import com.idibon.ml.feature.FeaturePipeline
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.json4s.JsonAST.{JBool, JString}
import org.json4s._

/**
  * Generator that produces data for K Classes, where each class is binary.
  *
  * Specifically this just implements taking data and creating the right labeled points from
  * it for training K binary classifiers.
  *
  */
class KClassDataFrameGenerator(builder: KClassDataFrameGeneratorBuilder)
    extends SparkDataGenerator {

  val scale = builder.scale.build()
  val lpg = new KClassLabeledPointGenerator
}
