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
class KClassDataFrameGenerator(builder: KClassDataFrameGeneratorBuilder) extends DataFrameBase {
  val scale = builder.scale.build()
  /**
    * Creates a map of label -> list of labelled points for that label.
    *
    * This shares feature vectors for a document across labels.
    *
    * @param pipeline
    * @param docs
    * @return
    */
  override def createPerLabelLPs(pipeline: FeaturePipeline,
                                 docs: () => TraversableOnce[JObject]): Map[String, List[LabeledPoint]] = {
    implicit val formats = org.json4s.DefaultFormats
    docs().flatMap(document => {
      val annotations = (document \ "annotations").extract[JArray]
      // Run the pipeline to generate the feature vector -- so we will be sharing a reference to this.
      // CAVEAT: we might need to create an explicit FeatureVector for each label in the future.
      val featureVector = pipeline(document)
      if (featureVector.numActives < 1) {
        logger.info(s"No feature vector created for document ${document}")
        List() // we will skip it
      } else {
        // create list of (label, points)
        annotations.arr.map({ jsonValue => {
          // for each annotation, we assume it was provided so we can make a training point out of it.
          val JString(label) = jsonValue \ "label" \ "name"
          val JBool(isPositive) = jsonValue \ "isPositive"
          // Assign a number that MLlib understands
          val labelNumeric = if (isPositive) 1.0 else 0.0
          (label, LabeledPoint(labelNumeric, featureVector))
          }
        })
      }
      // need to covert toList for groupBy to work...
    }).toList
      // group by label
      .groupBy({ case (label, point) => label })
      // need to change (label, List((label, pt))) to just (label, List(pt))
      .map({ case (label, listLabelAndPoint) => {
      (label, listLabelAndPoint.map({ case (_, pt) => pt }))
    }
    }) // it's implicitly converted into a map so no need for explicit toMap
  }
}
