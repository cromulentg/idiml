package com.idibon.ml.train.datagenerator

import com.idibon.ml.common.Engine
import com.idibon.ml.feature.FeaturePipeline
import com.idibon.ml.train.alloy.MultiClass
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
  *
  * Note: createPerLabelLPs will instead of producing label -> dataframe, this will produce
  *   MutliClass.MODEL_KEY -> dataframe where each label has an index starting from 0.
  *
  */
class MultiClassDataFrameGenerator(builder: MultiClassDataFrameGeneratorBuilder)
  extends DataFrameBase with StrictLogging {
  val scale = builder.scale.build()
  /**
    * Creates a map of label -> RDD of labeled points.
    *
    * @param engine the engine to use to parallelize the data
    * @param perLabelLPs map of label to list of labeled points
    * @return
    */
  override def createPerLabelRDDs(engine: Engine,
                         perLabelLPs: Map[String, List[LabeledPoint]]): Map[String, RDD[LabeledPoint]] = {
    perLabelLPs.map {
      case (label, lp) => {
        val scaledLPs = this.scale.balance(label, engine.sparkContext.parallelize(lp))
        val splits = scaledLPs.groupBy(x => x.label)
          .map(x => s"Polarity: ${x._1}, Size: ${x._2.size}").collect().toList
        if (label.equals(MultiClass.MODEL_KEY))
          logger.info(s"\nCreated ${scaledLPs.count()} multi-class data points; with splits $splits")
        (label, scaledLPs)
      }
    }
  }

  /**
    * Creates a map of MultiClass.MODEL_KEY -> list of labelled points for the multiclass usecase.
    *
    * This includes a hack so that we know what integer index a label maps to. So the map returned
    * will be size(labels) + 1 in dimension.
    *
    * @param pipeline
    * @param docs
    * @return label -> data point indicating what integer label + MultiClass.MODEL_KEY -> list of labelled points
    */
  override def createPerLabelLPs(pipeline: FeaturePipeline,
                                 docs: () => TraversableOnce[JObject]): Map[String, List[LabeledPoint]] = {
    implicit val formats = org.json4s.DefaultFormats
    val labelToIntMap = mutable.HashMap[String, Int]()
    var numClasses = 0
    val trainingData = docs().flatMap(document => {
      // Run the pipeline to generate the feature vector
      val featureVector = pipeline(document)
      if (featureVector.numActives < 1) {
        logger.info(s"No feature vector created for document ${document}")
        List() // we will skip it
      } else {
        // get annotations
        val annotations = (document \ "annotations").extract[JArray]
        // for each annotation create a labelled point
        annotations.arr.map({ jsonValue => {
          // for each annotation, we assume it was provided so we can make a training point out of it.
          val JString(label) = jsonValue \ "label" \ "name"
          val JBool(isPositive) = jsonValue \ "isPositive"
          // If we haven't seen this label before, instantiate a list
          if (!labelToIntMap.contains(label)) {
            labelToIntMap.put(label, numClasses)
            numClasses += 1
          }
          val labelNumber = labelToIntMap.get(label).get
          (isPositive, LabeledPoint(labelNumber, featureVector))
        }
        }) // discard negative polarity annotations (can I do that in the above step?)
          .filter({ case (isPositive, lbPt) => isPositive })
          // map it to just points
          .map({ case (_, lbPt) => lbPt })
      }
    }).toList
    // TODO: think admittedly a very big hack....
    val label_to_num = labelToIntMap.map({ case (label, num) => (label, List(LabeledPoint(num, Vectors.zeros(0))))}).toMap
    // TODO: with this hack need to guard against labels being like this
    label_to_num + (MultiClass.MODEL_KEY -> trainingData)
  }
}
