package com.idibon.ml.train.alloy

import com.idibon.ml.predict.{Label}
import org.json4s.JObject

/**
  * Houses objects representing a data set used for training.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>" on 3/2/16.
  */

/**
  * Object to encapsulate a training data set.
  *
  * The three possible splits are mutually exclusive.
  *
  * @param info information about this data set.
  * @param train documents used for training.
  * @param test documents used for evaluation.
  * @param dev Optional. Documents used for evaluation when tuning parameters.
  */
case class TrainingDataSet(info: DataSetInfo,
                           train:() => TraversableOnce[JObject],
                           test: () => TraversableOnce[JObject] = () => Seq[JObject](),
                           dev: () => TraversableOnce[JObject] = () => Seq[JObject]())

/**
  * Object to encapsulate information about a dataset.
 *
  * @param fold what fold number this data set is.
  * @param portion what portion of the data from the fold this data set is.
  * @param labelToDouble the mapping of label to double value.
  */
case class DataSetInfo(fold: Int, portion: Double, labelToDouble: Map[Label, Double])
