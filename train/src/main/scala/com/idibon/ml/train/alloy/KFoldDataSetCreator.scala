package com.idibon.ml.train.alloy

import com.idibon.ml.predict.Label
import com.typesafe.scalalogging.StrictLogging
import org.json4s.JsonAST.JObject

import scala.util.Random


/**
  * Trait to house creating logical data sets.
  *
  * Trainers can use this trait to get access to creating K-folds from a single training data set.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>" on 3/1/16.
  */
trait KFoldDataSetCreator extends StrictLogging {

  /**
    * Creates the datasets for each fold.
    *
    * The idea is that we first go by label, and place those datapoints nicely in each fold. We then
    * combine all the folds & portions together. That way, all label data is distributed as naturally
    * across the folds and portions as in real life.
    *
    * @param docs
    * @param numFolds
    * @param portions
    * @param foldSeed
    * @return
    */
  def createFoldDataSets(docs: () => TraversableOnce[JObject],
                         numFolds: Int,
                         portions: Array[Double],
                         foldSeed: Long): Seq[Fold] = {
    // create random mapping of training examples to fold on examples that pertain to this uuid label
    val validationFoldMapping = createValidationFoldMapping(docs().toStream, numFolds, foldSeed)
    // create "folds"
    val folds = (0 until numFolds).map { fold: Int =>
      val holdout = getKthItems(docs().toStream, validationFoldMapping.toStream, fold)
      val trainingStreams = portions.map { portion =>
        new PortionStream(
          portion,
          getPortion(
            getAllButKthItems(docs().toStream, validationFoldMapping.toStream, fold),
            portion
          )
        )
      }
      new Fold(fold, holdout, trainingStreams)
    }
    folds.toSeq
  }

  /**
    * Creates a data set for X folds where the training set is the passed in portion.
    *
    * @param docs
    * @param numFolds
    * @param portion
    * @param foldSeed
    * @param labelToDouble
    * @return
    */
  def createFoldDataSets(docs: () => TraversableOnce[JObject],
                         numFolds: Int,
                         portion: Double,
                         foldSeed: Long,
                         labelToDouble: Map[Label, Double]): Seq[TrainingDataSet] = {
    // create random mapping of training examples to fold on examples that pertain to this uuid label
    val validationFoldMapping = createValidationFoldMapping(docs().toStream, numFolds, foldSeed)
    // handle small data use case
    if (validationFoldMapping.length < numFolds * 4) {
      logger.warn("Individual fold data will be meaningless; You don't have enough data; " +
        "defaulting to training and evaluating on all data.")
      val dataSetInfo = new DataSetInfo(0, 1.0, labelToDouble)
      Seq(new TrainingDataSet(dataSetInfo, () => docs().toStream, () => docs().toStream))
    } else {
      // create "folds"
      val folds = (0 until numFolds).map { fold: Int =>
        val testSet = getKthItems(docs().toStream, validationFoldMapping.toStream, fold)
        val trainingSet = getPortion(
          getAllButKthItems(docs().toStream, validationFoldMapping.toStream, fold),
          portion)
        val dataSetInfo = new DataSetInfo(fold, portion, labelToDouble)
        new TrainingDataSet(dataSetInfo, () => trainingSet, () => testSet)
      }
      folds.toSeq
    }
  }

  /**
    * Helper method to create the mapping of training item to fold.
    *
    * This method takes a stream of training data, which is assumed to be filtered to the
    * corresponding examples we want already, and maps a fold number to it in a giant array.
    * The array is then shuffled to get a random ordering (so we don't have to
    * assume anything about the order of the stream). I.e. the initial mapping makes sure
    * we correctly distribute fold allocations, and then the shuffle randomizes it all.
    *
    * What's returned is an array of shorts (since folds are small numbers) that you can
    * then sequentially take from to map an example to a fold.
    *
    * @param docs stream of training data already prefiltered to what we want to map to.
    * @param numFolds the number of folds we are producing.
    * @param seed the value we use to base the shuffle off of. Using the same value means we
    *             can recreate the same folds with the same data.
    * @return an array of shorts, where the value is a random fold assignment.
    */
  def createValidationFoldMapping(docs: Stream[JObject], numFolds: Int, seed: Long): Array[Short] = {
    val validationFoldMapping = new Array[Short](docs.size)
    validationFoldMapping.zipWithIndex.foreach { case (_, index) =>
      validationFoldMapping(index) = (index % numFolds).toShort
    }
    val r = new Random(seed)
    r.shuffle(validationFoldMapping.toList).toArray
  }

  /**
    * Helper method to filter the stream to only items that map to this particular fold.
    *
    * This zips the document stream with the assigned fold stream, and then filters
    * based on the fold stream and then maps it back to just documents.
    *
    * This is used to create the hold out set.
    *
    * @param docs stream of documents to filter.
    * @param assignedFolds stream of assignments that will be mapped sequentially with documents.
    * @param k the fold to restrict returned documents to.
    * @return filtered stream of documents corresponding to K.
    */
  def getKthItems(docs: Stream[JObject], assignedFolds: Stream[Short], k: Int): Stream[JObject] = {
    docs.zip(assignedFolds).filter(_._2 == k).map(_._1)
  }

  /**
    * Helper method to filter the stream to only items that don't map to this particular fold.
    *
    * This zips the document stream with the assigned fold stream, and then filters
    * based on the fold stream and then maps it back to just documents.
    *
    * This is used to create the training set.
    *
    * @param docs stream of documents to filter.
    * @param assignedFolds stream of assignments that will be mapped sequentially with documents.
    * @param k the fold to not restrict returned documents to.
    * @return filtered stream of documents corresponding to all folds apart from K.
    */
  def getAllButKthItems(docs: Stream[JObject], assignedFolds: Stream[Short], k: Int): Stream[JObject] = {
    docs.zip(assignedFolds).filter(_._2 != k).map(_._1)
  }

  /**
    * Given a stream, takes the first portion of it.
    *
    * @param docs stream of documents to take from.
    * @param portion double between (0.0, 1.0] representing % to take from stream.
    * @return a stream limited to portion of original stream.
    */
  def getPortion(docs: Stream[JObject], portion: Double): Stream[JObject] = {
    val numToTake = docs.size.toDouble * portion
    docs.take(numToTake.toInt)
  }
}


/**
  * Portion stream holds a training stream for a portion.
  *
  * @param portion
  * @param stream
  */
private[alloy] case class PortionStream(portion: Double, stream: Stream[JObject])


/**
  * Object representing a fold, a hold out set and a sequence of training streams for each portion.
  *
  * @param fold
  * @param holdout
  * @param trainingStreams
  */
private[alloy] case class Fold(fold: Int, holdout: Stream[JObject], trainingStreams: Seq[PortionStream])
