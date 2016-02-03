package com.idibon.ml.predict

import scala.collection.mutable
import scala.collection.JavaConverters._

/**
  * Parent class of all prediction results.
  *
  * Assumptions:
  * - Instantiations of this class would be case classes, since case classes are nice.
  * - PredictResult is basically a placeholder til we feel out single label vs multiple label and
  * the scope of what models/objects deal with the results. Since we might just cleave off worrying
  * what high level object type we're dealing with, and instead just use the lower level case class
  * throughout.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>"
  * @param modelIdentifier a string representing a model, right now the class name.
  * @param topLabel
  * @param topLabelProbability
  * @param topLabelSignificantFeatures
  * @param topLabelMatchCount
  * @param topLabelFlags
  */
abstract class PredictResult(modelIdentifier: String,
                             topLabel: String,
                             topLabelProbability: Float,
                             topLabelSignificantFeatures: List[(String, Float)],
                             topLabelMatchCount: Int,
                             topLabelFlags: Map[PredictResultFlag.Value, Boolean]) {
  /**
    * Return the top result. This is a single label result for a document.
    *
    * @return
    */
  def getTopResult(): SingleLabelDocumentResult = {
    new SingleLabelDocumentResult(
      modelIdentifier, topLabel, topLabelProbability, topLabelSignificantFeatures,
      topLabelMatchCount, topLabelFlags)
  }

  /**
    * Return all results this represents.
    *
    * @return list of results, in order of descending probability
    */
  def getAllResults(): java.util.List[SingleLabelDocumentResult]

  /**
    * Returns the label this result is for.
    *
    * @return
    */
  def getLabel(): String = topLabel
}

/**
  * Flags for special cases that we need to be aware of during prediction.
  */
object PredictResultFlag extends Enumeration {
  type PredictResultFlag = Value
  val FORCED // e.g. used when a blacklist or whitelist rule is used.
  = Value
}

/**
  * Class for storing multiple label results for a document. Used by multi-class models.
  *
  * @param modelIdentifier a string representing a model, right now the class name.
  * @param labels
  * @param probabilities
  * @param significantFeatures
  * @param matchCounts
  * @param flags
  */
case class MultiLabelDocumentResult(modelIdentifier: String,
                                    labels: List[String],
                                    probabilities: List[Float],
                                    significantFeatures: List[List[(String, Float)]],
                                    matchCounts: List[Int],
                                    flags: List[Map[PredictResultFlag.Value, Boolean]])
  extends PredictResult(modelIdentifier,
    labels.head, probabilities.head, significantFeatures.head, matchCounts.head, flags.head)
    with mutable.Iterable[SingleLabelDocumentResult] {
  override def iterator: Iterator[SingleLabelDocumentResult] = new MultiLabelDocumentResultIterator(this)

  override def getAllResults(): java.util.List[SingleLabelDocumentResult] = {
    iterator.toList.asJava
  }
}

/**
  * Iterator to make it easy to go through individual label results.
  *
  * @param mldr
  */
class MultiLabelDocumentResultIterator(mldr: MultiLabelDocumentResult) extends Iterator[SingleLabelDocumentResult] {
  private var index = 0

  override def hasNext: Boolean = index < mldr.labels.size

  override def next(): SingleLabelDocumentResult = {
    if (!hasNext) throw new IndexOutOfBoundsException("No more results to return.")
    val label = mldr.labels(index)
    val prob = mldr.probabilities(index)
    val sigFeatures = mldr.significantFeatures(index)
    val matchCount = mldr.matchCounts(index)
    val flags = mldr.flags(index)
    index += 1
    val result = new SingleLabelDocumentResultBuilder(mldr.modelIdentifier, label)
      .addSignificantFeatures(sigFeatures)
      .setProbability(prob)
      .addToMatchCount(matchCount)
    flags.foreach({ case (flag, value) => result.setFlags(flag, value) })
    result.build()
  }
}

/**
  * Builder to make a MultiLabelDocumentResult.
  *
  * @param modelIdentifier
  * @param labelResults
  */
class MultiLabelDocumentResultBuilder(modelIdentifier: String,
                                      labelResults: Map[String, SingleLabelDocumentResultBuilder]) {

  /**
    * Alternate constructor.
    *
    * @param modelIdentifier
    * @param labels
    */
  def this(modelIdentifier: String, labels: List[String]) = {
    this(modelIdentifier,
      labels.map(label =>
        (label, new SingleLabelDocumentResultBuilder(modelIdentifier, label))).toMap)
  }

  /**
    * Sets the probability.
    *
    * @param label
    * @param probability
    * @return
    */
  def setProbability(label: String, probability: Float): MultiLabelDocumentResultBuilder = {
    labelResults(label).setProbability(probability)
    this
  }

  /**
    * Adds to the current match count.
    *
    * @param label
    * @param matchCount
    * @return
    */
  def addToMatchCount(label: String, matchCount: Int): MultiLabelDocumentResultBuilder = {
    labelResults(label).addToMatchCount(matchCount)
    this
  }

  /**
    * Sets the match count.
    *
    * @param label
    * @param matchCount
    * @return
    */
  def setMatchCount(label: String, matchCount: Int): MultiLabelDocumentResultBuilder = {
    labelResults(label).setMatchCount(matchCount)
    this
  }

  /**
    * Appends a feature to this result.
    *
    * @param label
    * @param feature
    * @return
    */
  def addSignificantFeature(label: String, feature: (String, Float)): MultiLabelDocumentResultBuilder = {
    labelResults(label).addSignificantFeature(feature)
    this
  }

  /**
    * Adds a whole list to this result.
    *
    * @param label
    * @param significantFeatures
    * @return
    */
  def addSignificantFeatures(label: String, significantFeatures: List[(String, Float)]): MultiLabelDocumentResultBuilder = {
    labelResults(label).addSignificantFeatures(significantFeatures)
    this
  }

  /**
    * Sets flags that we use for decision logic.
    *
    * @param label
    * @param name
    * @param value
    * @return
    */
  def setFlags(label: String, name: PredictResultFlag.Value, value: Boolean): MultiLabelDocumentResultBuilder = {
    labelResults(label).setFlags(name, value)
    this
  }

  /**
    * Copies from a previous multi-label document result.
    *
    * @param results
    * @return
    */
  def copyFromExistingMultiLabelDocumentResult(results: MultiLabelDocumentResult): MultiLabelDocumentResultBuilder = {
    // assert labels match
    require(results.labels.equals(this.labelResults.keys.toList), "Labels should match")
    results.labels.zipWithIndex.foreach({ case (label, resultIndex) => {
      val result = this.labelResults(label)
        .setProbability(results.probabilities(resultIndex))
        .setSignificantFeatures(results.significantFeatures(resultIndex))
        .setMatchCount(results.matchCounts(resultIndex))
      results.flags(resultIndex).foreach({ case (flag, value) => result.setFlags(flag, value) })
    }
    })
    this
  }

  /**
    * Builds a MultiLabelDocumentResult object.
    *
    * @return
    */
  def build(): MultiLabelDocumentResult = {
    val results = labelResults.map({ case (label, sldrb) => (label, sldrb.build()) })
    val order = results.map({ case (label, sldr) => (label, sldr.probability) }).toList.sortWith(_._2 < _._2)
    val baseTuple = (
      List[String](), //labels
      List[Float](), //probs
      List[List[(String, Float)]](), // sig features
      List[Int](), // match counts
      List[Map[PredictResultFlag.Value, Boolean]]()) // flags
    // create list of lists
    val x = order.foldLeft(baseTuple)({
        case (lists, (label, prob)) => {
          (label :: lists._1,
            prob :: lists._2,
            results(label).significantFeatures :: lists._3,
            results(label).matchCount :: lists._4,
            results(label).flags :: lists._5)
        }
      })
    new MultiLabelDocumentResult(this.modelIdentifier, x._1, x._2, x._3, x._4, x._5)
  }
}

/**
  * Class representing a single label prediction for a document.
  *
  * @param modelIdentifier
  * @param label
  * @param probability
  * @param significantFeatures
  * @param matchCount
  * @param flags
  */
case class SingleLabelDocumentResult(modelIdentifier: String,
                                     label: String,
                                     probability: Float,
                                     significantFeatures: List[(String, Float)],
                                     matchCount: Int,
                                     flags: Map[PredictResultFlag.Value, Boolean]) extends PredictResult(
  modelIdentifier, label, probability, significantFeatures, matchCount, flags
) {
  override def getAllResults(): java.util.List[SingleLabelDocumentResult] = List(this).asJava
}

/**
  * Class to build a single label document prediction result.
  *
  * @param modelIdentifier a string representing a model, right now the class name.
  * @param label
  */
class SingleLabelDocumentResultBuilder(modelIdentifier: String, label: String) {
  private var probability: Float = 0.0f
  private val significantFeatures = scala.collection.mutable.MutableList[(String, Float)]()
  private var matchCount: Int = 0
  private val specialFlags = mutable.HashMap[PredictResultFlag.Value, Boolean]()

  /**
    * Sets the probability.
    *
    * @param probability
    * @return
    */
  def setProbability(probability: Float): SingleLabelDocumentResultBuilder = {
    this.probability = probability
    this
  }

  /**
    * Adds to the current match count.
    *
    * @param matchCount
    * @return
    */
  def addToMatchCount(matchCount: Int): SingleLabelDocumentResultBuilder = {
    this.matchCount += matchCount
    this
  }

  /**
    * Sets the match count.
    *
    * @param matchCount
    * @return
    */
  def setMatchCount(matchCount: Int): SingleLabelDocumentResultBuilder = {
    this.matchCount = matchCount
    this
  }

  /**
    * Appends a feature to this result.
    *
    * @param feature
    * @return
    */
  def addSignificantFeature(feature: (String, Float)): SingleLabelDocumentResultBuilder = {
    this.significantFeatures += feature
    this
  }

  /**
    * Adds a whole list to this result.
    *
    * @param significantFeatures
    * @return
    */
  def addSignificantFeatures(significantFeatures: List[(String, Float)]): SingleLabelDocumentResultBuilder = {
    this.significantFeatures ++= significantFeatures
    this
  }

  /**
    * Sets a whole list to this result.
    *
    * @param significantFeatures
    * @return
    */
  def setSignificantFeatures(significantFeatures: List[(String, Float)]): SingleLabelDocumentResultBuilder = {
    this.significantFeatures.clear()
    this.significantFeatures ++= significantFeatures
    this
  }

  /**
    * Sets flags that we use for decision logic.
    *
    * @param name
    * @param value
    * @return
    */
  def setFlags(name: PredictResultFlag.Value, value: Boolean): SingleLabelDocumentResultBuilder = {
    specialFlags.put(name, value)
    this
  }

  /**
    * Copies from a previous single label result.
    *
    * @param result
    * @return
    */
  def copyFromExistingSingleLabelDocumentResult(result: SingleLabelDocumentResult): SingleLabelDocumentResultBuilder = {
    this.probability = result.probability
    this.significantFeatures ++= result.significantFeatures
    this.matchCount = result.matchCount
    result.flags.foreach(x => specialFlags.put(x._1, x._2))
    this
  }

  /**
    * Builds the SingleLabelDocumentResult object.
    *
    * @return
    */
  def build(): SingleLabelDocumentResult = {
    new SingleLabelDocumentResult(
      modelIdentifier, label, probability, significantFeatures.toList, matchCount, specialFlags.toMap)
  }

}
