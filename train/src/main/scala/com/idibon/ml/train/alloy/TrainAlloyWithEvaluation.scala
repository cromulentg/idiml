package com.idibon.ml.train.alloy

import com.idibon.ml.alloy.Alloy
import com.idibon.ml.common.Engine
import com.idibon.ml.predict.{Classification, PredictOptions}
import com.idibon.ml.train.datagenerator.json.{Annotation, Document}
import org.json4s.JsonAST.{JObject}
import scala.collection.JavaConversions._

/**
  * @author "Stefan Krawczyk <stefan@idibon.com>" on 3/2/16.
  */
/**
  * Class to encapsulate training an alloy and evaluating it.
  *
  * @param name
  * @param engine
  * @param trainer
  * @param dataSet
  * @param trainingSummaryCreator
  */
class TrainAlloyWithEvaluation(name: String,
                               engine: Engine,
                               trainer: AlloyTrainer,
                               dataSet: TrainingDataSet,
                               trainingSummaryCreator: AlloyEvaluator) {
  val uuidStrToDouble = dataSet.info.labelToDouble.map(x => (x._1.uuid.toString, x._2))

  /**
    * Method to train, evaluate and create a training summary.
    *
    * @param labelsAndRules
    * @param config
    * @return training summary based on the evaluation set of the data set.
    */
  def apply(labelsAndRules: JObject,
            config: Option[JObject]) = {
    // train
    val trained = trainer.trainAlloy(name, dataSet.train, labelsAndRules, config)
    // get thresholds
    val thresholds = trained.getSuggestedThresholds().map({
      case (label, thresh) => (label.uuid.toString, thresh.toFloat)
    }).toMap
    // evaluate
    val evalPoints = evaluate(trained, thresholds)
    // create summary
    trainingSummaryCreator.createTrainingSummary(
      engine, evalPoints, uuidStrToDouble, name, dataSet.info.portion)
  }

  /**
    * Evaluate the trained alloy and create raw data points.
    *
    * @param trained
    * @param thresholds
    * @return
    */
  def evaluate(trained: Alloy[Classification],
               thresholds: Map[String, Float]): Seq[EvaluationDataPoint] = {
    dataSet.test().map(doc => {
      val goldSet = getGoldSet(doc)
      goldSet.isEmpty match {
        case true => None
        case false => {
          val predicted = trained.predict(doc, PredictOptions.DEFAULT)
          // create eval data point
          val evaluationDataPoint = trainingSummaryCreator.createEvaluationDataPoint(
            uuidStrToDouble, goldSet, predicted, thresholds)
          Some(evaluationDataPoint)
        }
      }
    }).toSeq.collect({ case Some(evalPoint) => evalPoint })
  }

  /**
    * Gets the gold value from the evaluation set.
    *
    * @param jsValue
    * @return
    */
  def getGoldSet(jsValue: JObject): Map[String, Seq[Annotation]] = {
    implicit val formats = org.json4s.DefaultFormats
    val document = jsValue.extract[Document]
    document.annotations
      .filter({ case (annot) => annot.isPositive })
      .map({ case (annot) => (annot.label.name, Seq(annot)) })
      .toMap
  }
}
